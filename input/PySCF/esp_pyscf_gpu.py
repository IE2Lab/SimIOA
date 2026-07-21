"""
Compute normalized cation/anion-solvent affinity (alpha_S, beta_S) using PySCF.

Reproduces the methodology of Li et al., Nature Energy (2025):
    "Unified affinity paradigm for the rational design of
     high-efficiency lithium metal electrolytes"

Key advantage over cubegen + RegularGridInterpolator:
    ESP is evaluated **analytically** at every surface vertex,
    eliminating the interpolation artifacts that cause Si-molecule overflow.

Requirements:
    pip install pyscf numpy scipy scikit-image
    pip install geometric              # optional: geometry optimisation
    pip install rdkit                  # optional: SMILES input
    pip install gpu4pyscf-cuda12x      # optional: GPU acceleration (or -cuda11x)
    #   + cupy matching your CUDA version

GPU acceleration (gpu4pyscf):
    Pass use_gpu=True (or 'auto') to run the SCF, geometry optimisation,
    density grid, and ESP integrals on the GPU.  The SCF/optimisation
    speed-up is large and reliable; each grid step (density, ESP) attempts
    its GPU kernel and falls back to CPU individually if that kernel is not
    present in your gpu4pyscf build, so results are always produced.

        # Python
        results = compute_affinities(xyz_file='teos_opt.xyz', use_gpu=True)
        results = compute_affinities(smiles='COCCOC', opt_geom=True,
                                     use_gpu='auto')   # GPU if present
        # CLI
        python esp_pyscf.py --xyz teos_opt.xyz --gpu
        python esp_pyscf.py --smiles "COCCOC" --opt --gpu-auto

Usage (Python):
    from esp_pyscf import compute_affinities

    # From a Gaussian-optimised XYZ file (PySCF runs its own SP on that geometry)
    results = compute_affinities(xyz_file='dme_opt.xyz')

    # Save ESP as cube file for visualisation in VESTA / VMD
    results = compute_affinities(xyz_file='dme_opt.xyz',
                                  save_cube='dme_esp.cube')

    # From SMILES (requires rdkit)
    results = compute_affinities(smiles='COCCOC')

    # Print main results
    print(f"alpha_S = {results['alpha_s']:.2f} kcal/mol")
    print(f"beta_S  = {results['beta_s']:.2f} kcal/mol")

    # Per-atom breakdown
    for row in results['per_atom']:
        print(row)

Usage (CLI):
    python esp_pyscf.py --xyz dme_opt.xyz
    python esp_pyscf.py --smiles "COCCOC" --opt --cube dme_esp.cube
    python esp_pyscf.py --xyz teos.xyz --spacing 0.15  # finer grid for Si
    python esp_pyscf.py --smiles "COCCOC" --opt --opt-xyz dme_opt.xyz  # save opt geom
    python esp_pyscf.py --smiles "COCCOC" --opt --opt-xyz            # auto-named XYZ

Return dict keys
----------------
Main results:
    alpha_s, beta_s          float  kcal/mol
Global surface statistics:
    esp_min_kcal             float  global minimum ESP on surface (kcal/mol)
    esp_max_kcal             float  global maximum ESP on surface (kcal/mol)
    Vbar_neg_kcal            float  area-weighted mean ESP over S⁻ (kcal/mol)
    Vbar_pos_kcal            float  area-weighted mean ESP over S⁺ (kcal/mol)
    S_neg, S_pos, S_tot      float  surface areas in Bohr²
    V_mol_bohr3              float  molecular volume in Bohr³
Per-atom breakdown:
    per_atom                 list of dicts, one per atom:
        index                int    atom index (0-based)
        symbol               str    element symbol
        S_atom_bohr2         float  surface area belonging to this atom (Bohr²)
        esp_min_kcal         float  min ESP on this atom's surface patch
        esp_max_kcal         float  max ESP on this atom's surface patch
        esp_mean_kcal        float  area-weighted mean ESP on this atom's patch
Mesh diagnostics:
    n_verts, n_tris          int    mesh size
    n_clipped                int    number of vertices that hit the ESP safety clip
"""

import sys
import struct
import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes

# ── Constants ────────────────────────────────────────────────────────────
BOHR2ANG     = 0.529177210903
ANG2BOHR     = 1.0 / BOHR2ANG
HARTREE2KCAL = 627.5094740631

# Bondi van der Waals radii (Angstrom) — same set used in the paper
BONDI_RADII = {
    "H":  1.20, "He": 1.40, "Li": 1.82, "Be": 1.53,
    "B":  1.92, "C":  1.70, "N":  1.55, "O":  1.52,
    "F":  1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
    "Al": 1.84, "Si": 2.10, "P":  1.80, "S":  1.80,
    "Cl": 1.75, "Ar": 1.88, "K":  2.75, "Ca": 2.31,
    "Br": 1.85, "I":  1.98,
}


# ── GPU backend helpers ──────────────────────────────────────────────────
def gpu_available():
    """Return True if gpu4pyscf and a working CUDA device are importable."""
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        import gpu4pyscf          # noqa: F401
        return True
    except Exception:
        return False


def _to_numpy(a):
    """Convert a cupy array (or anything array-like) to a numpy array."""
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    except ImportError:
        pass
    return np.asarray(a)


def _resolve_gpu(use_gpu):
    """
    Interpret the `use_gpu` request.

    use_gpu=True   → require GPU; raise if unavailable
    use_gpu='auto' → use GPU if available, else CPU (prints which)
    use_gpu=False  → CPU
    """
    if use_gpu is True:
        if not gpu_available():
            raise RuntimeError(
                "use_gpu=True but gpu4pyscf / CUDA is not available. "
                "Install with:  pip install gpu4pyscf-cuda12x  (or -cuda11x)")
        return True
    if isinstance(use_gpu, str) and use_gpu.lower() == "auto":
        avail = gpu_available()
        print(f"  [gpu] auto-detect: {'GPU found' if avail else 'no GPU, using CPU'}")
        return avail
    return False


# ── Molecule construction ────────────────────────────────────────────────
def build_molecule(smiles=None, xyz_file=None, xyz_string=None,
                   basis="6-311+g(d,p)", charge=0, spin=0, verbose=0):
    """
    Build a PySCF Mole from a geometry source.

    Parameters
    ----------
    smiles    : SMILES string (requires rdkit; used for 3-D embedding + MMFF pre-opt)
    xyz_file  : path to an XYZ file  (Gaussian writes these natively)
    xyz_string: inline "Symbol  x  y  z" block, one atom per line (Angstrom)
    basis     : basis set name accepted by PySCF (default matches the paper)
    charge, spin : net charge and 2S (unpaired electrons)
    verbose   : PySCF verbosity level (0 = silent)

    Notes
    -----
    When providing a Gaussian-optimised geometry via xyz_file / xyz_string,
    PySCF will run a fresh DFT single-point on that geometry.  The level of
    theory is controlled by the `xc` and `basis` arguments of run_dft() /
    compute_affinities(), not by whatever Gaussian used.
    """
    from pyscf import gto

    if smiles is not None:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, randomSeed=42) == -1:
            raise RuntimeError("RDKit 3-D embedding failed")
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
        conf = m.GetConformer()
        lines = []
        for i in range(m.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            s = m.GetAtomWithIdx(i).GetSymbol()
            lines.append(f"{s}  {p.x:.8f}  {p.y:.8f}  {p.z:.8f}")
        atom_str = "\n".join(lines)

    elif xyz_file is not None:
        with open(xyz_file) as fh:
            raw = fh.readlines()
        # Strip optional natom + comment header (standard XYZ format)
        try:
            natom = int(raw[0].strip())
            atom_str = "".join(raw[2 : 2 + natom])
        except ValueError:
            atom_str = "".join(raw)

    elif xyz_string is not None:
        atom_str = xyz_string

    else:
        raise ValueError("Provide one of: smiles, xyz_file, xyz_string")

    mol = gto.Mole()
    mol.atom   = atom_str
    mol.basis  = basis
    mol.charge = charge
    mol.spin   = spin
    mol.verbose = verbose
    mol.build()
    return mol


# ── DFT single-point (or optimisation) ──────────────────────────────────
def run_dft(mol, xc="b3lyp", disp="d3bj", opt_geom=False, use_gpu=False):
    """
    Run DFT and return the converged mean-field object.

    Parameters
    ----------
    mol      : pyscf.gto.Mole
    xc       : XC functional string (default b3lyp)
    disp     : dispersion correction label — 'd3bj', 'd3zero', 'd3', or None
               'd3bj'  ≈ GD3BJ used by the paper
    opt_geom : if True, optimise geometry first with geomeTRIC
               (pip install geometric)
    use_gpu  : if True, build the mean-field object with gpu4pyscf so the
               SCF *and* the geometry optimisation run on the GPU.
               (Resolve with _resolve_gpu() before calling.)

    Notes
    -----
    * The SCF and (if requested) geometry optimisation are the most
      expensive steps for medium/large molecules; gpu4pyscf accelerates
      both by 1-2 orders of magnitude on a modern GPU.
    * No grid specification is required here.  The DFT quadrature grid
      (mf.grids) is only used for the XC energy during the SCF; it plays
      no role in the ESP evaluation, which is done analytically.
    """
    if use_gpu:
        from gpu4pyscf import dft as _dft
        backend = "GPU (gpu4pyscf)"
    else:
        from pyscf import dft as _dft
        backend = "CPU (pyscf)"
    print(f"  Backend: {backend}")

    def _make_mf(m):
        mf = _dft.RKS(m) if m.spin == 0 else _dft.UKS(m)
        mf.xc       = xc
        mf.grids.level = 4    # fine quadrature for accurate XC
        mf.conv_tol = 1e-10
        if disp is not None:
            try:
                mf.disp = disp
            except Exception:
                print(f"  [warn] dispersion '{disp}' not available; skipping")
        return mf

    if opt_geom:
        from pyscf.geomopt.geometric_solver import optimize
        print("  Running geometry optimisation (geomeTRIC"
              f"{', GPU gradients' if use_gpu else ''})...")
        mf_pre = _make_mf(mol)
        mol = optimize(mf_pre)          # geomeTRIC works with gpu4pyscf mf too

    mf = _make_mf(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


# ── Analytical ESP evaluation ────────────────────────────────────────────
def _evaluate_esp_gpu(mol, dm, coords_bohr, batch_size):
    """
    GPU electronic + nuclear ESP using gpu4pyscf's int1e_grids kernel.

    Raises if the installed gpu4pyscf lacks the grid-integral entry point,
    so the caller can fall back to CPU.
    """
    import cupy as cp
    # int1e_grids computes Σ_{μν} dm_{μν} (μ| 1/|r−r_g| |ν) directly on GPU,
    # contracting with the density matrix on the fly (memory-efficient).
    from gpu4pyscf.gto.int3c1e import int1e_grids

    dm_gpu = cp.asarray(_to_numpy(dm))
    coords = cp.asarray(coords_bohr)
    npts   = coords.shape[0]
    v_esp  = cp.zeros(npts)

    # Nuclear part on GPU
    charges = cp.asarray([mol.atom_charge(i) for i in range(mol.natm)])
    Rs      = cp.asarray(mol.atom_coords())            # Bohr
    for i0 in range(0, npts, batch_size):
        i1  = min(i0 + batch_size, npts)
        blk = coords[i0:i1]
        d   = cp.linalg.norm(blk[:, None, :] - Rs[None, :, :], axis=2)
        d   = cp.maximum(d, 1e-12)
        v_esp[i0:i1] += (charges[None, :] / d).sum(axis=1)

    # Electronic part on GPU (batched)
    for i0 in range(0, npts, batch_size):
        i1  = min(i0 + batch_size, npts)
        blk = coords[i0:i1]
        v_elec = int1e_grids(mol, blk, dm=dm_gpu)      # (nbatch,)
        v_esp[i0:i1] -= cp.asarray(v_elec)

    return cp.asnumpy(v_esp)


def evaluate_esp(mol, dm, coords_bohr, batch_size=500, use_gpu=False):
    """
    Evaluate the total electrostatic potential analytically at arbitrary points.

        V(r) = Σ_A  Z_A / |r − R_A|          ← nuclear
             − Σ_{μν}  P_{μν} (μν | 1/r)     ← electronic

    The CPU path uses PySCF's three-centre int3c2e integrals via a "fake
    molecule" of unit Gaussians placed at the evaluation points.  The GPU
    path uses gpu4pyscf's int1e_grids kernel.  No grid interpolation is
    involved either way — ESP values are exact given the wavefunction.

    Parameters
    ----------
    mol         : pyscf.gto.Mole
    dm          : (nao, nao) one-particle density matrix
    coords_bohr : (N, 3) evaluation points in Bohr
    batch_size  : number of points per integral batch (memory control)
    use_gpu     : if True, attempt the gpu4pyscf path; on any failure it
                  prints a warning and falls back to the CPU path.

    Returns
    -------
    v_esp : (N,) numpy array in Hartree/e
    """
    coords_bohr = np.ascontiguousarray(coords_bohr)

    if use_gpu:
        try:
            return _evaluate_esp_gpu(mol, dm, coords_bohr, batch_size)
        except Exception as exc:
            print(f"  [gpu] ESP integrals unavailable on GPU ({exc}); "
                  f"falling back to CPU for this step")

    # ---- CPU path ----
    from pyscf import gto as _gto, df

    dm_np = _to_numpy(dm)
    npts  = len(coords_bohr)
    v_esp = np.zeros(npts)

    # Nuclear
    for iatm in range(mol.natm):
        Z = mol.atom_charge(iatm)
        R = mol.atom_coord(iatm)          # Bohr
        dist = np.linalg.norm(coords_bohr - R[None, :], axis=1)
        dist = np.maximum(dist, 1e-12)
        v_esp += Z / dist

    # Electronic (batched three-centre integrals)
    for i0 in range(0, npts, batch_size):
        i1      = min(i0 + batch_size, npts)
        batch   = np.ascontiguousarray(coords_bohr[i0:i1])
        fakemol = _gto.fakemol_for_charges(batch)
        j3c     = df.incore.aux_e2(mol, fakemol, intor="int3c2e")
        v_esp[i0:i1] -= np.einsum("ijp,ij->p", j3c, dm_np)

    return v_esp


# ── Density on a regular grid (for marching cubes only) ──────────────────
def _density_on_grid(mol, dm, grid_1d, use_gpu=False):
    """
    Evaluate electron density on a regular 3-D grid (x, y, z in Bohr).

    This grid is used ONLY to locate the isodensity surface via marching
    cubes.  It is NOT used for ESP values — those are computed analytically
    at the marching-cubes vertex positions.

    Parameters
    ----------
    use_gpu : if True, use gpu4pyscf's numint (cupy) to evaluate AO and ρ;
              falls back to CPU on any failure.

    Returns rho_3d : (nx, ny, nz) numpy array.
    """
    x, y, z  = grid_1d
    nx, ny, nz = len(x), len(y), len(z)
    rho_3d   = np.empty((nx, ny, nz), dtype=np.float64)
    yz = np.array(np.meshgrid(y, z, indexing="ij")).reshape(2, -1).T

    if use_gpu:
        try:
            import cupy as cp
            from gpu4pyscf.dft import numint as gni
            ni     = gni.NumInt()
            dm_gpu = cp.asarray(_to_numpy(dm))
            for ix in range(nx):
                coords = cp.asarray(
                    np.column_stack([np.full(len(yz), x[ix]), yz]))
                ao  = ni.eval_ao(mol, coords)
                rho = ni.eval_rho(mol, ao, dm_gpu)
                rho_3d[ix] = cp.asnumpy(rho).reshape(ny, nz)
            return rho_3d
        except Exception as exc:
            print(f"  [gpu] density-on-grid unavailable on GPU ({exc}); "
                  f"falling back to CPU for this step")

    # ---- CPU path ----
    from pyscf.dft import numint as ni
    dm_np = _to_numpy(dm)
    for ix in range(nx):
        coords = np.column_stack([np.full(len(yz), x[ix]), yz])
        ao     = ni.eval_ao(mol, coords)
        rho_3d[ix] = ni.eval_rho(mol, ao, dm_np).reshape(ny, nz)

    return rho_3d


# ── Isodensity surface ───────────────────────────────────────────────────
def extract_isosurface(mol, dm, iso=1e-3, padding=6.0, spacing=0.20,
                       use_gpu=False):
    """
    Extract the ρ = iso surface via marching cubes.

    How this relates to ESP
    -----------------------
    1. We evaluate ρ on a Cartesian grid (spacing ~0.20 Bohr).
    2. marching_cubes() finds where ρ = 0.001 and returns vertex
       coordinates in Bohr.
    3. We then call evaluate_esp() at those exact vertex coordinates
       using the analytical integral engine — no interpolation.

    So the density grid only needs to be fine enough to locate the
    surface accurately, not fine enough to reproduce a steep Coulombic
    potential (which is what caused the Si overflow in the old pipeline).

    Parameters
    ----------
    spacing : float
        Grid spacing in Bohr.  0.20 is a good default.
        For Si-containing or other heavy-atom molecules use 0.15.

    Returns
    -------
    verts : (V, 3) Bohr  — surface vertex positions
    faces : (F, 3)       — triangle connectivity
    """
    coords = mol.atom_coords()          # Bohr
    lo = coords.min(axis=0) - padding
    hi = coords.max(axis=0) + padding

    x = np.arange(lo[0], hi[0] + spacing, spacing)
    y = np.arange(lo[1], hi[1] + spacing, spacing)
    z = np.arange(lo[2], hi[2] + spacing, spacing)

    print(f"  Density grid: {len(x)}×{len(y)}×{len(z)} "
          f"= {len(x)*len(y)*len(z):,} pts  (Δ = {spacing:.2f} Bohr)")

    rho = _density_on_grid(mol, dm, (x, y, z), use_gpu=use_gpu)

    step  = (float(x[1]-x[0]), float(y[1]-y[0]), float(z[1]-z[0]))
    verts_idx, faces, _, _ = marching_cubes(rho, level=iso, spacing=step)
    verts = verts_idx + lo[None, :]     # grid-origin shift → real Bohr coords

    print(f"  Isosurface: {len(verts):,} vertices, {len(faces):,} triangles")
    return verts, faces


# ── Cube file writing ────────────────────────────────────────────────────
def _write_cube(mol, outfile, origin, spacing, data_3d, comment1, comment2):
    """
    Low-level Gaussian-format .cube writer.

    Parameters
    ----------
    mol      : pyscf.gto.Mole (for the atom block)
    outfile  : output path
    origin   : (3,) grid origin in Bohr
    spacing  : grid spacing in Bohr (isotropic)
    data_3d  : (nx, ny, nz) volumetric data
    comment1, comment2 : the two header comment lines

    Notes
    -----
    Cube-format data ordering: outer loop over x, middle over y, inner over z
    (z varies fastest).  A newline is emitted at the end of each z-row, which
    is the strictly-conformant layout that all readers (ChimeraX, VMD, VESTA,
    Multiwfn) accept.
    """
    nx, ny, nz = data_3d.shape
    with open(outfile, "w") as fh:
        fh.write(f" {comment1}\n")
        fh.write(f" {comment2}\n")
        # natom + origin (positive natom, positive voxel counts ⇒ Bohr units)
        fh.write(f"  {mol.natm:4d}  {origin[0]:12.6f}  "
                 f"{origin[1]:12.6f}  {origin[2]:12.6f}\n")
        # voxel axis vectors
        fh.write(f"  {nx:4d}  {spacing:12.6f}  {0.0:12.6f}  {0.0:12.6f}\n")
        fh.write(f"  {ny:4d}  {0.0:12.6f}  {spacing:12.6f}  {0.0:12.6f}\n")
        fh.write(f"  {nz:4d}  {0.0:12.6f}  {0.0:12.6f}  {spacing:12.6f}\n")
        # atom block: Znuc, charge, x, y, z
        for iatm in range(mol.natm):
            Z = mol.atom_charge(iatm)
            R = mol.atom_coord(iatm)
            fh.write(f"  {int(Z):4d}  {float(Z):12.6f}  "
                     f"{R[0]:12.6f}  {R[1]:12.6f}  {R[2]:12.6f}\n")
        # volumetric data: newline at end of every z-row, max 6 values/line
        for ix in range(nx):
            for iy in range(ny):
                col = data_3d[ix, iy, :]
                for iz in range(0, nz, 6):
                    chunk = col[iz:iz+6]
                    fh.write("".join(f"{v:13.5E}" for v in chunk) + "\n")


def _make_grid(mol, padding, spacing):
    """Return (origin, (x,y,z) axes) for a padded box around the molecule (Bohr)."""
    coords = mol.atom_coords()
    lo = coords.min(axis=0) - padding
    hi = coords.max(axis=0) + padding
    x = np.arange(lo[0], hi[0] + spacing, spacing)
    y = np.arange(lo[1], hi[1] + spacing, spacing)
    z = np.arange(lo[2], hi[2] + spacing, spacing)
    return lo, (x, y, z)


def save_cubes(mol, dm, prefix, padding=5.0, spacing=0.20,
               write_density=True, write_esp=True, use_gpu=False):
    """
    Write density and/or ESP cube files on an IDENTICAL grid.

    For the standard ESP-mapped-on-density visualization in ChimeraX / VMD
    you need BOTH cubes, and they must share the same grid so the ESP values
    line up with the density surface.  This function guarantees that.

    Parameters
    ----------
    mol      : pyscf.gto.Mole
    dm       : density matrix
    prefix   : output filename stem; produces '<prefix>_density.cube'
               and '<prefix>_esp.cube'
    padding  : box padding around the molecule in Bohr
    spacing  : grid spacing in Bohr (0.20 for visualization; 0.10–0.15 finer)
    write_density, write_esp : toggle each output

    Returns
    -------
    dict of written file paths, e.g. {'density': ..., 'esp': ...}
    """
    origin, (x, y, z) = _make_grid(mol, padding, spacing)
    nx, ny, nz = len(x), len(y), len(z)
    print(f"  Cube grid: {nx}×{ny}×{nz} = {nx*ny*nz:,} pts "
          f"(Δ = {spacing:.2f} Bohr)")

    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    written = {}

    if write_density:
        rho_3d = _density_on_grid(mol, dm, (x, y, z), use_gpu=use_gpu)
        out = f"{prefix}_density.cube"
        _write_cube(mol, out, origin, spacing, rho_3d,
                    "Electron density written by esp_pyscf.py",
                    "Density (e/Bohr^3) — use isovalue 0.001 for the vdW surface")
        written["density"] = out
        print(f"  → {out}   (ρ range [{rho_3d.min():.2e}, {rho_3d.max():.2e}] e/Bohr³)")

    if write_esp:
        esp_flat = evaluate_esp(mol, dm, pts, batch_size=500, use_gpu=use_gpu)
        esp_3d   = esp_flat.reshape(nx, ny, nz)
        out = f"{prefix}_esp.cube"
        _write_cube(mol, out, origin, spacing, esp_3d,
                    "Electrostatic potential written by esp_pyscf.py",
                    "ESP (Hartree/e) — map as COLOR onto the density surface")
        written["esp"] = out
        print(f"  → {out}   (ESP range [{esp_flat.min()*HARTREE2KCAL:.1f}, "
              f"{esp_flat.max()*HARTREE2KCAL:.1f}] kcal/mol)")

    print("\n  Visualisation (both cubes share the same grid):")
    print("  ─ ChimeraX ─────────────────────────────────────────────")
    print(f"      open {prefix}_density.cube")
    print(f"      open {prefix}_esp.cube")
    print(f"      volume #1 level 0.001 style surface")
    print(f"      color sample #1 map #2 palette red:white:blue range -0.05,0.05")
    print("  ─ VMD ──────────────────────────────────────────────────")
    print(f"      Load {prefix}_density.cube as a New Molecule,")
    print(f"      then File ▸ Load Data Into Molecule ▸ {prefix}_esp.cube")
    print(f"      Rep: Isosurface, Draw=Solid Surface, Isovalue=0.001,")
    print(f"      Coloring Method=Volume (select the ESP field, vol id 1)")
    return written


# Backward-compatible single-file ESP writer
def save_esp_cube(mol, dm, outfile, padding=5.0, spacing=0.20, use_gpu=False):
    """Write only the ESP cube (kept for backward compatibility).

    NOTE: an ESP cube alone has no isosurface to display.  Prefer
    save_cubes(...) which also writes the density cube needed to render
    the surface that the ESP colours.
    """
    origin, (x, y, z) = _make_grid(mol, padding, spacing)
    nx, ny, nz = len(x), len(y), len(z)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    esp_3d = evaluate_esp(mol, dm, pts, batch_size=500,
                          use_gpu=use_gpu).reshape(nx, ny, nz)
    _write_cube(mol, outfile, origin, spacing, esp_3d,
                "Electrostatic potential written by esp_pyscf.py",
                "ESP (Hartree/e) — map as COLOR onto a density surface")
    print(f"  Cube saved: {outfile}")


# ── Save geometry as XYZ ─────────────────────────────────────────────────
def save_xyz(mol, outfile, comment=""):
    """
    Write the molecule's current geometry to a standard XYZ file (Angstrom).

    Useful for saving the DFT-optimised structure after run_dft(..., opt_geom=True)
    so it can be reused (e.g. fed back in via xyz_file=) or inspected in a viewer.

    Parameters
    ----------
    mol     : pyscf.gto.Mole  (coordinates are read in Bohr and converted to Å)
    outfile : output path, e.g. 'dme_opt.xyz'
    comment : text for the XYZ comment line (line 2)
    """
    coords = mol.atom_coords() * BOHR2ANG        # Bohr → Angstrom
    with open(outfile, "w") as fh:
        fh.write(f"{mol.natm}\n")
        fh.write(f"{comment}\n")
        for i in range(mol.natm):
            s = mol.atom_symbol(i)
            x, y, z = coords[i]
            fh.write(f"{s:<3s} {x:15.8f} {y:15.8f} {z:15.8f}\n")
    print(f"  Geometry saved: {outfile}  ({mol.natm} atoms, Å)")


# ── Core: alpha_S / beta_S with full per-atom breakdown ─────────────────
def compute_alpha_beta(mol, dm, verts, faces, esp_clip=0.15, use_gpu=False):
    """
    Compute α_S, β_S plus per-atom surface/ESP statistics.

    Parameters
    ----------
    mol, dm     : PySCF molecule and density matrix
    verts       : (V, 3) surface vertex coordinates in Bohr
    faces       : (F, 3) triangle connectivity
    esp_clip    : safety clamp on |ESP| in Hartree (default 0.15 ≈ 94 kcal/mol)
                  With analytical ESP this should very rarely activate.
    use_gpu     : evaluate the surface ESP on the GPU (with CPU fallback)

    Returns
    -------
    dict — see module docstring for all keys.
    """

    # ── 1. Analytical ESP at every surface vertex ──────────────────────
    print("  Evaluating analytical ESP at surface vertices ...")
    esp = evaluate_esp(mol, dm, verts, batch_size=400, use_gpu=use_gpu)

    n_bad  = int(np.sum(~np.isfinite(esp)))
    n_clip = int(np.sum(np.abs(esp) > esp_clip))
    if n_bad:
        print(f"  ⚠ {n_bad} non-finite values → zeroed")
    if n_clip:
        print(f"  ⚠ {n_clip}/{len(esp)} vertices exceed ±{esp_clip} Ha → clipped")
    esp = np.where(np.isfinite(esp), esp, 0.0)
    esp = np.clip(esp, -esp_clip, esp_clip)

    # ── 2. Atom assignment: w_A = 1 − |r − R_A| / R_A  (SI eq 1) ──────
    atom_coords = mol.atom_coords()                       # (natm, 3) Bohr
    symbols     = [mol.atom_symbol(i) for i in range(mol.natm)]
    radii       = np.array([BONDI_RADII.get(s, 1.70) * ANG2BOHR
                            for s in symbols])            # Bohr

    dists_va = cdist(verts, atom_coords)                  # (V, natm)
    w        = 1.0 - dists_va / radii[None, :]           # (V, natm)
    owner    = np.argmax(w, axis=1)                       # (V,) atom index

    # ── 3. Triangle geometry ────────────────────────────────────────────
    v0, v1, v2   = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross        = np.cross(v1 - v0, v2 - v0)
    areas        = 0.5 * np.linalg.norm(cross, axis=1)   # (F,) Bohr²

    # ESP and dominant atom at each triangle centroid
    esp_tri    = (esp[faces[:, 0]] + esp[faces[:, 1]] + esp[faces[:, 2]]) / 3.0
    owner_tri  = np.array([
        np.bincount(owner[faces[i]], minlength=mol.natm).argmax()
        for i in range(len(faces))
    ])                                                     # (F,) atom index

    # ── 4. Global surface integrals ─────────────────────────────────────
    pos = esp_tri > 0
    neg = esp_tri < 0

    I_pos = float(np.sum(esp_tri[pos] * areas[pos]))      # Hartree·Bohr²
    I_neg = float(np.sum(esp_tri[neg] * areas[neg]))

    S_pos = float(np.sum(areas[pos]))
    S_neg = float(np.sum(areas[neg]))
    S_tot = float(np.sum(areas))

    # ── 5. Molecular volume (divergence theorem) ────────────────────────
    centroids = (v0 + v1 + v2) / 3.0
    V_mol     = abs(float(np.sum(centroids[:, 0] * cross[:, 0]))) / 6.0

    # ── 6. α_S, β_S ─────────────────────────────────────────────────────
    eps      = 1e-30
    Vbar_neg = I_neg / max(S_neg, eps)
    Vbar_pos = I_pos / max(S_pos, eps)

    alpha_ha = Vbar_neg * np.sqrt(max(S_neg, 0)) * V_mol ** (-1.0/3)
    beta_ha  = Vbar_pos * np.sqrt(max(S_pos, 0)) * V_mol ** (-1.0/3)

    alpha_kcal = float(alpha_ha * HARTREE2KCAL)
    beta_kcal  = float(beta_ha  * HARTREE2KCAL)

    # ── 7. Per-atom statistics ───────────────────────────────────────────
    # For each atom: surface area of its patch, and ESP min/max/mean on that patch.
    per_atom = []
    for iatm in range(mol.natm):
        mask = owner_tri == iatm
        if not np.any(mask):
            per_atom.append({
                "index": iatm, "symbol": symbols[iatm],
                "S_atom_bohr2": 0.0,
                "esp_min_kcal": None, "esp_max_kcal": None,
                "esp_mean_kcal": None,
            })
            continue

        a_atom   = areas[mask]
        e_atom   = esp_tri[mask]
        S_a      = float(a_atom.sum())
        esp_mean = float((e_atom * a_atom).sum() / S_a) * HARTREE2KCAL

        per_atom.append({
            "index":        iatm,
            "symbol":       symbols[iatm],
            "S_atom_bohr2": S_a,
            "esp_min_kcal": float(e_atom.min() * HARTREE2KCAL),
            "esp_max_kcal": float(e_atom.max() * HARTREE2KCAL),
            "esp_mean_kcal": esp_mean,
        })

    return {
        # ── Main results ──────────────────────────────────────────
        "alpha_s":       alpha_kcal,
        "beta_s":        beta_kcal,
        # ── Global surface ESP statistics ─────────────────────────
        "esp_min_kcal":  float(esp.min()  * HARTREE2KCAL),
        "esp_max_kcal":  float(esp.max()  * HARTREE2KCAL),
        "Vbar_neg_kcal": float(Vbar_neg   * HARTREE2KCAL),
        "Vbar_pos_kcal": float(Vbar_pos   * HARTREE2KCAL),
        # ── Surface areas (Bohr²) ─────────────────────────────────
        "S_neg":         S_neg,
        "S_pos":         S_pos,
        "S_tot":         S_tot,
        # ── Molecular volume (Bohr³) ──────────────────────────────
        "V_mol_bohr3":   V_mol,
        # ── Per-atom breakdown ────────────────────────────────────
        "per_atom":      per_atom,
        # ── Mesh diagnostics ──────────────────────────────────────
        "n_verts":       len(verts),
        "n_tris":        len(faces),
        "n_clipped":     n_clip,
    }


# ── Public entry point ───────────────────────────────────────────────────
def compute_affinities(
    smiles=None,
    xyz_file=None,
    xyz_string=None,
    basis="6-311+g(d,p)",
    xc="b3lyp",
    disp="d3bj",
    charge=0,
    spin=0,
    opt_geom=False,
    iso=1e-3,
    grid_spacing=0.20,
    padding=6.0,
    esp_clip=0.15,
    save_cube=None,
    save_cube_prefix=None,
    save_opt_xyz=None,
    use_gpu=False,
    verbose=0,
):
    """
    Full pipeline: geometry → DFT → isosurface → analytical ESP → α_S, β_S.

    Parameters
    ----------
    smiles / xyz_file / xyz_string
        Geometry source — provide exactly one.
        xyz_file accepts a standard XYZ file, e.g. from Gaussian's %chk
        → formchk → Open Babel, or directly from Gaussian's opt output.
    basis       : basis set (default 6-311+g(d,p), matching the paper)
    xc          : XC functional (default b3lyp)
    disp        : dispersion label (default d3bj ≈ paper's GD3)
    charge, spin: net charge and 2S
    opt_geom    : run PySCF geometry optimisation before SP (needs geometric)
    iso         : isodensity level e/Bohr³ (default 0.001)
    grid_spacing: Bohr spacing for the density grid used by marching cubes
                  (default 0.20; use 0.15 for Si/heavy-atom molecules)
    padding     : bounding-box padding in Bohr (default 6.0)
    esp_clip    : safety clamp on |ESP| in Hartree (default 0.15)
    save_cube        : if given, path to write an ESP-only cube file
                       (kept for backward compatibility; an ESP cube alone
                       has no isosurface to display)
    save_cube_prefix : if given, write BOTH '<prefix>_density.cube' and
                       '<prefix>_esp.cube' on the same grid — this is what
                       you want for ChimeraX / VMD ESP-mapped-on-density plots
    save_opt_xyz     : if given, path to write the final geometry as an XYZ
                       file.  Most useful together with opt_geom=True to save
                       the DFT-optimised structure (in Angstrom).
    use_gpu          : True / 'auto' / False.  When enabled (and gpu4pyscf +
                       CUDA are available) the SCF, geometry optimisation,
                       density grid and ESP integrals run on the GPU.
                       'auto' uses the GPU if present, otherwise CPU.
                       Each grid step falls back to CPU individually if its
                       GPU kernel is missing in your gpu4pyscf build.
    verbose          : PySCF verbosity (0 = silent)

    Returns
    -------
    dict  — see module docstring for all keys.
    """
    print("=" * 64)
    print("  α_S / β_S  via analytical ESP   (PySCF / gpu4pyscf)")
    print("=" * 64)

    # 0. Resolve GPU request up-front
    gpu = _resolve_gpu(use_gpu)

    # 1. Molecule
    print("\n[1/4] Building molecule ...")
    mol = build_molecule(smiles=smiles, xyz_file=xyz_file,
                         xyz_string=xyz_string, basis=basis,
                         charge=charge, spin=spin, verbose=verbose)
    print(f"  Atoms: {mol.natm}   AOs: {mol.nao}   Basis: {basis}")
    for i in range(mol.natm):
        s = mol.atom_symbol(i)
        c = mol.atom_coord(i) * BOHR2ANG
        print(f"    {s:>2s}  {c[0]:10.6f} {c[1]:10.6f} {c[2]:10.6f}  Å")

    # 2. DFT
    print(f"\n[2/4] DFT single-point "
          f"({xc}/{basis}{', ' + disp if disp else ''}) ...")
    mf = run_dft(mol, xc=xc, disp=disp, opt_geom=opt_geom, use_gpu=gpu)
    dm = _to_numpy(mf.make_rdm1())      # bring density matrix to CPU (numpy)
    # Use the (possibly optimised) geometry from the mean-field object
    mol = mf.mol
    print(f"  E_total = {float(mf.e_tot):.10f} Ha")

    # 2a. Optional: save the (optimised) geometry as XYZ
    if save_opt_xyz is not None:
        tag = "DFT-optimised" if opt_geom else "input"
        print(f"\n[2a] Writing {tag} geometry to XYZ ...")
        save_xyz(mol, save_opt_xyz,
                 comment=f"{tag} geometry  {xc}/{basis}"
                         f"  E={float(mf.e_tot):.8f} Ha")

    # 2b. Optional: save cube(s) for visualisation
    if save_cube_prefix is not None:
        print(f"\n[2b] Writing density + ESP cube files ...")
        save_cubes(mol, dm, save_cube_prefix,
                   padding=padding, spacing=grid_spacing, use_gpu=gpu)
    elif save_cube is not None:
        print(f"\n[2b] Writing ESP cube file (ESP only) ...")
        save_esp_cube(mol, dm, save_cube,
                      padding=padding, spacing=grid_spacing, use_gpu=gpu)

    # 3. Isosurface
    print(f"\n[3/4] Extracting ρ = {iso} e/Bohr³ isosurface ...")
    verts, faces = extract_isosurface(mol, dm, iso=iso,
                                       padding=padding,
                                       spacing=grid_spacing, use_gpu=gpu)

    # 4. α_S, β_S
    print(f"\n[4/4] Computing α_S, β_S ...")
    res = compute_alpha_beta(mol, dm, verts, faces,
                             esp_clip=esp_clip, use_gpu=gpu)

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print(f"  α_S          = {res['alpha_s']:9.3f}  kcal/mol")
    print(f"  β_S          = {res['beta_s']:9.3f}  kcal/mol")
    print(f"  ──────────────────────────────────────────")
    print(f"  ESP_min      = {res['esp_min_kcal']:9.3f}  kcal/mol  (global surface min)")
    print(f"  ESP_max      = {res['esp_max_kcal']:9.3f}  kcal/mol  (global surface max)")
    print(f"  V̄(S⁻)       = {res['Vbar_neg_kcal']:9.3f}  kcal/mol  (area-wtd mean over S⁻)")
    print(f"  V̄(S⁺)       = {res['Vbar_pos_kcal']:9.3f}  kcal/mol  (area-wtd mean over S⁺)")
    print(f"  S⁻           = {res['S_neg']:9.3f}  Bohr²")
    print(f"  S⁺           = {res['S_pos']:9.3f}  Bohr²")
    print(f"  S_total      = {res['S_tot']:9.3f}  Bohr²")
    print(f"  V_mol        = {res['V_mol_bohr3']:9.3f}  Bohr³")
    print(f"  Mesh         = {res['n_verts']:,} verts / {res['n_tris']:,} tris")
    if res["n_clipped"]:
        print(f"  ⚠ Clipped   = {res['n_clipped']}")
    print(f"\n  Per-atom breakdown:")
    print(f"  {'Idx':>4s}  {'Sym':>4s}  {'S_atom (Bohr²)':>16s}  "
          f"{'ESP_min':>10s}  {'ESP_max':>10s}  {'ESP_mean':>10s}  (kcal/mol)")
    print(f"  {'─'*4}  {'─'*4}  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*10}")
    for row in res["per_atom"]:
        if row["esp_min_kcal"] is None:
            print(f"  {row['index']:>4d}  {row['symbol']:>4s}  "
                  f"{'—':>16s}  {'—':>10s}  {'—':>10s}  {'—':>10s}")
        else:
            print(f"  {row['index']:>4d}  {row['symbol']:>4s}  "
                  f"{row['S_atom_bohr2']:16.3f}  "
                  f"{row['esp_min_kcal']:10.3f}  "
                  f"{row['esp_max_kcal']:10.3f}  "
                  f"{row['esp_mean_kcal']:10.3f}")
    print("─" * 64)

    return res


# ── Batch processing ────────────────────────────────────────────────────
def batch_compute(molecules, **kwargs):
    """
    Compute α_S, β_S for a list of molecules.

    Parameters
    ----------
    molecules : list of dicts
        Each dict must contain one of 'smiles', 'xyz_file', 'xyz_string'.
        An optional 'name' key is used for labelling.
    **kwargs : passed through to compute_affinities

    Returns
    -------
    list of result dicts (one per molecule)
    """
    results = []
    for i, entry in enumerate(molecules):
        entry  = dict(entry)                     # don't mutate caller's list
        name   = entry.pop("name", f"mol_{i}")
        print(f"\n{'#'*64}\n# {name}  ({i+1}/{len(molecules)})\n{'#'*64}")
        try:
            res          = compute_affinities(**entry, **kwargs)
            res["name"]  = name
            res["status"] = "ok"
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            res = {"name": name, "status": "error", "error": str(exc)}
        results.append(res)

    # Summary table
    print(f"\n{'='*64}")
    print(f"  {'Name':<24s} {'α_S (kcal/mol)':>16s}  {'β_S (kcal/mol)':>16s}")
    print(f"  {'─'*24}  {'─'*16}  {'─'*16}")
    for r in results:
        if r.get("status") == "ok":
            print(f"  {r['name']:<24s} {r['alpha_s']:16.3f}  {r['beta_s']:16.3f}")
        else:
            print(f"  {r['name']:<24s}  *** error ***")
    print("=" * 64)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute α_S / β_S with analytical ESP (PySCF)")
    geom = parser.add_mutually_exclusive_group(required=True)
    geom.add_argument("--xyz",    type=str, help="Path to XYZ file")
    geom.add_argument("--smiles", type=str, help="SMILES string")
    parser.add_argument("--basis",   default="6-311+g(d,p)")
    parser.add_argument("--xc",      default="b3lyp")
    parser.add_argument("--disp",    default="d3bj")
    parser.add_argument("--charge",  type=int, default=0)
    parser.add_argument("--spin",    type=int, default=0)
    parser.add_argument("--opt",     action="store_true",
                        help="Optimise geometry before SP (needs geometric)")
    parser.add_argument("--spacing", type=float, default=0.20,
                        help="Grid spacing in Bohr for marching cubes (default 0.20)")
    parser.add_argument("--iso",     type=float, default=1e-3)
    parser.add_argument("--cube",    type=str, default=None,
                        help="Write ESP-only cube to this path (no density surface)")
    parser.add_argument("--cube-prefix", dest="cube_prefix", type=str, default=None,
                        help="Write BOTH <prefix>_density.cube and <prefix>_esp.cube "
                             "on the same grid (use these for ChimeraX/VMD)")
    parser.add_argument("--opt-xyz", dest="opt_xyz", type=str, nargs="?",
                        const="__AUTO__", default=None,
                        help="Save the final geometry as XYZ. With --opt this is "
                             "the DFT-optimised structure. Give a path, or pass the "
                             "flag alone to auto-name it from the input.")
    gpu_grp = parser.add_mutually_exclusive_group()
    gpu_grp.add_argument("--gpu", action="store_true",
                        help="Run SCF, optimisation, density grid and ESP on GPU "
                             "via gpu4pyscf (errors if no GPU is available)")
    gpu_grp.add_argument("--gpu-auto", action="store_true",
                        help="Use the GPU if available, otherwise fall back to CPU")
    args = parser.parse_args()

    gpu_arg = True if args.gpu else ("auto" if args.gpu_auto else False)

    # Resolve the optimised-geometry output path
    opt_xyz = args.opt_xyz
    if opt_xyz == "__AUTO__":
        import os
        if args.xyz:
            stem = os.path.splitext(os.path.basename(args.xyz))[0]
        else:
            stem = "molecule"
        opt_xyz = f"{stem}_opt.xyz"

    compute_affinities(
        xyz_file   = args.xyz,
        smiles     = args.smiles,
        basis      = args.basis,
        xc         = args.xc,
        disp       = args.disp,
        charge     = args.charge,
        spin       = args.spin,
        opt_geom   = args.opt,
        grid_spacing = args.spacing,
        iso        = args.iso,
        save_cube        = args.cube,
        save_cube_prefix = args.cube_prefix,
        save_opt_xyz     = opt_xyz,
        use_gpu          = gpu_arg,
    )