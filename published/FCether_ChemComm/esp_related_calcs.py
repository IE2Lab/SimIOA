#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from skimage.measure import marching_cubes
from scipy.interpolate import RegularGridInterpolator

HARTREE_TO_KCAL_MOL = 627.509474
AU_TO_EV = 27.211386
HARTREE_TO_KJ_MOL = 2625.5

def read_cube_values(path: str) -> np.ndarray:
    with open(path) as f:
        f.readline(); f.readline()
        nat = int(f.readline().split()[0])
        nx, *_ = map(float, f.readline().split()); nx = int(nx)
        ny, *_ = map(float, f.readline().split()); ny = int(ny)
        nz, *_ = map(float, f.readline().split()); nz = int(nz)
        for _ in range(nat): f.readline()
        vals = []
        for line in f: 
            vals += [float(x) for x in line.split()]
    return np.array(vals).reshape(nx, ny, nz)

def get_esp_min_max(rho_cube: str, esp_cube: str, iso: float = 1e-3, tol: float = 0.15, units: str = 'eV') -> tuple:
    """
    Get the minimum and maximum electrostatic potential (ESP) values from the ESP cube file.
    Arguments:
        iso = 1.0e-3           # target isodensity
        tol = 0.15             # ±15% band around the surface
    Returns:
        tuple: minimum and maximum ESP values
    """
    # rho = read_cube_values(rho_cube)
    # esp = read_cube_values(esp_cube)

    rho, origin, axes, atoms = read_cube(rho_cube)
    esp, _, _, _ = read_cube(esp_cube)

    mask = (rho > iso*(1-tol)) & (rho < iso*(1+tol))

    Vs_min_au = esp[mask].min()
    Vs_max_au = esp[mask].max()

    if units == 'au':
        return Vs_min_au, Vs_max_au
    elif units == 'eV':
        conversion_factor = 27.211386
        return Vs_min_au * conversion_factor, Vs_max_au * conversion_factor
    elif units == 'kJ/mol':
        conversion_factor = 2625.5
        return Vs_min_au * conversion_factor, Vs_max_au * conversion_factor
    else:
        raise ValueError(f"Unknown units: {units}")

def read_cube(path: str) -> tuple:
    """
    Minimal Gaussian .cube reader.
    Returns:
      data  : ndarray (nx, ny, nz)
      origin: (3,) in bohr
      axes  : (3,3) rows are per-step vectors for x, y, z (bohr)
      atoms : list of (Z, x, y, z) in bohr
    """
    atoms = []
    with open(path, "r") as f:
        # 2 title/comment lines
        _ = f.readline(); _ = f.readline()
        # natoms and origin
        parts = f.readline().split()
        nat = int(parts[0])
        origin = np.array(list(map(float, parts[1:4])))

        # grid lines: count and step vectors
        def read_axis():
            p = list(map(float, f.readline().split()))
            n = int(p[0]); vec = np.array(p[1:4])
            return n, vec

        nx, vx = read_axis()
        ny, vy = read_axis()
        nz, vz = read_axis()
        axes = np.vstack([vx, vy, vz])

        # atom lines: (Z, charge?) x y z   (we keep Z and coords)
        for _ in range(nat):
            p = f.readline().split()
            Z = int(float(p[0]))
            x, y, z = map(float, p[2:5])
            atoms.append((Z, x, y, z))

        # remaining floats are volumetric values
        vals = []
        for line in f:
            if line.strip():
                vals.extend(map(float, line.split()))

    data = np.array(vals, dtype=float).reshape(nx, ny, nz, order='C')
    return data, origin, axes, atoms

def index_to_world(verts_idx, origin, axes):
    """
    Map (i,j,k) index-space vertices to world coords:
      r = origin + i*vx + j*vy + k*vz
    """
    # verts_idx shape: (N,3)
    vx, vy, vz = axes
    return origin + verts_idx[:, [0]]*vx + verts_idx[:, [1]]*vy + verts_idx[:, [2]]*vz

def tri_area(p0, p1, p2):
    return 0.5*np.linalg.norm(np.cross(p1 - p0, p2 - p0))

def tri_centroid(p0, p1, p2):
    return (p0 + p1 + p2) / 3.0

def nearest_atom_index(point, atoms_xyz):
    d2 = np.sum((atoms_xyz - point)**2, axis=1)
    return int(np.argmin(d2))

def split_triangle_by_zero(p, v):
    """
    Split a triangle (p0,p1,p2) with ESP values (v0,v1,v2) at vertices
    into up to two sub-tri sets: negative and positive.
    Returns: list of (subtri_points, subtri_values, sign) with sign in {'neg','pos'}
    Zero-crossings are linearly interpolated (values at crossings set to 0).
    """
    p0, p1, p2 = p
    v0, v1, v2 = v
    signs = np.sign([v0, v1, v2])
    nneg = np.sum(np.array([v0, v1, v2]) < 0.0)
    npos = np.sum(np.array([v0, v1, v2]) > 0.0)

    # All same sign (or zeros)
    if nneg == 3:
        return [([p0, p1, p2], [v0, v1, v2], 'neg')]
    if npos == 3:
        return [([p0, p1, p2], [v0, v1, v2], 'pos')]
    if (nneg == 0 and npos == 0):
        # all zero: ignore (no contribution)
        return []

    # Helper: interpolate zero crossing on edge (pa,va)->(pb,vb)
    def edge_zero(pa, va, pb, vb):
        # va and vb have opposite signs
        t = va / (va - vb)  # where value crosses zero
        return pa + t*(pb - pa)

    # Cases: 1 neg / 2 pos  OR  1 pos / 2 neg
    # Normalize to handle by swapping labels
    points = [p0, p1, p2]; vals = [v0, v1, v2]
    neg_idx = [i for i in range(3) if vals[i] < 0.0]
    pos_idx = [i for i in range(3) if vals[i] > 0.0]

    out = []
    if len(neg_idx) == 1 and len(pos_idx) == 2:
        n = neg_idx[0]; pA = points[n]; vA = vals[n]
        pB = points[pos_idx[0]]; vB = vals[pos_idx[0]]
        pC = points[pos_idx[1]]; vC = vals[pos_idx[1]]
        q1 = edge_zero(pA, vA, pB, vB)
        q2 = edge_zero(pA, vA, pC, vC)
        # Negative subtriangle: (pA, q1, q2) with values (vA, 0, 0)
        out.append(([pA, q1, q2], [vA, 0.0, 0.0], 'neg'))
        # Positive quad split into two triangles: (pB, pC, q2) and (pB, q2, q1)
        out.append(([pB, pC, q2], [vB, vC, 0.0], 'pos'))
        out.append(([pB, q2, q1], [vB, 0.0, 0.0], 'pos'))
    elif len(pos_idx) == 1 and len(neg_idx) == 2:
        p = [points[i] for i in [0,1,2]]
        v = [vals[i] for i in [0,1,2]]
        pA = points[pos_idx[0]]; vA = vals[pos_idx[0]]
        pB = points[neg_idx[0]]; vB = vals[neg_idx[0]]
        pC = points[neg_idx[1]]; vC = vals[neg_idx[1]]
        q1 = edge_zero(pA, vA, pB, vB)
        q2 = edge_zero(pA, vA, pC, vC)
        # Positive subtriangle
        out.append(([pA, q1, q2], [vA, 0.0, 0.0], 'pos'))
        # Negative region split into two
        out.append(([pB, pC, q2], [vB, vC, 0.0], 'neg'))
        out.append(([pB, q2, q1], [vB, 0.0, 0.0], 'neg'))
    else:
        # cases with zeros on vertices; assign by sign or skip if strictly 0
        # Simple fallback: classify by average sign (minor effect)
        meanv = np.mean([v0, v1, v2])
        sgn = 'pos' if meanv > 0 else 'neg'
        out.append(([p0,p1,p2], [v0,v1,v2], sgn))

    return out

def mesh_volume(verts_world, faces):
    """
    Oriented volume of a closed triangular mesh (may be negative; take abs).
    V = sum over faces of dot(p0, cross(p1, p2)) / 6
    Assumes faces index rows [i,j,k] into verts_world.
    """
    v0 = verts_world[faces[:,0]]
    v1 = verts_world[faces[:,1]]
    v2 = verts_world[faces[:,2]]
    crossp = np.cross(v1, v2)
    vol = np.sum(np.einsum('ij,ij->i', v0, crossp)) / 6.0
    return abs(vol)

# Bondi van der Waals radii (Å) for common elements in these solvents
bondi_vdw_A = {
    1: 1.20,  6: 1.70,  7: 1.55,  8: 1.52,  9: 1.47,
    14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75,
    35: 1.85, 53: 1.98
}
ANG2BOHR = 1.88972612546

def compute_affinities(dens_cube, esp_cube, iso: float = 1.0e-3) -> dict:
    # load cubes
    rho, origin, axes, atoms = read_cube(dens_cube)
    esp, origin2, axes2, atoms2 = read_cube(esp_cube)
    assert rho.shape == esp.shape, "Density and ESP grids must match."
    assert np.allclose(origin, origin2) and np.allclose(axes, axes2), "Cube grids differ."
    nx, ny, nz = rho.shape

    # extract isosurface of density (vdW/promolecular surface)
    verts_idx, faces, normals, _ = marching_cubes(rho, level=iso, allow_degenerate=False)

    # ESP interpolator in index-space (shared grid)
    interp = RegularGridInterpolator(
        (np.arange(nx), np.arange(ny), np.arange(nz)),
        esp, method='linear', bounds_error=False, fill_value=np.nan
    )

    # sample ESP at surface vertices
    esp_on_verts = interp(verts_idx)  # a.u. (Hartree/e)

    # world coordinates for geometry (bohr)
    verts_world = index_to_world(verts_idx, origin, axes)

    # atom coordinates (bohr)
    atoms_xyz = np.array([[x,y,z] for (_,x,y,z) in atoms])
    atoms_Z   = np.array([Z for (Z,_,_,_) in atoms], dtype=int)

    # vector of R_A in bohr, one per atom (fallback 1.70 Å if element missing)
    R_bohr = np.array([bondi_vdw_A.get(Z, 1.70) * ANG2BOHR for Z in atoms_Z], dtype=float)

    def argmax_wA(point_xyz):
        """
        Implements SI Eq.(1): w_A = 1 - |r - r_A| / R_A
        Returns index of atom with maximum w_A (winner-take-all).
        """
        d = np.linalg.norm(atoms_xyz - point_xyz, axis=1)  # |r - r_A| in bohr
        w = 1.0 - d / R_bohr                               # may be negative far away; that's ok for argmax
        return int(np.argmax(w))

    # accumulators
    Smin = 0.0; Smax = 0.0
    Imin = 0.0; Imax = 0.0  # surface integrals of MEP over negative/positive regions
   
    # optional per-atom areas
    Smin_atom = np.zeros(len(atoms_xyz))
    Smax_atom = np.zeros(len(atoms_xyz))

    # NEW: per-atom ESP integrals (area * V) on Smin/Smax
    Imin_atom = np.zeros(len(atoms_xyz))
    Imax_atom = np.zeros(len(atoms_xyz))

    # iterate faces; integrate with zero-crossing splits
    for f in faces:
        i, j, k = f
        p0, p1, p2 = verts_world[i], verts_world[j], verts_world[k]
        v0, v1, v2 = esp_on_verts[i], esp_on_verts[j], esp_on_verts[k]
        parts = split_triangle_by_zero([p0,p1,p2], [v0,v1,v2])
        for pts, vals, sgn in parts:
            a = tri_area(pts[0], pts[1], pts[2])  # bohr^2
            vavg = np.mean(vals)                 # a.u.
            c = tri_centroid(pts[0], pts[1], pts[2])
            # ai = nearest_atom_index(c, atoms_xyz)
            ai = argmax_wA(c)  # use w_A to assign atom
            if sgn == 'neg':
                Smin += a
                Smin_atom[ai] += a
                Imin += a * vavg   # negative contribution (vavg <= 0)
                Imin_atom[ai] += a * vavg
            else:
                Smax += a
                Smax_atom[ai] += a
                Imax += a * vavg   # positive contribution (vavg >= 0)
                Imax_atom[ai] += a * vavg

    # enclosed volume from mesh (bohr^3)
    Vtot = mesh_volume(verts_world, faces)

    # NEW: per-atom area-averaged ESP (a.u.) on Smin/Smax
    Vmin_atom_au = np.zeros(len(atoms_xyz))
    Vmax_atom_au = np.zeros(len(atoms_xyz))
    mask_Smin = Smin_atom > 0
    mask_Smax = Smax_atom > 0
    Vmin_atom_au[mask_Smin] = Imin_atom[mask_Smin] / Smin_atom[mask_Smin]
    Vmax_atom_au[mask_Smax] = Imax_atom[mask_Smax] / Smax_atom[mask_Smax]

    # eqns (3) and (4): αS, βS (a.u.). NOTE: Imin<=0, Imax>=0
    # αS = (∬_Smin MEP_min dS) * Smin^(1/2) * Vtot^(-1/3)
    # βS = (∬_Smax MEP_max dS) * Smax^(1/2) * Vtot^(-1/3)
    # units: result is in a.u. of potential (Hartree/e)
    # scale_min = (Smin**0.5) * (Vtot**(-1.0/3.0)) if Smin > 0 else 0.0
    # scale_max = (Smax**0.5) * (Vtot**(-1.0/3.0)) if Smax > 0 else 0.0
    # alpha_au = Imin * scale_min
    # beta_au  = Imax * scale_max

    # NEW (use area-averaged MEP on each region):
    scale_min = (Smin**0.5) * (Vtot**(-1/3)) if Smin > 0 else 0.0
    scale_max = (Smax**0.5) * (Vtot**(-1/3)) if Smax > 0 else 0.0
    Vbar_min_au = (Imin / Smin) if Smin > 0 else 0.0   # a.u.
    Vbar_max_au = (Imax / Smax) if Smax > 0 else 0.0   # a.u.
    alpha_au = Vbar_min_au * scale_min
    beta_au  = Vbar_max_au * scale_max

    out = {
        "Smin_bohr2": Smin,
        "Smax_bohr2": Smax,
        "Vtot_bohr3": Vtot,
        "alpha_au": alpha_au,
        "beta_au": beta_au,
        "alpha_kcal_mol": alpha_au * HARTREE_TO_KCAL_MOL,
        "beta_kcal_mol":  beta_au  * HARTREE_TO_KCAL_MOL,
        "alpha_eV": alpha_au * AU_TO_EV,
        "beta_eV":  beta_au  * AU_TO_EV,
        "Smin_atom_bohr2": Smin_atom,
        "Smax_atom_bohr2": Smax_atom,
        # NEW: per-atom mean ESP on Smin/Smax (a.u.)
        "Vmin_atom_au": Vmin_atom_au,
        "Vmax_atom_au": Vmax_atom_au,
    }
    return out

def area_weighted_mean_on_surface(density_cube, esp_cube, iso=1.0e-3):
    # Unpack
    rho, origin, axes = density_cube
    esp, origin2, axes2 = esp_cube
    assert rho.shape == esp.shape and np.allclose(origin, origin2) and np.allclose(axes, axes2)

    nx, ny, nz = rho.shape
    # Isosurface from density
    verts_idx, faces, normals, _ = marching_cubes(rho, level=iso, allow_degenerate=False)

    # Interpolator of counterion ESP (index-space)
    interp = RegularGridInterpolator((np.arange(nx), np.arange(ny), np.arange(nz)),
                                     esp, method='linear', bounds_error=False, fill_value=np.nan)
    Vv = interp(verts_idx)  # ESP at surface vertices (a.u.)

    # World coords for triangle areas (bohr)
    verts_xyz = index_to_world(verts_idx, origin, axes)

    # Area-weighted mean: sum_face [ area * mean(vertex ESP) ] / sum_face [ area ]
    tri = faces.astype(int)
    p0, p1, p2 = verts_xyz[tri[:,0]], verts_xyz[tri[:,1]], verts_xyz[tri[:,2]]
    areas = 0.5*np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    Vmean_face = (Vv[tri[:,0]] + Vv[tri[:,1]] + Vv[tri[:,2]]) / 3.0

    mask = np.isfinite(Vmean_face) & (areas > 0)
    Aw = areas[mask].sum()
    if Aw == 0:
        return np.nan
    return (areas[mask] * Vmean_face[mask]).sum() / Aw  # <V> in a.u.

def salt_affinities_kcalmol(
    anion_density_cube_path,  anion_potential_cube_path,   # for beta_salt (anion field on cation surface)
    cation_density_cube_path, cation_potential_cube_path,  # for alpha_salt (cation field on anion surface)
    iso=1.0e-3
):
    # Load cubes
    rho_A, org_A, ax_A = read_cube(anion_density_cube_path)
    V_A,   org_A2, ax_A2 = read_cube(anion_potential_cube_path)
    rho_C, org_C, ax_C = read_cube(cation_density_cube_path)
    V_C,   org_C2, ax_C2 = read_cube(cation_potential_cube_path)

    # α_salt: cation ESP on anion surface (multiply by anion charge, -1)
    Vbar_cation_on_anion = area_weighted_mean_on_surface(
        density_cube=(rho_A, org_A, ax_A),
        esp_cube=(V_C, org_C2, ax_C2),
        iso=iso
    )
    alpha_kcal = (-1.0) * Vbar_cation_on_anion * HARTREE_TO_KCAL_MOL

    # β_salt: anion ESP on cation surface (multiply by cation charge, +1)
    Vbar_anion_on_cation = area_weighted_mean_on_surface(
        density_cube=(rho_C, org_C, ax_C),
        esp_cube=(V_A, org_A2, ax_A2),
        iso=iso
    )
    beta_kcal = (+1.0) * Vbar_anion_on_cation * HARTREE_TO_KCAL_MOL

    return alpha_kcal, beta_kcal, Vbar_cation_on_anion, Vbar_anion_on_cation

# alpha_kcal, beta_kcal, VbarC_on_A, VbarA_on_C = salt_affinities_kcalmol(
#     "dens_anion.cube", "esp_anion.cube",
#     "dens_cation.cube","esp_cation.cube",
#     iso=1e-3
# )
# print(f"alpha_salt ≈ {alpha_kcal:.2f} kcal/mol")
# print(f"beta_salt  ≈ {beta_kcal:.2f} kcal/mol")

def main():
    ap = argparse.ArgumentParser(description="Compute αS, βS (eqns 3,4) from density & ESP cube files.")
    ap.add_argument("density_cube", help="Density cube (Gaussian cubegen 'density').")
    ap.add_argument("esp_cube",     help="ESP cube (Gaussian cubegen 'potential').")
    ap.add_argument("--iso", type=float, default=1.0e-3,
                    help="Isodensity (a.u.) for the vdW/promolecular surface (default: 1e-3).")
    args = ap.parse_args()

    res = compute_affinities(args.density_cube, args.esp_cube, iso=args.iso)

    print("# Surface/volume (bohr units)")
    print(f"Smin = {res['Smin_bohr2']:.6e}  Smax = {res['Smax_bohr2']:.6e}  Vtot = {res['Vtot_bohr3']:.6e}")
    print("# Affinities (atomic units, eV, kcal/mol)")
    print(f"αS = {res['alpha_au']:.6e} a.u.  = {res['alpha_eV']:.6f} eV  = {res['alpha_kcal_mol']:.3f} kcal/mol")
    print(f"βS = {res['beta_au']:.6e} a.u.  = {res['beta_eV']:.6f} eV  = {res['beta_kcal_mol']:.3f} kcal/mol")

if __name__ == "__main__":
    main()