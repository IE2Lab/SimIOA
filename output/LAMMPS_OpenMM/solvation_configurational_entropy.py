#!/usr/bin/env python3
"""
Compute solvation-configurational entropy from MD trajectories.

This script is built from the logic already present in
`rdf_cn_solv-dist_solv-struc_res-time.ipynb`: for each sampled frame and each
solute ion, it counts the number of first-shell neighbors that satisfy a set of
species-specific selections and cutoff distances.

According to the SI you attached, the paper defines

    S_sc = -k_B * sum_i p_i ln(p_i)

where p_i is the probability of the i-th solvation-structure type. In practice,
that means:

1. Define a discrete solvation-structure type around the cation.
2. Count the occurrence N_i of each type over all sampled solvation-shell
   observations.
3. Convert to probabilities p_i = N_i / N.
4. Evaluate S_sc from the Gibbs/Shannon expression above.

In the original notebook, the natural paper-faithful structure type is the
joint first-shell composition, e.g. `(anion_count, solvent_count)` or, more
generally, the vector of counts across all shell species. That is the default
descriptor in this script.

The script can also compute coarser alternative descriptors such as total
coordination number or ion-pairing class. Those are useful if you explicitly
want alternative entropy-like summaries, but they are not the strict SI
definition unless that coarse-grained state is the "solvation structure type"
you intend to study.

Example
-------
python solvation_configurational_entropy.py \
    --topology config.pdb \
    --trajectory nvt3.lammpsdump \
    --solute "resname Li" \
    --shell-species FSI "resname fsa and (name O* F*)" 2.75 \
    --shell-species P3FO "resname p3f and (name O* F*)" 2.85 \
    --anion-species FSI \
    --start-ps 500 \
    --sample-every 50 \
    --output-prefix P3FO_LiFSI
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np
import pandas as pd

K_BOLTZMANN = 1.380649e-23  # J K^-1
N_AVOGADRO = 6.02214076e23  # mol^-1
R_GAS_CONSTANT = K_BOLTZMANN * N_AVOGADRO  # J mol^-1 K^-1


@dataclass(frozen=True)
class ShellSpecies:
    name: str
    selection: str
    cutoff: float


@dataclass
class ShellEvent:
    frame: int
    time_ps: float
    solute_index: int
    solute_resid: int
    counts: Dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate solvation-configurational entropy descriptors."
    )
    parser.add_argument("--topology", required=True, help="Topology file.")
    parser.add_argument("--trajectory", required=True, help="Trajectory file.")
    parser.add_argument(
        "--solute",
        required=True,
        help='MDAnalysis selection for the solute center, e.g. "resname Li".',
    )
    parser.add_argument(
        "--shell-species",
        action="append",
        nargs=3,
        metavar=("NAME", "SELECTION", "RCUT"),
        required=True,
        help=(
            "Species name, MDAnalysis selection, and first-shell cutoff in "
            "angstrom. Repeat once per shell species."
        ),
    )
    parser.add_argument(
        "--count-mode",
        choices=("residue", "atom"),
        default="residue",
        help="Count distinct residues (default) or raw atoms inside each shell.",
    )
    parser.add_argument(
        "--start-ps",
        type=float,
        default=None,
        help="Start sampling at this trajectory time in ps.",
    )
    parser.add_argument(
        "--stop-ps",
        type=float,
        default=None,
        help="Stop sampling at this trajectory time in ps.",
    )
    parser.add_argument(
        "--dt-ps",
        type=float,
        default=None,
        help=(
            "Fallback timestep in ps when the trajectory does not expose time. "
            "If omitted, the script tries ts.time and then trajectory.dt."
        ),
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Use every Nth frame after the start criterion.",
    )
    parser.add_argument(
        "--periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use periodic distance searches (default: True).",
    )
    parser.add_argument(
        "--anion-species",
        default=None,
        help=(
            "Species used to classify SSIP/CIP/AGG ion-pairing states. "
            "Required only for the ion_pairing descriptor."
        ),
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        default=["joint_counts"],
        choices=("joint_counts", "total_cn", "presence", "marginal", "ion_pairing"),
        help=(
            "Descriptor families to evaluate. The paper-faithful S_sc is "
            "'joint_counts', where each unique first-shell composition is a "
            "solvation-structure type."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="solvation_entropy",
        help="Prefix for CSV/JSON output files.",
    )
    parser.add_argument(
        "--max-states-to-print",
        type=int,
        default=15,
        help="How many top states to echo to stdout per descriptor.",
    )
    return parser.parse_args()


def build_species(spec_args: Sequence[Sequence[str]]) -> List[ShellSpecies]:
    species = []
    for name, selection, cutoff in spec_args:
        species.append(ShellSpecies(name=name, selection=selection, cutoff=float(cutoff)))
    return species


def get_time_ps(ts, frame_index: int, dt_ps: float | None) -> float:
    time_ps = getattr(ts, "time", None)
    if time_ps is not None:
        return float(time_ps)
    if dt_ps is not None:
        return frame_index * dt_ps
    raise ValueError(
        "Trajectory time is unavailable. Provide --dt-ps so frame indices can "
        "be converted to ps."
    )


def state_to_string(state) -> str:
    if isinstance(state, tuple):
        return "|".join(str(x) for x in state)
    return str(state)


def format_joint_state(state: Tuple[int, ...], species_names: Sequence[str]) -> str:
    return "|".join(f"{name}={count}" for name, count in zip(species_names, state))


def entropy_summary(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        raise ValueError("Cannot calculate entropy from an empty distribution.")
    probabilities = np.array([count / total for count in counter.values()], dtype=float)
    h_nats = float(-(probabilities * np.log(probabilities)).sum())
    n_states = int(len(counter))
    h_norm = h_nats / math.log(n_states) if n_states > 1 else 0.0
    return {
        "samples": int(total),
        "observed_states": n_states,
        "minus_sum_p_log_p": h_nats,
        "S_over_kB": h_nats,
        "S_kB_per_shell": h_nats,
        "S_bits": h_nats / math.log(2.0),
        "S_J_per_K_per_shell": K_BOLTZMANN * h_nats,
        "S_J_molK_equivalent": R_GAS_CONSTANT * h_nats,
        "normalized_H": h_norm,
        "effective_states_expH": math.exp(h_nats),
    }


def distribution_dataframe(
    counter: Counter,
    descriptor_name: str,
    state_formatter,
) -> pd.DataFrame:
    total = sum(counter.values())
    rows = []
    for state, count in counter.most_common():
        prob = count / total
        rows.append(
            {
                "descriptor": descriptor_name,
                "state": state_formatter(state),
                "count": count,
                "probability": prob,
                "minus_p_log_p": -prob * math.log(prob),
            }
        )
    return pd.DataFrame(rows)


def classify_ion_pairing(anion_count: int) -> str:
    if anion_count <= 0:
        return "SSIP"
    if anion_count == 1:
        return "CIP"
    return "AGG"


def collect_shell_events(
    universe: mda.Universe,
    solute_selection: str,
    species: Sequence[ShellSpecies],
    count_mode: str,
    start_ps: float | None,
    stop_ps: float | None,
    dt_ps: float | None,
    sample_every: int,
    periodic: bool,
) -> List[ShellEvent]:
    solute_atoms = universe.select_atoms(solute_selection)
    if len(solute_atoms) == 0:
        raise ValueError(f'Solute selection "{solute_selection}" matched no atoms.')

    events: List[ShellEvent] = []
    post_start_counter = 0

    for ts in universe.trajectory:
        time_ps = get_time_ps(ts, ts.frame, dt_ps)
        if start_ps is not None and time_ps < start_ps:
            continue
        if stop_ps is not None and time_ps > stop_ps:
            break

        if post_start_counter % sample_every != 0:
            post_start_counter += 1
            continue
        post_start_counter += 1

        for solute_atom in solute_atoms:
            center = mda.core.groups.AtomGroup([solute_atom])
            counts: Dict[str, int] = {}

            for spec in species:
                shell_atoms = universe.select_atoms(
                    f"({spec.selection}) and around {spec.cutoff:.6f} group center",
                    center=center,
                    periodic=periodic,
                )
                if count_mode == "residue":
                    counts[spec.name] = len(np.unique(shell_atoms.resids))
                else:
                    counts[spec.name] = len(shell_atoms)

            events.append(
                ShellEvent(
                    frame=int(ts.frame),
                    time_ps=float(time_ps),
                    solute_index=int(solute_atom.index),
                    solute_resid=int(solute_atom.resid),
                    counts=counts,
                )
            )

    return events


def build_events_dataframe(events: Sequence[ShellEvent], species_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for event in events:
        row = {
            "frame": event.frame,
            "time_ps": event.time_ps,
            "solute_index": event.solute_index,
            "solute_resid": event.solute_resid,
        }
        for name in species_names:
            row[f"count_{name}"] = event.counts[name]
        row["joint_counts_state"] = format_joint_state(
            tuple(event.counts[name] for name in species_names), species_names
        )
        row["total_cn"] = sum(event.counts[name] for name in species_names)
        row["presence_state"] = "|".join(
            f"{name}={int(event.counts[name] > 0)}" for name in species_names
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    species = build_species(args.shell_species)
    species_names = [spec.name for spec in species]

    if args.anion_species is not None and args.anion_species not in species_names:
        raise ValueError(
            f'--anion-species "{args.anion_species}" is not among the shell species: '
            f"{', '.join(species_names)}"
        )

    universe = mda.Universe(args.topology, args.trajectory)
    inferred_dt = args.dt_ps
    if inferred_dt is None:
        traj_dt = getattr(universe.trajectory, "dt", None)
        if traj_dt is not None:
            inferred_dt = float(traj_dt)

    events = collect_shell_events(
        universe=universe,
        solute_selection=args.solute,
        species=species,
        count_mode=args.count_mode,
        start_ps=args.start_ps,
        stop_ps=args.stop_ps,
        dt_ps=inferred_dt,
        sample_every=args.sample_every,
        periodic=args.periodic,
    )
    if not events:
        raise ValueError("No shell events were collected. Check your time window and selections.")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    events_df = build_events_dataframe(events, species_names)
    events_path = prefix.with_name(prefix.name + "_events.csv")
    events_df.to_csv(events_path, index=False)

    summary: Dict[str, Dict[str, float]] = {}
    dist_frames: List[pd.DataFrame] = []

    if "joint_counts" in args.descriptors:
        counter = Counter(tuple(event.counts[name] for name in species_names) for event in events)
        name = "joint_counts"
        summary[name] = entropy_summary(counter)
        summary[name]["matches_si_definition"] = True
        summary[name]["state_definition"] = (
            "Each unique vector of first-shell counts across the selected shell species "
            "is treated as one solvation-structure type."
        )
        dist_frames.append(
            distribution_dataframe(
                counter,
                descriptor_name=name,
                state_formatter=lambda state: format_joint_state(state, species_names),
            )
        )

    if "total_cn" in args.descriptors:
        counter = Counter(sum(event.counts[name] for name in species_names) for event in events)
        name = "total_cn"
        summary[name] = entropy_summary(counter)
        summary[name]["matches_si_definition"] = False
        summary[name]["state_definition"] = (
            "Coarse-grained state: total coordination number only."
        )
        dist_frames.append(
            distribution_dataframe(counter, descriptor_name=name, state_formatter=state_to_string)
        )

    if "presence" in args.descriptors:
        counter = Counter(
            tuple(int(event.counts[name] > 0) for name in species_names) for event in events
        )
        name = "presence"
        summary[name] = entropy_summary(counter)
        summary[name]["matches_si_definition"] = False
        summary[name]["state_definition"] = (
            "Coarse-grained state: presence/absence of each shell species."
        )
        dist_frames.append(
            distribution_dataframe(
                counter,
                descriptor_name=name,
                state_formatter=lambda state: "|".join(
                    f"{sname}={present}" for sname, present in zip(species_names, state)
                ),
            )
        )

    if "marginal" in args.descriptors:
        for sname in species_names:
            counter = Counter(event.counts[sname] for event in events)
            name = f"marginal_{sname}"
            summary[name] = entropy_summary(counter)
            summary[name]["matches_si_definition"] = False
            summary[name]["state_definition"] = (
                f"Coarse-grained state: marginal count distribution for {sname} only."
            )
            dist_frames.append(
                distribution_dataframe(
                    counter, descriptor_name=name, state_formatter=state_to_string
                )
            )

    if "ion_pairing" in args.descriptors:
        if args.anion_species is None:
            raise ValueError(
                "The ion_pairing descriptor requires --anion-species to be set."
            )
        counter = Counter(
            classify_ion_pairing(event.counts[args.anion_species]) for event in events
        )
        name = "ion_pairing"
        summary[name] = entropy_summary(counter)
        summary[name]["matches_si_definition"] = False
        summary[name]["state_definition"] = (
            f"Coarse-grained state: SSIP/CIP/AGG class derived from {args.anion_species} count."
        )
        dist_frames.append(
            distribution_dataframe(counter, descriptor_name=name, state_formatter=state_to_string)
        )

    distribution_df = pd.concat(dist_frames, ignore_index=True)
    distribution_path = prefix.with_name(prefix.name + "_distributions.csv")
    distribution_df.to_csv(distribution_path, index=False)

    summary_payload = {
        "topology": args.topology,
        "trajectory": args.trajectory,
        "solute_selection": args.solute,
        "species": [
            {"name": spec.name, "selection": spec.selection, "cutoff": spec.cutoff}
            for spec in species
        ],
        "count_mode": args.count_mode,
        "start_ps": args.start_ps,
        "stop_ps": args.stop_ps,
        "dt_ps": inferred_dt,
        "sample_every": args.sample_every,
        "periodic": args.periodic,
        "n_events": len(events),
        "descriptors": summary,
        "event_table": str(events_path),
        "distribution_table": str(distribution_path),
    }
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print(f"Collected {len(events)} solvation-shell events.")
    print(f"Event table: {events_path}")
    print(f"Distribution table: {distribution_path}")
    print(f"Summary JSON: {summary_path}")
    print()

    for descriptor_name, stats in summary.items():
        print(f"[{descriptor_name}]")
        if stats["matches_si_definition"]:
            print("  SI status: exact paper-style S_sc definition")
        else:
            print("  SI status: alternative coarse-grained entropy descriptor")
        print(
            "  -sum(p ln p) = {minus_sum_p_log_p:.6f}; "
            "S_sc/k_B = {S_over_kB:.6f}; "
            "S_sc = {S_J_per_K_per_shell:.6e} J K^-1 per shell; "
            "molar equivalent = {S_J_molK_equivalent:.6f} J mol^-1 K^-1; "
            "normalized = {normalized_H:.6f}".format(**stats)
        )
        print(f"  State definition: {stats['state_definition']}")
        descriptor_rows = distribution_df.loc[
            distribution_df["descriptor"] == descriptor_name
        ].head(args.max_states_to_print)
        for _, row in descriptor_rows.iterrows():
            print(
                f"    {row['state']}: count={int(row['count'])}, "
                f"p={row['probability']:.6f}"
            )
        print()


if __name__ == "__main__":
    main()
