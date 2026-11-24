#!/usr/bin/env python3
"""Portable PDB/MOL2 -> QDock/neal -> raw/logical pose and RMSD summaries.

The existing single-system CLI owns docking. This file only orchestrates
isolated sequential jobs and evaluates their saved outputs. No Kaiwu or old
project/batch scripts are imported. See README.md for the output definitions.
"""
import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parent
THRESHOLDS = ((2.0, "2A"), (2.5, "2p5A"))
TOP_NS = (1, 3, 5, 10)
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=()):
    fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(data, sid, prepared):
    root = Path(data) / sid
    return (root / ("receptor.pdbqt" if prepared else "receptor.pdb"),
            root / ("ligand.pdbqt" if prepared else "ligand.mol2"))


def relocated_input_hashes(config, data_dir):
    """Keep original digests while rebasing paths for a relocated dataset.

    Do not resolve the individual paths: a ligand symlink is an input name,
    while its target's bytes (not its target filename) are what we hash.
    """
    original = Path(config["options"]["data_dir"])
    return {str(Path(data_dir) / Path(path).relative_to(original)): digest
            for path, digest in config["input_sha256"].items()}


def changed_inputs(manifest):
    return [path for path, digest in manifest.items()
            if not Path(path).is_file() or file_hash(path) != digest]


def reference_coordinates(path, prepared_xyz, prepared_types):
    """Map INPUT native heavy atoms once, never fit/permutate a docked pose.

    MOL2 atom names can be duplicated (e.g. C/N), so use element-constrained
    assignment of the original coordinates and demand <=0.01 A agreement.
    This tolerates PDBQT rounding/reordering, NOT preparation changing geometry.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from QDock.GridPointMatching.structure import read_pdbqt

    elements = {"A": "C", "NA": "N", "OA": "O", "SA": "S", "HD": "H", "HS": "H"}
    heavy = ~np.isin(prepared_types, ["H", "HD", "HS"])
    target_elements = [elements.get(str(t), str(t)).upper() for t in prepared_types[heavy]]
    if path.suffix.lower() == ".pdbqt":
        xyz, _, types = read_pdbqt(path)
        source_heavy = ~np.isin(types, ["H", "HD", "HS"])
        xyz = xyz[source_heavy]
        source_elements = [elements.get(str(t), str(t)).upper() for t in types[source_heavy]]
    else:
        coords, source_elements, atom_section = [], [], False
        for line in path.read_text().splitlines():
            if line.startswith("@<TRIPOS>"):
                atom_section = line.strip() == "@<TRIPOS>ATOM"
            elif atom_section and line.strip():
                fields = line.split()
                element = fields[5].split(".")[0].upper()
                if element != "H":
                    coords.append([float(x) for x in fields[2:5]])
                    source_elements.append(element)
        xyz = np.asarray(coords, dtype=float)
    if not heavy.any() or xyz.shape != prepared_xyz[heavy].shape:
        raise ValueError("Input/prepared ligand heavy-atom counts disagree")
    cost = cdist(prepared_xyz[heavy], xyz)
    for i, element in enumerate(target_elements):
        cost[i, np.asarray(source_elements) != element] = np.inf
    row, col = linear_sum_assignment(cost)
    error = float(cost[row, col].max())
    if not np.isfinite(error) or error > .01:
        raise ValueError(f"Prepared/input reference heavy atoms differ by {error:.6g} A")
    return heavy, xyz[col], dict(reference_file=str(path), n_heavy_atoms=int(heavy.sum()),
        prepared_to_input_heavy_indices=col.tolist(), max_coordinate_difference_A=error,
        definition="fixed input-native atom mapping, heavy-atom RMSD in receptor frame; no alignment or symmetry correction")


def summarize_pose_rows(rows):
    """Retain invalid ranks in unrestricted TopN; filter THEN rank qualified TopN."""
    import statistics

    rows = sorted(rows, key=lambda r: r["energy_rank"])
    qualified = [r for r in rows if r["is_qualified"]]
    valid = [r for r in rows if r["rmsd"] is not None]
    metrics = dict(n_returned=len(rows), n_restored_poses=len(valid),
        n_unrestored=len(rows)-len(valid), n_qualified_poses=len(qualified),
        n_exact_one=sum(r["is_exact_one"] for r in rows),
        exact_one_rate=sum(r["is_exact_one"] for r in rows)/len(rows) if rows else 0.,
        n_states_with_duplicates=sum(r["n_duplicate_atoms"] > 0 for r in rows))
    for key in ("atom_coverage_ratio", "n_missing_atoms", "n_duplicate_atoms"):
        values = [r[key] for r in rows]
        metrics.update({f"{key}_{stat}": func(values) if values else None
                        for stat, func in (("min", min), ("mean", statistics.mean), ("max", max))})
    for prefix, pool in (("", rows), ("qualified_", qualified)):
        values = [r["rmsd"] for r in pool if r["rmsd"] is not None]
        metrics[prefix+"mrmsd"] = min(values) if values else None
        metrics[prefix+"mean_rmsd"] = statistics.mean(values) if values else None
        metrics[prefix+"top1_rmsd"] = pool[0]["rmsd"] if pool else None
        for threshold, label in THRESHOLDS:
            metrics[f"{prefix}n_poses_lt_{label}"] = sum(x < threshold for x in values)
            metrics[f"{prefix}success_lt_{label}"] = int(any(x < threshold for x in values))
            for topn in TOP_NS:
                metrics[f"{prefix}top{topn}_success_lt_{label}"] = int(any(
                    r["rmsd"] is not None and r["rmsd"] < threshold for r in pool[:topn]))
    return metrics


def analyze_branch(work, sid, branch, options, runtime, *, output_root=None):
    import numpy as np
    from QDock.GridPointMatching.structure import read_pdbqt

    row = dict(id=sid, branch=branch, status="failed", error="", solver="neal",
        log_file=str(work / f"{sid}.log"),
        prepared_inputs=options["prepared_inputs"], qubo_round_decimals=options["qubo_round_decimals"],
        precision_method="none", num_reads_requested=options["num_reads"],
        min_coverage=options["min_coverage"], max_duplicate_atoms=0, **runtime)
    for key in ("center_length", "grid_length", "edge_cutoff", "K_dist", "K_mono", "K_cov",
                "w_affinity", "w_elec", "w_solv", "affinity_max", "energy_max", "max_ising_size"):
        row[key] = options[key]
    # Failure rows have explicit zero success flags; undefined RMSD stays blank.
    row.update(summarize_pose_rows([]))
    directory = work / sid / branch
    output = (work if output_root is None else output_root) / sid / branch
    try:
        run = work / sid
        meta = read_json(run / "IsingMatrices/preprocess.json")
        row.update(n_qubo_variables=meta["original_ising_size"]-1,
                   n_ising_spins=meta["final_ising_size"])
        records = read_json(directory / "matches.json")
        states = np.load(directory / "states.npy", allow_pickle=False, mmap_mode="r")
        if states.ndim != 2 or not np.isin(states, [-1, 1]).all():
            raise ValueError("Saved logical states are not a binary spin matrix")
        if len(records) != len(states) or {r['source_index'] for r in records} != set(range(len(states))):
            raise ValueError("Saved records and source-state indices disagree")
        # The CLI resolves input symlinks; the prepared basename can therefore
        # differ from data/<id>/ligand.*, and may itself contain dots.
        name = read_json(run / "ligand_metadata.json")["name"]
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Invalid prepared ligand name in ligand_metadata.json")
        xyz, _, atom_types = read_pdbqt(run / "Ligands" / f"{name}.pdbqt")
        _, source = input_paths(options["data_dir"], sid, options["prepared_inputs"])
        heavy, reference, audit = reference_coordinates(source, xyz, atom_types)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "reference_audit.json", audit)
        coordinates = {}
        if (directory / "pose_coordinates.npy").is_file():
            poses = np.load(directory / "pose_coordinates.npy", allow_pickle=False)
            indices = np.load(directory / "pose_source_indices.npy", allow_pickle=False)
            if poses.shape != (len(indices), len(xyz), 3) or len(set(indices.tolist())) != len(indices):
                raise ValueError("Pose coordinate/index shape mismatch")
            coordinates = dict(zip(indices.tolist(), poses))
        if set(coordinates) != {r['source_index'] for r in records if r['pose_model'] is not None}:
            raise ValueError("Pose files and match records disagree")
        pose_rows, qualified_rank = [], 0
        for record in sorted(records, key=lambda r: r["energy_rank"]):
            count = np.asarray(record["selection_counts"], dtype=int)
            if count.shape != (len(xyz),) or (count < 0).any():
                raise ValueError("Invalid per-atom selection counts")
            pose = coordinates.get(record["source_index"])
            rmsd = None
            if pose is not None:
                if not np.isfinite(pose).all():
                    raise ValueError("Nonfinite pose coordinates")
                rmsd = float(np.sqrt(np.mean(np.sum((pose[heavy]-reference)**2, axis=1))))
            coverage = float(np.count_nonzero(count)/len(count))
            duplicates = int((count > 1).sum())
            qualified = rmsd is not None and coverage >= options["min_coverage"] and duplicates == 0
            qualified_rank += int(qualified)
            entry = dict(id=sid, branch=branch, source_index=record["source_index"],
                energy_rank=record["energy_rank"], qualified_energy_rank=qualified_rank if qualified else None,
                pose_model=record["pose_model"], recovery_status=record["recovery_status"], rmsd=rmsd,
                atom_coverage_ratio=coverage, n_selected_variables=int(count.sum()),
                n_missing_atoms=int((count == 0).sum()), n_duplicate_atoms=duplicates,
                n_extra_selections=int(np.maximum(count-1, 0).sum()), n_nonbinary_variables=0,
                is_exact_one=int(np.all(count == 1)), is_qualified=int(qualified),
                ranking_energy=record["ranking_energy"], original_qubo_energy=record["qubo_energy"])
            for cutoff, tag in THRESHOLDS:
                entry[f"success_lt_{tag}"] = int(rmsd is not None and rmsd < cutoff)
            pose_rows.append(entry)
        write_csv(output / "rmsd_per_pose.csv", pose_rows, ("id", "branch", "source_index"))
        row.update(summarize_pose_rows(pose_rows), status="ok", n_atoms=len(xyz),
            n_heavy_atoms=int(heavy.sum()), n_unique_states=len(np.unique(states, axis=0)),
            reference_match_max_error_A=audit["max_coordinate_difference_A"],
            ranking_target=read_json(directory / "state_space.json")["ranking_target"],
            minimum_ranking_energy=min(r["ranking_energy"] for r in pose_rows),
            minimum_original_qubo_energy=min(r["original_qubo_energy"] for r in pose_rows))
        if not row["n_restored_poses"]:
            row.update(status="no_pose", error="No recoverable poses")
        metadata = read_json(run / "solver_metadata.json")
        row["sampling_seconds"] = metadata["solver_call_wall_s"]
        if branch == "logical_fullprecision":
            diag = read_json(run / "logical_descent.json")
            row.update(logical_mean_energy_drop=float(np.mean(diag["energy_drop"])),
                logical_max_energy_drop=float(np.max(diag["energy_drop"])),
                logical_mean_steps=float(np.mean(diag["steps"])),
                logical_max_steps=int(np.max(diag["steps"])))
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        log_path = work / f"{sid}.log"
        if runtime.get("process_returncode", 0) and log_path.is_file():
            lines = log_path.read_text(errors="replace").splitlines()
            worker_error = next((line.strip() for line in reversed(lines) if line.strip()), "empty log")
            row["error"] = (f"Worker exited {runtime['process_returncode']}: {worker_error}; "
                            f"statistics: {row['error']}")
    return row


def write_summaries(work, rows, planned, branches):
    import statistics

    write_csv(work / "summary.csv", rows, ("id", "branch", "status", "error"))
    for branch in branches:
        write_csv(work / f"{branch}_rmsd_summary.csv", [r for r in rows if r["branch"] == branch],
                  ("id", "branch", "status", "error"))
    failures = [dict(id=r["id"], branch=r["branch"], status=r["status"], error=r["error"],
                     process_returncode=r.get("process_returncode")) for r in rows if r["status"] != "ok"]
    write_csv(work / "failures.csv", failures, ("id", "branch", "status", "error", "process_returncode"))
    aggregate = []
    for branch in branches:
        subset = [r for r in rows if r["branch"] == branch]
        for qualified in (False, True):
            prefix = "qualified_" if qualified else ""
            for _, tag in THRESHOLDS:
                entry = dict(branch=branch, pool="qualified" if qualified else "all",
                    rmsd_threshold=tag, n_planned=planned, n_processed=len(subset),
                    n_ok=sum(r["status"] == "ok" for r in subset),
                    n_failed=sum(r["status"] != "ok" for r in subset))
                for label in ("", *(f"top{n}_" for n in TOP_NS)):
                    count = sum(r.get(f"{prefix}{label}success_lt_{tag}", 0)
                                for r in subset if r["status"] == "ok")
                    entry[f"{label}success_count"] = count
                    entry[f"{label}success_rate"] = count/planned
                values = [r[prefix+"mrmsd"] for r in subset
                          if r["status"] == "ok" and r.get(prefix+"mrmsd") is not None]
                entry.update(n_with_mrmsd=len(values), mean_mrmsd=statistics.mean(values) if values else None)
                aggregate.append(entry)
    write_csv(work / "aggregate_summary.csv", aggregate)


def core_command(options, sid):
    receptor, ligand = input_paths(options["data_dir"], sid, options["prepared_inputs"])
    command = [sys.executable, "-u", "-m", "QDock.GridPointMatching", "--receptor", str(receptor),
        "--ligand", str(ligand), "--work-dir", str(Path(options["work_root"])/sid),
        "--solver", "neal", "--precision-method", "none"]
    for key in ("center_length", "grid_length", "edge_cutoff", "K_dist", "K_mono", "K_cov",
                "w_affinity", "w_elec", "w_solv", "affinity_max", "energy_max", "min_candidate_coverage",
                "qubo_round_decimals", "max_ising_size", "num_reads", "logical_tolerance", "logical_max_steps"):
        value = options[key]
        # argparse can mistake a separate negative scientific-notation value
        # (e.g. -1e-05) for another option. Bind numeric values to their flags.
        command.append("--"+key.replace("_", "-")+"="+("none" if value is None else str(value)))
    if options["prepared_inputs"]:
        command.append("--prepared-inputs")
    if options["neal_seed"] is not None:
        command.append("--neal-seed=" + str(options["neal_seed"]))
    if options["no_logical_postprocess"]:
        command.append("--no-logical-postprocess")
    return command


def parser(prepared):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path,
                   help="Default: bundled data[_pdbqt]; with --analyze-only, saved path unless explicitly overridden")
    p.add_argument("--work-root", type=Path, default=PROJECT/("work_neal_pdbqt" if prepared else "work_neal_prepare"))
    p.add_argument("--only", nargs="+", help="Subdirectory IDs; default: every directory under data")
    for key, default in (("center-length", 8.), ("grid-length", 2.), ("edge-cutoff", 2.182),
            ("K-dist", .107), ("K-mono", 31.823), ("K-cov", .888), ("w-affinity", 1.),
            ("w-elec", .507), ("w-solv", 1.), ("affinity-max", 5.), ("energy-max", 0.),
            ("min-candidate-coverage", .9), ("min-coverage", .9), ("logical-tolerance", 1e-9)):
        p.add_argument("--"+key, type=float, default=default)
    p.add_argument("--qubo-round-decimals", choices=["2", "3", "none"], default="2")
    p.add_argument("--max-ising-size", type=int, default=11999, help="Unsplit N+1, not split size; NOT a memory limit")
    p.add_argument("--num-reads", type=int, default=100)
    p.add_argument("--neal-seed", type=int, help="Omitted: neal SDK random default")
    p.add_argument("--logical-max-steps", type=int, default=10000)
    p.add_argument("--no-logical-postprocess", action="store_true")
    p.add_argument("--threads", type=int, default=1, help="BLAS/OpenMP limit per sequential system")
    p.add_argument("--check-only", action="store_true", help="Check dependencies/inputs; do not write or dock")
    p.add_argument("--analyze-only", action="store_true", help="Rebuild CSVs from work-root; no sampling")
    return p


def main(argv=None, *, prepared=False):
    p = parser(prepared)
    args = p.parse_args(argv)
    if args.check_only and args.analyze_only:
        p.error("Choose check-only OR analyze-only")
    options = vars(args).copy()
    options["data_dir"] = args.data_dir or PROJECT/("data_pdbqt" if prepared else "data")
    for key in ("data_dir", "work_root"):
        options[key] = str(options[key].resolve())
    options.update(prepared_inputs=prepared,
        qubo_round_decimals=None if args.qubo_round_decimals == "none" else int(args.qubo_round_decimals))
    if args.threads < 1 or args.num_reads < 1 or not 0 <= args.min_coverage <= 1:
        p.error("Require threads/num-reads >=1 and 0 <= min-coverage <=1")
    work = Path(options["work_root"])
    if args.analyze_only:
        config = read_json(work / "batch_config.json")
        options, systems = config["options"].copy(), config["systems"]
        options["work_root"] = str(work)
        if args.data_dir is not None:
            options["data_dir"] = str(args.data_dir.resolve())
    else:
        data = Path(options["data_dir"])
        if not data.is_dir():
            p.error(f"No data directory: {data}")
        systems = list(dict.fromkeys(args.only)) if args.only else sorted(x.name for x in data.iterdir() if x.is_dir())
        if not systems or any(Path(s).name != s or s in (".", "..") for s in systems):
            p.error("Require nonempty, simple system directory names")
        dependencies = ("autogrid4",) if prepared else ("autogrid4", "prepare_receptor", "prepare_ligand", "obabel")
        tools = {name: shutil.which(name) for name in dependencies}
        if not all(tools.values()):
            p.error("Missing PATH dependencies: " + ", ".join(k for k, v in tools.items() if not v))
        if args.check_only:
            missing = [str(f) for sid in systems for f in input_paths(data, sid, prepared) if not f.is_file()]
            print(json.dumps(dict(systems=systems, tools=tools, missing=missing, options=options), indent=2))
            return int(bool(missing))
        if work.exists() and any(work.iterdir()):
            p.error(f"Refusing nonempty work-root: {work}; choose a new path or --analyze-only")
        if data == work or data in work.parents or work in data.parents:
            p.error("Keep output and input directory trees separate")
        work.mkdir(parents=True, exist_ok=True)
        config = dict(created_utc=datetime.now(timezone.utc).isoformat(), options=options, systems=systems,
            tools=tools, source_model_reused=False, solver="neal.sample_qubo", precision_method="none",
            reference="native input heavy atoms; fixed atom mapping, no superposition", top_ns=TOP_NS,
            thresholds_A=[t for t, _ in THRESHOLDS], strict_less_than=True, no_sample_deduplication=True,
            aggregate_denominator="all selected systems, including failed systems",
            input_sha256={str(f): file_hash(f) for sid in systems for f in input_paths(data, sid, prepared) if f.is_file()},
            code_sha256={str(f.relative_to(PROJECT)): file_hash(f) for f in
                [Path(__file__), PROJECT/"batch_qdock_pdbqt.py", *sorted((PROJECT/"QDock/GridPointMatching").glob("*.py"))]})
        write_json(work / "batch_config.json", config)
    manifest = relocated_input_hashes(config, options["data_dir"])
    if args.analyze_only:
        changed = changed_inputs(manifest)
        if changed:
            raise RuntimeError("Input files changed or missing; existing statistics untouched: " + ", ".join(changed))
    for key in THREAD_VARS:
        os.environ[key] = str(options["threads"])
    branches = ["raw"] + ([] if options["no_logical_postprocess"] else ["logical_fullprecision"])
    # Reanalysis writes only to a temporary tree until input hashes pass again.
    # Failed validation (including changes during analysis) cannot overwrite CSVs.
    staging = (tempfile.TemporaryDirectory(prefix=".reanalyze_", dir=work)
               if args.analyze_only else nullcontext(work))
    with staging as output_root:
        output = Path(output_root)
        rows = []
        for index, sid in enumerate(systems, 1):
            if args.analyze_only:
                runtime_path = work / f"{sid}.runtime.json"
                runtime = read_json(runtime_path) if runtime_path.exists() else {}
            else:
                command = core_command(options, sid)
                print(f"[{index}/{len(systems)}] {sid}: start neal ({options['num_reads']} reads)", flush=True)
                write_json(work / f"{sid}.command.json", command)
                start = time.perf_counter()
                with (work / f"{sid}.log").open("w") as log:
                    done = subprocess.run(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                runtime = dict(process_returncode=done.returncode, docking_seconds=time.perf_counter()-start)
                write_json(work / f"{sid}.runtime.json", runtime)
            # Preserve/analyze raw even if the worker's logical branch failed.
            result = [analyze_branch(work, sid, branch, options, runtime, output_root=output) for branch in branches]
            rows.extend(result)
            write_summaries(output, rows, len(systems), branches)
            print(f"[{index}/{len(systems)}] {sid}: " + "; ".join(
                f"{r['branch']}={r['status']}, qualified mRMSD={r.get('qualified_mrmsd')}" for r in result), flush=True)
        changed = changed_inputs(manifest)
        write_json(output / "batch_status.json", dict(status="finished", n_systems=len(systems),
            n_failed_branches=sum(r["status"] != "ok" for r in rows), input_files_unchanged=not changed,
            changed_input_files=changed, analysis_data_dir=options["data_dir"]))
        if changed:
            raise RuntimeError("Input files changed during batch: " + ", ".join(changed))
        if args.analyze_only:
            for source in sorted(output.rglob("*")):
                if source.is_file():
                    destination = work / source.relative_to(output)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
    return int(any(r["status"] != "ok" or r.get("process_returncode", 0) != 0 for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
