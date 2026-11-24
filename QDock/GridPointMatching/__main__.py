"""Single-system entry; no batch, memory supervision or summary analysis."""
import argparse
from dataclasses import asdict
from pathlib import Path
import os

from .qdock import GPMDock, write_json
from .solver import FastSAConfig, NealConfig, PrecisionConfig


def optional_int(value):
    return None if value.lower() == "none" else int(value)


def optional_float(value):
    return None if value.lower() == "none" else float(value)


def parser():
    p = argparse.ArgumentParser(prog='python -m QDock.GridPointMatching', description=__doc__)
    p.add_argument("--receptor", required=True, help="PDB; native PDBQT with --prepared-inputs")
    p.add_argument("--ligand", required=True, help="MOL2; native PDBQT with --prepared-inputs")
    p.add_argument("--prepared-inputs", action="store_true")
    p.add_argument("--work-dir", required=True, help="New or empty output directory")
    p.add_argument("--center-length", type=float, default=8.)
    p.add_argument("--grid-length", type=float, default=2.)
    p.add_argument("--edge-cutoff", type=float, required=True)
    p.add_argument("--K-dist", type=float, required=True)
    p.add_argument("--K-onehot", type=optional_float, default=None)
    p.add_argument("--K-mono", type=optional_float, default=None)
    p.add_argument("--K-cov", type=optional_float, default=None)
    for term in ("affinity", "elec", "solv"):
        p.add_argument("--w-" + term, type=float, default=1.)
    p.add_argument("--normalize", choices=["none", "zscore", "minmax"], default="none")
    for term in ("affinity", "elec", "solv", "energy"):
        p.add_argument("--" + term + "-min", type=optional_float, default=None)
        p.add_argument("--" + term + "-max", type=optional_float,
                       default=5. if term == "affinity" else 0. if term == "energy" else None)
    p.add_argument("--combined-round-decimals", type=optional_int, default=None)
    p.add_argument("--min-candidate-coverage", type=float, default=.9)
    p.add_argument("--qubo-round-decimals", type=optional_int, default=2)
    p.add_argument("--solver", choices=["neal", "fastsa"], default="neal")
    p.add_argument("--precision-method", choices=["none", "split"],
                   help="Default: split for FastSA, none for neal; neal does not support split")
    p.add_argument("--precision", type=int, default=14)
    p.add_argument("--max-ising-size", type=optional_int, default=11999)
    p.add_argument("--no-logical-postprocess", action="store_true")
    p.add_argument("--logical-tolerance", type=float, default=1e-9)
    p.add_argument("--logical-max-steps", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=.8)
    p.add_argument("--iterations-per-t", type=int, default=5)
    p.add_argument("--size-limit", type=int, default=100)
    p.add_argument("--n-pos", type=int, default=30,
                   help="Legacy minimum sampling budget: max(size_limit, 3*n_pos); NO truncation")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--process-num", type=int, default=1)
    p.add_argument("--thread-num", type=int, default=128)
    p.add_argument("--num-reads", type=int, default=100, help="neal only: number of independent reads")
    p.add_argument("--neal-seed", type=int, help="neal only: optional seed; omitted uses SDK default")
    p.add_argument("--build-only", action="store_true", help="Prepare/maps/build QUBO without SDK calls")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.precision_method is None:
        args.precision_method = "none" if args.solver == "neal" else "split"
    if args.solver == "neal" and args.precision_method != "none":
        raise ValueError("--solver neal requires --precision-method none (no Kaiwu/split)")
    receptor, ligand = Path(args.receptor).resolve(), Path(args.ligand).resolve()
    if not receptor.is_file() or not ligand.is_file():
        raise FileNotFoundError("Both receptor and ligand inputs must exist")
    precision = PrecisionConfig(
        qubo_round_decimals=args.qubo_round_decimals,
        precision_method=args.precision_method, precision=args.precision,
        max_ising_size=args.max_ising_size, logical_postprocess=not args.no_logical_postprocess,
        logical_tolerance=args.logical_tolerance, logical_max_steps=args.logical_max_steps,
    )
    fastsa = FastSAConfig(
        alpha=args.alpha, iterations_per_t=args.iterations_per_t, size_limit=args.size_limit,
        n_pos=args.n_pos, seed=args.seed, process_num=args.process_num, thread_num=args.thread_num,
    )
    neal = NealConfig(num_reads=args.num_reads, seed=args.neal_seed)
    precision.validate()
    if args.solver == "neal":
        neal.sampler_arguments()
    else:
        fastsa.optimizer_arguments()
    work = Path(args.work_dir).resolve()
    if work.exists() and any(work.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty work directory: " + str(work))
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / "run_config.json",
               dict(args=vars(args), precision=asdict(precision), fastsa=asdict(fastsa),
                    neal=asdict(neal), solver_type=args.solver,
                    receptor=str(receptor), ligand=str(ligand)))
    previous = Path.cwd()
    try:
        os.chdir(work)
        dock = GPMDock()
        dock.make_receptor(str(receptor), prepared=args.prepared_inputs)
        dock.make_ligand([str(ligand)], prepared=args.prepared_inputs)
        dock.make_box_ligand(str(ligand), center_length=args.center_length,
                            grid_length=args.grid_length, prepared=args.prepared_inputs)
        options = {term + "_range": (getattr(args, term + "_min"), getattr(args, term + "_max"))
                   for term in ("affinity", "elec", "solv", "energy")}
        results = dock.dock(
            args.edge_cutoff, args.K_dist, K_onehot=args.K_onehot,
            K_mono=args.K_mono, K_cov=args.K_cov,
            w_affinity=args.w_affinity, w_elec=args.w_elec, w_solv=args.w_solv,
            normalize=None if args.normalize == "none" else args.normalize,
            combined_round_decimals=args.combined_round_decimals,
            min_candidate_coverage=args.min_candidate_coverage,
            precision_config=precision, fastsa_config=fastsa,
            solver_type=args.solver, neal_config=neal,
            sim_dock=not args.build_only, **options)
        if not args.build_only:
            for result in results:
                print("[COMPLETE] raw states=%d, raw poses=%d, logical poses=%s" % (
                    len(result.solutions.physical_states), len(result.branches["raw"].poses),
                    len(result.branches["logical_fullprecision"].poses)
                    if "logical_fullprecision" in result.branches else "not available"), flush=True)
                if result.solutions.post_error:
                    raise RuntimeError("Raw preserved; logical postprocessing failed: " + result.solutions.post_error)
        return 0
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    raise SystemExit(main())
