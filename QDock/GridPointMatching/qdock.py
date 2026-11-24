"""Docking workflow and output persistence.

Input -> maps -> QUBO -> FastSA/optional split or neal -> logical/rigid recovery.
Numerical work lives in the corresponding modules, not in an outer batch script.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from .structure import Receptor, Ligand
from .grid import make_grid
from .qubo import QuboModel, build_qubo
from .solver import (
    PrecisionConfig, FastSAConfig, NealConfig, IsingProblem, SolveResult,
    prepare_ising, solve_fastsa, solve_neal,
)
from .postprocess import PoseBranch, recover_branch


def write_json(path, value):
    """Serialize numeric run artifacts, including NumPy scalars/arrays."""
    def encode(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + "\n")


@dataclass
class DockingResult:
    model: QuboModel
    ising: IsingProblem
    solutions: SolveResult
    branches: dict[str, PoseBranch] = field(default_factory=dict)


class GPMDock:
    """Stateful, single-work-directory docking workflow.

    Different systems must use different processes/directories. The CLI handles
    the work directory; Python callers choose their own working directory.
    """
    def __init__(self):
        self.receptor = None
        self.ligands = []
        self.grid = None
        self.results = []

    @property
    def box_coords(self):
        if self.grid is None:
            raise RuntimeError("Create the box/maps first")
        return self.grid.coordinates

    def make_receptor(self, path, prepared=False):
        if Path(path).suffix.lower() != (".pdbqt" if prepared else ".pdb"):
            raise ValueError("Receptor must be PDBQT" if prepared else "Receptor must be PDB")
        self.receptor = Receptor(path, "receptor.pdbqt", prepared=prepared)
        self.ligands, self.grid = [], None

    def make_ligand(self, paths, prepared=False):
        if self.receptor is None:
            raise RuntimeError("Create the receptor first")
        paths = list(paths)
        if not paths:
            raise ValueError("At least one ligand is required")
        names = [Path(path).stem for path in paths]
        if len(set(names)) != len(names):
            raise ValueError("Ligand filenames must have unique stems; rename duplicate "
                             "basenames or use separate work directories")
        # Validate the entire list before preparation can overwrite any file.
        for name in names:
            for suffix in (".pdbqt", ".pdb"):
                target = Path("Ligands") / (name + suffix)
                if target.exists() or target.is_symlink():
                    raise FileExistsError(f"Refusing existing prepared ligand: {target}; use a new work directory")
        self.ligands = [Ligand(path, "Ligands", prepared=prepared) for path in paths]
        self.grid = None

    def make_box_ligand(self, path, center_length=8., grid_length=2., prepared=False):
        if not self.ligands:
            raise RuntimeError("Create receptor and ligands first")
        reference = Ligand(path, "Box_Rawligand", prepared=prepared)
        coordinates = reference.coords
        if prepared:
            coordinates = coordinates[~np.isin(reference.autodock_atom_types, ["H", "HD", "HS"])]
        if not len(coordinates):
            raise ValueError("Cannot build a box from an empty/heavy-atom-free ligand")
        center = np.mean(coordinates, axis=0)
        lengths = center_length + 2 * np.max(np.abs(coordinates - center), axis=0)
        if prepared:
            lengths = np.minimum(lengths, 30.)
        self.grid = make_grid(self.receptor, self.ligands, center, lengths, grid_length)

    def make_box_input(self, x, y, z, dx, dy, dz, grid_length=1.):
        if not self.ligands:
            raise RuntimeError("Create receptor and ligands first")
        self.grid = make_grid(
            self.receptor, self.ligands, np.array([x, y, z]), np.array([dx, dy, dz]), grid_length,
        )

    def build_qubo(self, ligand, edge_cutoff, K_dist, **options):
        if self.grid is None:
            raise RuntimeError("Create the receptor, ligand and box/maps first")
        return build_qubo(ligand, self.grid, edge_cutoff, K_dist, **options)

    def dock(self, edge_cutoff, K_dist, *, precision_config=None, fastsa_config=None,
             sim_dock=True, save_qubo=True, save_match=True, save_pose=True,
             optimizer=None, solver_type="neal", neal_config=None, **model_options):
        """Return models (build-only) or raw/post DockingResult objects.

        All solver-returned rows are retained. No summary analysis, feasibility
        filtering, RMSD-based refinement, batch scheduling or CIM calls occur.
        Like the CLI, Python callers must use a fresh output directory for each
        run. Existing docking artifacts are rejected before model construction.
        """
        if not self.ligands or self.grid is None:
            raise RuntimeError("Create the receptor, ligands and box/maps first")
        if solver_type not in ("fastsa", "neal"):
            raise ValueError("solver_type must be fastsa or neal")
        precision_config = precision_config or PrecisionConfig(
            precision_method="none" if solver_type == "neal" else "split")
        fastsa_config = fastsa_config or FastSAConfig()
        precision_config.validate()
        if solver_type == "neal":
            neal_config = neal_config or NealConfig()
            neal_config.sampler_arguments()
            if precision_config.precision_method != "none":
                raise ValueError("neal requires precision_method='none'; split is unsupported")
            if optimizer is not None:
                raise ValueError("optimizer injection is for FastSA/Ising adapters, not direct neal QUBO")
        else:
            fastsa_config.optimizer_arguments()
        # Input preparation/maps may already exist in the current directory;
        # only result artifacts are forbidden. Check before writing even QUBOs.
        for name in ("Docking", "IsingMatrices", "raw", "logical_fullprecision",
                     "solver_raw_returned.npy", "solver_metadata.json", "copy_conflicts.npy",
                     "logical_descent.json", "postprocess_status.json", "box_coords.npy",
                     "ligand_coords.npy", "ligand_metadata.json"):
            target = Path(name)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing existing docking results: {target}; use a new work directory")
        results = []
        for ligand in self.ligands:
            model = self.build_qubo(
                ligand, edge_cutoff, K_dist, save_qubo=save_qubo, **model_options,
            )
            if not sim_dock:
                results.append(model)
                continue
            root = Path(".") if len(self.ligands) == 1 else Path("Docking") / ligand.name
            root.mkdir(parents=True, exist_ok=True)
            problem = prepare_ising(
                model.qubo, precision_config,
                progress=lambda stage, data: print("[SPLIT]", stage, data, flush=True),
                solver_type=solver_type,
            )
            _save_ising(problem, root / "IsingMatrices")
            print(f"[{solver_type}] original={len(problem.original_full)} "
                  f"final={len(problem.solver_matrix)}", flush=True)
            def save_returned(states, metadata):
                # Persist before energy validation and optional descent, not after pose recovery.
                np.save(root / "solver_raw_returned.npy", np.asarray(states))
                write_json(root / "solver_metadata.json", metadata)

            if solver_type == "neal":
                solutions = solve_neal(problem, model.qubo, neal_config, precision_config,
                                       offset=model.offset, on_sample=save_returned)
            else:
                solutions = solve_fastsa(problem, fastsa_config, precision_config,
                    offset=model.offset, optimizer=optimizer, on_sample=save_returned)
            # Commit raw/post states before attempting pose serialization.
            _save_solutions(solutions, root)
            result = DockingResult(model, problem, solutions)
            for name, branch in solutions.branches.items():
                poses = recover_branch(name, branch, problem.variable_names, ligand, self.box_coords)
                _save_poses(poses, root / name, ligand, save_match, save_pose)
                result.branches[name] = poses
            np.save(root / "box_coords.npy", self.box_coords)
            np.save(root / "ligand_coords.npy", ligand.coords)
            write_json(root / "ligand_metadata.json", {
                "name": ligand.name, "atom_types": ligand.autodock_atom_types,
                "charges": ligand.qs, "n_atoms": ligand.n,
            })
            results.append(result)
        self.results = results
        return results


def _save_ising(problem, directory):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "ising_original_full.npy", problem.original_full)
    np.save(directory / "var_names.npy", np.asarray(problem.variable_names))
    if problem.quantized_integer is not None:
        np.save(directory / "ising_quantized_integer.npy", problem.quantized_integer)
    np.save(directory / "ising_solver.npy", problem.solver_matrix)
    np.save(directory / "restore_last_var_idx.npy", problem.restore_ends)
    write_json(directory / "preprocess.json", problem.metadata)


def _save_solutions(solutions, root):
    np.save(root / "solver_raw_returned.npy", solutions.physical_states)
    np.save(root / "copy_conflicts.npy", solutions.copy_conflicts)
    if solutions.solver_metadata is not None:
        write_json(root / "solver_metadata.json", solutions.solver_metadata)
    for name, branch in solutions.branches.items():
        directory = root / name
        directory.mkdir(exist_ok=True)
        np.save(directory / "states.npy", branch.states)
        np.save(directory / "ranking_energies.npy", branch.energies)
        np.save(directory / "qubo_energies.npy", branch.qubo_energies)
        write_json(directory / "state_space.json", {
            "saved_states_space": "restored logical", "ranking_target": branch.ranking_target,
        })
    if solutions.logical_diagnostics is not None:
        write_json(root / "logical_descent.json", solutions.logical_diagnostics)
    status = ("failed" if solutions.post_error else
              "ok" if "logical_fullprecision" in solutions.branches else "disabled")
    write_json(root / "postprocess_status.json", {
        "status": status, "error": solutions.post_error, "raw_post_pools_combined": False,
    })


def _save_poses(branch, directory, ligand, save_match, save_pose):
    directory.mkdir(parents=True, exist_ok=True)
    records, pose_model = [], 0
    for record in branch.records:
        if record.coordinates is not None:
            pose_model += 1
        records.append({
            "source_index": record.source_index, "energy_rank": record.energy_rank,
            "ranking_energy": record.ranking_energy, "qubo_energy": record.qubo_energy,
            "selection_counts": record.selection_counts, "recovery_status": record.recovery_status,
            "pose_model": pose_model if record.coordinates is not None else None,
            "selected_variables": record.selected_variables,
        })
    if save_match:
        write_json(directory / "matches.json", records)
    poses = branch.poses
    if save_pose and poses:
        import prody
        np.save(directory / "pose_coordinates.npy", np.asarray(poses))
        np.save(directory / "pose_source_indices.npy", np.asarray([
            record.source_index for record in branch.records if record.coordinates is not None
        ], dtype=np.int64))
        structure = ligand.ligand.copy()
        structure.setCoords(poses[0])
        if len(poses) > 1:
            structure.addCoordset(np.asarray(poses[1:]))
        prody.writePDB(str(directory / f"{ligand.name}_poses.pdb"), structure)
