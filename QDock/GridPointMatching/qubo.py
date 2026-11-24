"""Candidate filtering and the unrounded AD4-like docking QUBO.

Numeric expression/accumulation order is retained for saved-result parity.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from pyqubo import Binary, Constraint


@dataclass
class QuboModel:
    qubo: dict[tuple[str, str], float]
    offset: float
    variable_names: list[str]
    atom_variables: dict[int, list[str]]
    weight_table: dict[str, float]
    ligand_name: str

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / (self.ligand_name + ".npy"), self.qubo)
        np.save(directory / (self.ligand_name + "_weights.npy"), self.weight_table)
        np.save(directory / (self.ligand_name + "_offset.npy"), [self.offset])


def _normalize_energy_range(range_value, name):
    """Normalize a (min, max) energy range. None means open on both sides."""
    if range_value is None:
        return (None, None)
    if not isinstance(range_value, (tuple, list, np.ndarray)) or len(range_value) != 2:
        raise ValueError("%s must be None or a tuple/list like (min, max)" % name)
    low, high = range_value
    low = None if low is None else float(low)
    high = None if high is None else float(high)
    if low is not None and high is not None and high < low:
        raise ValueError("%s upper bound must be >= lower bound" % name)
    return (low, high)


def _range_mask(values, energy_range):
    """Return a boolean mask for values inside an inclusive energy range."""
    low, high = energy_range
    mask = np.ones(values.shape, dtype=bool)
    if low is not None:
        mask &= values >= low
    if high is not None:
        mask &= values <= high
    return mask


def normalize_energies(energies, method='zscore', minmax_range=(-1.0, 1.0)):
    """
    Normalize energy values for balanced weighting.

    Args:
        energies: numpy array of energy values
        method: 'zscore' for z-score normalization, 'minmax' for min-max scaling
        minmax_range: target range for min-max scaling (default: [-1, 1])

    Returns:
        Normalized energy array
    """
    if len(energies) == 0:
        return energies

    if method == 'zscore':
        mean = np.mean(energies)
        std = np.std(energies)
        if std > 1e-10:
            return (energies - mean) / std
        else:
            return energies - mean
    elif method == 'minmax':
        if not isinstance(minmax_range, (tuple, list, np.ndarray)) or len(minmax_range) != 2:
            raise ValueError("minmax_range must be a tuple/list of length 2, e.g. (-1, 1)")
        low = float(minmax_range[0])
        high = float(minmax_range[1])
        if high <= low:
            raise ValueError("minmax_range upper bound must be greater than lower bound")

        min_e = np.min(energies)
        max_e = np.max(energies)
        if max_e - min_e > 1e-10:
            return (energies - min_e) * (high - low) / (max_e - min_e) + low
        else:
            return np.full_like(energies, 0.5 * (low + high), dtype=float)
    else:
        return energies


def build_qubo(ligand, grid, edge_cutoff, K_dist, K_onehot=None,
               K_mono=None, K_cov=None,
               w_affinity=1., w_elec=1., w_solv=1., normalize=None,
               affinity_range=None, elec_range=None, solv_range=None,
               energy_range=(None, 0.), minmax_range=(-1., 1.),
               combined_round_decimals=None, min_candidate_coverage=1.,
               save_qubo=True):
    """Build the UNROUNDED QUBO. Candidate/model mathematics match CIM_v0.

    Prepared and freshly prepared inputs enter the same model builder.
    No solver or precision preprocessing is run here.
    """
    # Build the original, unrounded objective in the established order.
    hamiltonian = 0
    vertices = []

    if K_onehot is not None:
        if K_mono is not None or K_cov is not None:
            raise ValueError("Use K_onehot OR independent K_mono/K_cov")
        K_onehot = float(K_onehot)
        if not np.isfinite(K_onehot) or K_onehot <= 0:
            raise ValueError("K_onehot must be finite and > 0")
        K_mono, K_cov = 2 * K_onehot, K_onehot
        hamiltonian += ligand.n * K_onehot  # Preserve the legacy constant offset.
    else:
        K_mono = float(25.09 if K_mono is None else K_mono)
        K_cov = float(0.0 if K_cov is None else K_cov)
    if not np.isfinite(K_mono) or K_mono <= 0 or not np.isfinite(K_cov) or K_cov < 0:
        raise ValueError("Require finite K_mono > 0 and K_cov >= 0")
    try:
        K_dist = float(K_dist)
    except (TypeError, ValueError) as exc:
        raise ValueError("K_dist must be a number") from exc
    if not np.isfinite(K_dist) or K_dist < 0.0:
        raise ValueError("K_dist must be finite and >= 0")

    if combined_round_decimals is not None:
        if isinstance(combined_round_decimals, bool) or not isinstance(combined_round_decimals, (int, np.integer)):
            raise ValueError("combined_round_decimals must be a non-negative integer or None")
        if combined_round_decimals < 0:
            raise ValueError("combined_round_decimals must be >= 0")

    affinity_range = _normalize_energy_range(affinity_range, "affinity_range")
    elec_range = _normalize_energy_range(elec_range, "elec_range")
    solv_range = _normalize_energy_range(solv_range, "solv_range")
    energy_range = _normalize_energy_range(energy_range, "energy_range")

    # Weight lookup table: maps variable name -> physical unary weight.
    weight_table = {}
    atom_to_variables = {i: [] for i in range(ligand.n)}

    # Filter independent terms first; combine/normalize only once.
    all_affinity = []
    all_elec = []
    all_solv = []
    vertex_info = []  # (atom index, grid index)

    for i in range(ligand.n):
        t = ligand.autodock_atom_types[i]
        q_i = ligand.qs[i]  # Partial charge of ligand atom

        # AutoDock4.2-like map decomposition.  The atom-type map already
        # contains weighted vdW, hamiltonian-bond, and charge-independent
        # desolvation.  e.map and d.map supply the two charge-dependent
        # terms and are already weighted by AutoGrid.
        affinity_all = grid.affinity[t]
        elec_all = grid.electrostatic * q_i
        solv_all = grid.desolvation * abs(q_i)

        candidate_mask = (
            _range_mask(affinity_all, affinity_range) &
            _range_mask(elec_all, elec_range) &
            _range_mask(solv_all, solv_range)
        )
        candidate_positions = np.argwhere(candidate_mask).flatten()

        print(f"[atom {i} {t} q={q_i:+.3f}] "
            f"aff raw: {affinity_all.min():.3f} .. {affinity_all.max():.3f} | "
            f"elec raw: {elec_all.min():.3f} .. {elec_all.max():.3f} | "
            f"solv raw: {solv_all.min():.3f} .. {solv_all.max():.3f}")

        for pos in candidate_positions:
            affinity_e = affinity_all[pos]
            # Electrostatic: charge * potential (AutoDock4 formula)
            elec_e = elec_all[pos]
            # Missing charge-dependent part of the AD4 desolvation score.
            solv_e = solv_all[pos]

            all_affinity.append(affinity_e)
            all_elec.append(elec_e)
            all_solv.append(solv_e)
            vertex_info.append((i, pos))

    if len(vertex_info) == 0:
        raise ValueError(
            "No candidate grid points found; adjust affinity/elec/solv ranges"
        )

    # Combine raw terms first. This preserves AutoDock-like relative term scales.
    all_affinity = np.array(all_affinity)
    all_elec = np.array(all_elec)
    all_solv = np.array(all_solv)

    combined_raw = (
        w_affinity * all_affinity +
        w_elec * all_elec +
        w_solv * all_solv
    )

    if normalize:
        combined_weights = normalize_energies(
            combined_raw,
            method=normalize,
            minmax_range=minmax_range
        )
    else:
        combined_weights = combined_raw

    if combined_round_decimals is not None:
        combined_weights = np.round(combined_weights, decimals=int(combined_round_decimals))

    # Apply combined range after raw term filters + normalization.
    selected_idx = np.argwhere(_range_mask(combined_weights, energy_range)).flatten()
    if len(selected_idx) == 0:
        raise ValueError(
            "No candidate grid points found after the combined energy range"
        )

    vertex_info = [vertex_info[idx] for idx in selected_idx]
    combined_weights = combined_weights[selected_idx]

    selected_atoms = {int(row[0]) for row in vertex_info}
    missing_atoms = sorted(set(range(ligand.n)) - selected_atoms)
    if not 0 < min_candidate_coverage <= 1:
        raise ValueError('min_candidate_coverage must be in (0, 1]')
    candidate_coverage = len(selected_atoms) / ligand.n
    with open('candidate_diagnostics.json', 'w') as handle:
        json.dump({'n_atoms': ligand.n, 'zero_candidate_atoms': missing_atoms,
                   'candidate_counts': np.bincount([int(r[0]) for r in vertex_info], minlength=ligand.n).tolist(),
                   'maximum_possible_coverage': candidate_coverage,
                   'min_candidate_coverage': min_candidate_coverage}, handle, indent=2)
    if candidate_coverage + 1e-12 < min_candidate_coverage:
        raise ValueError(
            "Candidate coverage %.6f below %.6f; zero-candidate atoms: %s"
            % (candidate_coverage, min_candidate_coverage, missing_atoms)
        )

    # Unary terms: physical grid energy before coverage reward.
    # Variable naming: "{atom_idx}_{grid_pos}" (no weight in name)
    for idx, (i, pos) in enumerate(vertex_info):
        weight = combined_weights[idx]

        var_name = f"{i}_{pos}"
        x = Binary(var_name)
        weight_table[var_name] = weight
        vertices.append((x, i, pos))
        atom_to_variables[int(i)].append((x, var_name))
        hamiltonian += weight * x**2

    print(f"QUBO: {len(vertices)} variables, {len(vertices) * (len(vertices) - 1) // 2} potential edges")

    if len(vertices) * (len(vertices) - 1) / 2 > 100000000:
        raise ValueError("QUBO potential-edge guard exceeded (100000000)")
    # Geometry penalty only couples DIFFERENT ligand atoms.
    distance_penalty = 0
    n_distance_edges = 0
    for p in range(len(vertices)):
        x1, i1, pos1 = vertices[p]
        for q in range(p + 1, len(vertices)):
            x2, i2, pos2 = vertices[q]
            distance_mismatch = abs(ligand.d_matrix[i1, i2] - grid.distances[pos1, pos2])
            if i1 != i2 and distance_mismatch > edge_cutoff:
                distance_penalty += x1 * x2
                n_distance_edges += 1
    # A zero distance Hamiltonian is valid but Constraint(0) is not a
    # pyqubo expression. Omitting this zero term leaves the model unchanged.
    if n_distance_edges:
        hamiltonian += K_dist * Constraint(distance_penalty, label='link')

    # Independent at-most-one penalty and linear selection reward.
    # Candidate filtering above uses physical energy, before this reward.
    for atom_index in range(ligand.n):
        atom_vars = [row[0] for row in atom_to_variables[atom_index]]
        count = sum(atom_vars)
        hamiltonian += (K_mono / 2.0) * (count ** 2 - count) - K_cov * count
    atom_variable_names = {
        atom_index: [name for _, name in rows]
        for atom_index, rows in atom_to_variables.items()
    }
    model = hamiltonian.compile()
    qubo, offset = model.to_qubo()

    result = QuboModel(qubo, float(offset), sorted({v for pair in qubo for v in pair}),
                       atom_variable_names, weight_table, ligand.name)
    if save_qubo:
        result.save("QUBOs")
    return result
