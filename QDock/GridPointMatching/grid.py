"""AutoGrid execution, AD4 map loading and grid coordinates."""
from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass
class Grid:
    coordinates: np.ndarray
    distances: np.ndarray
    affinity: dict[str, np.ndarray]
    electrostatic: np.ndarray
    desolvation: np.ndarray


def read_map(path):
    """Read one AutoGrid map; its AD4 coefficients are already included."""
    with open(path) as stream:
        return np.asarray([float(line) for line in stream.readlines()[6:]])


def make_grid(receptor, ligands, center, lengths, spacing):
    """Generate the same box/lattice and maps as the established 2Å workflow."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("grid_length must be positive and finite")
    # AutoGrid receives five decimal places. Use that exact spacing for the
    # dimensions and recovered coordinates too, not the unrounded CLI value.
    spacing = float(f"{spacing:.5f}")
    if spacing <= 0:
        raise ValueError("grid_length rounds to zero at AutoGrid's five decimal places")
    dims = (np.asarray(lengths) / spacing).astype(np.int64)
    dims[dims % 2 == 0] -= 1
    if np.any(dims < 1):
        raise ValueError("Box is smaller than one grid point")
    receptor_types = np.unique(receptor.autodock_atom_types)
    ligand_types = np.unique(np.hstack([ligand.autodock_atom_types for ligand in ligands]))
    text = (
        "outlev 1\n"
        f"npts {dims[0]-1} {dims[1]-1} {dims[2]-1}\n"
        f"gridfld {receptor.name}.fld # avs field file\n"
        f"spacing {spacing:.5f}\n"
        "qasp 0.01097\n"
        f"receptor_types {' '.join(receptor_types.tolist())}\n"
        f"ligand_types {' '.join(ligand_types.tolist())}\n"
        f"receptor {receptor.pdbqt_path}\n"
        f"gridcenter {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n"
        "smooth 0.500000\n"
    )
    text += "".join(f"map Receptor.{atom_type}.map\n" for atom_type in ligand_types)
    text += "elecmap Receptor.e.map\ndsolvmap Receptor.d.map\ndielectric -0.1465\n"
    gpf, log = Path(f"{receptor.name}.gpf"), Path(f"{receptor.name}_grid.log")
    gpf.write_text(text)
    status = subprocess.run(["autogrid4", "-p", str(gpf), "-l", str(log)], check=False)
    if status.returncode:
        tail = "\n".join(log.read_text().splitlines()[-30:]) if log.exists() else "No AutoGrid log"
        raise RuntimeError(f"AutoGrid failed for {receptor.name}:\n{tail}")
    xyz = Path(f"{receptor.name}.xyz")
    if not xyz.is_file():
        raise RuntimeError(f"AutoGrid did not create {xyz}")
    lines = [line for line in xyz.read_text().splitlines() if line.split()]
    if len(lines) < 3:
        raise RuntimeError(f"Incomplete AutoGrid geometry: {xyz}")
    x0, y0, z0 = [float(line.split()[0]) for line in lines[:3]]
    # Preserve x-fastest indexing and use the actual rounded AutoGrid origin.
    coordinates = np.asarray([
        [x0 + x * spacing, y0 + y * spacing, z0 + z * spacing]
        for z in range(dims[2]) for y in range(dims[1]) for x in range(dims[0])
    ])
    return Grid(
        coordinates, squareform(pdist(coordinates)),
        {atom_type: read_map(f"Receptor.{atom_type}.map") for atom_type in ligand_types},
        read_map("Receptor.e.map"), read_map("Receptor.d.map"),
    )
