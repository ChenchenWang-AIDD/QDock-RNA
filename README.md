# QDock-RNA: docking examples with the neal solver

This repository provides example code for building and solving QDock Grid Point
Matching (GPM) docking problems with the classical simulated annealing solver
**neal**. It includes receptor/ligand preparation, AutoGrid map generation, QUBO
construction, sampling, logical postprocessing, rigid pose recovery, and RMSD
analysis.

Two input routes use the same docking implementation:

| Route | Included examples | Input files | Entry point |
|---|---|---|---|
| Prepare from PDB/MOL2 | Protein complexes: `1y6r`, `2iwx`, `3dxg` | `receptor.pdb`, `ligand.mol2` | `batch_qdock.py` |
| Use prepared PDBQT | RNA complexes: `3g4m`, `2kxm`, `3suh` | `receptor.pdbqt`, `ligand.pdbqt` | `batch_qdock_pdbqt.py` |

The default workflow rounds QUBO coefficients to two
decimal places, samples the QUBO with neal, and applies logical descent to the
original **unrounded** objective.

## Installation

### 1. Create the environment


```bash
mamba env create -f environment.yml
mamba activate qdock_rna
```

The environment includes `dwave-neal` and `dwave-samplers`. The code uses
`dwave.samplers.SimulatedAnnealingSampler.sample_qubo`.

### 2. Install ADFRsuite

Download the Linux package from the
[official ADFRsuite release](https://github.com/sannerlab/downloads/releases/tag/adfrsuite-1.0)
and complete the installation following the included instructions.

After installation, activate the environment and append the installed ADFRsuite
`bin` directory to `PATH`, replacing the example path with your installation:

```bash
mamba activate qdock_rna
export PATH="$PATH:/path/to/ADFRsuite-1.0/bin"
command -v python prepare_receptor prepare_ligand autogrid4 obabel
```

## Example 1: build from receptor PDB and ligand MOL2

This route uses the three protein examples in `data/`. It prepares PDBQT inputs
with ADFRsuite, calculates AutoGrid maps, builds and samples a new QUBO, and
analyzes the recovered raw and logically postprocessed poses.


```bash
# Check input files and required external executables.
python batch_qdock.py --check-only

# Run all three protein examples with the default parameters.
python -u batch_qdock.py --work-root work_neal_protein
```
## Example 2: build from prepared receptor and ligand PDBQT

This route uses the three RNA examples in `data_pdbqt/`. It preserves the supplied
PDBQT charges and AutoDock atom types and skips only PDBQT preparation. Box
construction, AutoGrid, candidate filtering, QUBO construction, sampling, logical
postprocessing, and analysis still run.

```bash
python batch_qdock_pdbqt.py --check-only
python -u batch_qdock_pdbqt.py --work-root work_neal_rna
```

## Basic parameters

The following options apply to **both batch scripts**. Distances are in Å.

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | Bundled `data` or `data_pdbqt` | Input directory containing one subdirectory per system. |
| `--only` | All system subdirectories | Space-separated system IDs to run. |
| `--work-root` | `work_neal_prepare` or `work_neal_pdbqt` | Output directory; must be new or empty. |
| `--center-length` | `8.0` | Padding added to each full axis length around the ligand coordinate extent; not the total box size. |
| `--grid-length` | `2.0` | Candidate-grid spacing. |
| `--edge-cutoff` | `2.182` | Allowed ligand/grid pair-distance mismatch before a distance penalty applies. |
| `--K-dist` | `0.107` | Distance-incompatibility penalty. |
| `--K-mono` | `31.823` | Penalty for selecting multiple grid points for one ligand atom. |
| `--K-cov` | `0.888` | Linear reward for selecting atom–grid assignments. |
| `--w-affinity`, `--w-elec`, `--w-solv` | `1.0`, `0.507`, `1.0` | Multipliers for the three AutoGrid-derived energy contributions. |
| `--affinity-max` | `5.0` | Discard candidates whose affinity-map energy exceeds this value. |
| `--energy-max` | `0.0` | Discard candidates whose combined weighted energy exceeds this value. |
| `--min-candidate-coverage` | `0.9` | Minimum fraction of modeled ligand atoms with at least one candidate, required before solving. |
| `--qubo-round-decimals` | `2` | Round assembled QUBO coefficients to `2` or `3` decimals; use `none` to disable. |
| `--max-ising-size` | `11999` | Maximum unsplit Ising dimension, equal to the QUBO variable count plus one. |
| `--num-reads` | `100` | Number of neal samples; duplicate outputs are retained. |
| `--neal-seed` | Not set | Optional random seed. |
| `--threads` | `1` | BLAS/OpenMP thread limit; systems are processed sequentially. |
| `--no-logical-postprocess` | Not set | Disable logical postprocessing; it is enabled by default. |
| `--logical-tolerance` | `1e-9` | Minimum strict energy decrease accepted during logical descent. |
| `--logical-max-steps` | `10000` | Maximum accepted logical flips per state. |
| `--min-coverage` | `0.9` | Coverage threshold for qualified poses in the result statistics. |

Use `python batch_qdock.py --help` or `python batch_qdock_pdbqt.py --help`
for all available options.

## Results and analysis

Both scripts generate separate `raw` and `logical_fullprecision` branches.
Logical postprocessing flips original decision variables to decrease the
unrounded objective.

| Output | Contents |
|---|---|
| `summary.csv` | Per-system, per-branch status and docking metrics. |
| `raw_rmsd_summary.csv` | Raw-branch statistics. |
| `logical_fullprecision_rmsd_summary.csv` | Logical-branch statistics, when enabled. |
| `aggregate_summary.csv` | Sampling and Top1/3/5/10 success rates at RMSD <2.0 Å and <2.5 Å. |
| `failures.csv` | Failed or unrecoverable system/branch records. |
| `<id>/<branch>/rmsd_per_pose.csv` | Per-output RMSD, energy, coverage, duplicate selections, and qualification. |
| `<id>/<branch>/ligand_poses.pdb` | Recovered ligand poses for the supplied example filenames. |
| `<id>.log`, `batch_config.json` | Execution diagnostics, parameters, input hashes, and provenance. |

For these redocking examples, the input ligand supplies both the box reference
and the reference pose for evaluation.

By default, a qualified pose must be recoverable, cover at least 90% of the modeled ligand
atoms, and have **no atom assigned to multiple grid points**. Modeled atoms
include retained polar hydrogens; RMSD uses heavy atoms only. Statistics include
both all recovered poses and the qualified subset.
