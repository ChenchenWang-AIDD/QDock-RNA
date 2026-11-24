"""PDBQT preparation and atom parsing shared by receptors and ligands.

Prepared input preserves native atom order, charges and AutoDock types.
External preparation runs only for ordinary PDB/MOL2 input.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import numpy as np
import prody
from scipy.spatial.distance import pdist, squareform


def _float_token(value):
    try:
        return float(value)
    except ValueError:
        return None


def _coordinate_tokens(values):
    if not all(("." in value or "e" in value.lower()) for value in values):
        return None
    parsed = [_float_token(value) for value in values]
    if any(value is None for value in parsed):
        return None
    return parsed


def _find_pdbqt_coord_start(parts):
    for start in (5, 6):
        if len(parts) >= start + 3:
            values = _coordinate_tokens(parts[start:start + 3])
            if values is not None:
                return start, values

    for start in range(5, max(5, len(parts) - 2)):
        values = _coordinate_tokens(parts[start:start + 3])
        if values is not None:
            return start, values
    return None, None


def _parse_pdbqt_atom_line(line, path, line_no):
    parts = line.split()

    try:
        coords = [
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ]
    except ValueError:
        coord_start, coords = _find_pdbqt_coord_start(parts)

        if coord_start is None:
            raise ValueError(
                "Could not parse coordinates from %s:%s: %s"
                % (path, line_no, line.rstrip())
            )

    if len(parts) < 2:
        raise ValueError(
            "Could not parse charge/type from %s:%s: %s"
            % (path, line_no, line.rstrip())
        )

    charge = _float_token(parts[-2])
    if charge is None:
        try:
            charge = float(line[66:76])
        except ValueError:
            raise ValueError(
                "Could not parse charge from %s:%s: %s"
                % (path, line_no, line.rstrip())
            )

    atom_type = parts[-1].strip()
    if not atom_type:
        atom_type = line[77:79].strip()
    if not atom_type:
        raise ValueError(
            "Could not parse AutoDock atom type from %s:%s: %s"
            % (path, line_no, line.rstrip())
        )

    return coords, charge, atom_type


def _short_field(value, width):
    return str(value or "")[:width]


def _parse_pdbqt_metadata(line, parts, coord_start):
    fixed_coords = [
        _float_token(line[30:38]),
        _float_token(line[38:46]),
        _float_token(line[46:54]),
    ]
    use_fixed = (
        all(value is not None for value in fixed_coords)
        and _float_token(line[66:76]) is not None
    )

    record = line[:6].strip() or (parts[0] if parts else "ATOM")
    serial = line[6:11].strip() or (parts[1] if len(parts) > 1 else "")
    atom_name = line[12:16].strip()
    altloc = line[16:17].strip()
    resname = line[17:20].strip()
    chain = line[21:22].strip()
    resseq = line[22:26].strip()
    icode = line[26:27].strip()

    if not use_fixed and coord_start is not None and len(parts) > 2:
        atom_token = parts[2]
        atom_name = atom_token
        altloc = ""
        resname = ""
        chain = ""
        resseq = ""
        icode = ""

        if coord_start == 6 and len(parts) >= 6:
            res_token = parts[3]
            if len(res_token) > 3 and res_token[-3:].isalpha():
                altloc = res_token[:-3][-1:]
                resname = res_token[-3:]
            else:
                resname = res_token
            chain = parts[4]
            resseq = parts[5]
        elif coord_start == 5 and len(parts) >= 5:
            prefix = atom_token[:-3] if len(atom_token) > 4 and atom_token[-3:].isalpha() else atom_token
            if len(atom_token) > 4 and atom_token[-3:].isalpha():
                resname = atom_token[-3:]
                if len(prefix) > 4:
                    altloc = prefix[-1:]
                    atom_name = prefix[:-1]
                else:
                    atom_name = prefix
            if len(parts[3]) == 1:
                chain = parts[3]
                resseq = parts[4]
            else:
                resname = parts[3]
                resseq = parts[4]

    return {
        "record": record,
        "serial": serial,
        "atom_name": atom_name,
        "altloc": altloc,
        "resname": resname,
        "chain": chain,
        "resseq": resseq,
        "icode": icode,
    }


def _format_pdbqt_atom_line(line, path, line_no):
    parts = line.split()
    coords, charge, atom_type = _parse_pdbqt_atom_line(line, path, line_no)
    coord_start, _ = _find_pdbqt_coord_start(parts)
    metadata = _parse_pdbqt_metadata(line, parts, coord_start)

    occ = _float_token(line[54:60])
    temp = _float_token(line[60:66])
    if coord_start is not None and len(parts) > coord_start + 4:
        occ = _float_token(parts[coord_start + 3]) if occ is None else occ
        temp = _float_token(parts[coord_start + 4]) if temp is None else temp
    occ = 1.0 if occ is None else occ
    temp = 0.0 if temp is None else temp

    x, y, z = coords
    return (
        "%-6s%5s %-4s%1s%3s %1s%4s%1s   "
        "%8.3f%8.3f%8.3f%6.2f%6.2f    %6.3f %-2s\n"
        % (
            _short_field(metadata["record"], 6),
            _short_field(metadata["serial"], 5),
            _short_field(metadata["atom_name"], 4),
            _short_field(metadata["altloc"], 1),
            _short_field(metadata["resname"], 3),
            _short_field(metadata["chain"], 1),
            _short_field(metadata["resseq"], 4),
            _short_field(metadata["icode"], 1),
            x,
            y,
            z,
            occ,
            temp,
            charge,
            _short_field(atom_type, 2),
        )
    )


def _normalize_pdbqt_atom_columns(path):
    lines = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if line[0:6] in ["ATOM  ","HETATM"]:
                lines.append(_format_pdbqt_atom_line(line, path, line_no))
            else:
                lines.append(line)

    with open(path, "w") as f:
        f.writelines(lines)



def read_pdbqt(path):
    coordinates, charges, atom_types = [], [], []
    with open(path) as stream:
        for number, line in enumerate(stream, 1):
            if line.startswith(("ATOM  ", "HETATM")):
                coord, charge, atom_type = _parse_pdbqt_atom_line(line, path, number)
                coordinates.append(coord)
                charges.append(charge)
                atom_types.append(atom_type)
    if not coordinates:
        raise ValueError(f"No atoms in PDBQT: {path}")
    return np.asarray(coordinates), np.asarray(charges), np.asarray(atom_types)


def _copy_prepared(source, destination):
    source, destination = Path(source), Path(destination)
    if source.suffix.lower() != ".pdbqt":
        raise ValueError("Prepared input must be PDBQT")
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


class Receptor:
    def __init__(self, path, pdbqt_path, prepared=False):
        self.name = Path(path).name.split(".")[0]
        self.pdbqt_path = os.fspath(pdbqt_path)
        if prepared:
            _copy_prepared(path, self.pdbqt_path)
        else:
            self.make_pdbqt(path, self.pdbqt_path)
        self.coords, self.qs, self.autodock_atom_types = read_pdbqt(self.pdbqt_path)

    @staticmethod
    def make_pdbqt(source, destination):
        subprocess.run(
            ["prepare_receptor", "-r", str(source), "-o", str(destination), "-A", "hydrogens"],
            check=True,
        )
        _normalize_pdbqt_atom_columns(destination)


class Ligand:
    def __init__(self, path, directory, prepared=False):
        self.name = Path(path).stem
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.pdbqt_path = str(directory / f"{self.name}.pdbqt")
        self.pdb_path = str(directory / f"{self.name}.pdb")
        if prepared:
            _copy_prepared(path, self.pdbqt_path)
        else:
            self.make_pdbqt(path, self.pdbqt_path)
        self.coords, self.qs, self.autodock_atom_types = read_pdbqt(self.pdbqt_path)
        if prepared:
            self._write_coordinate_pdb()
        else:
            subprocess.run(["obabel", self.pdbqt_path, "-O", self.pdb_path], check=True)
        self.n = len(self.coords)
        self.d_matrix = squareform(pdist(self.coords))
        self.ligand = prody.parsePDB(self.pdb_path)

    @staticmethod
    def make_pdbqt(source, destination):
        # ADFR reads the basename, so stage locally rather than running in the
        # input dataset or requiring an outer batch script to copy the ligand.
        source, destination = Path(source).resolve(), Path(destination).resolve()
        with tempfile.TemporaryDirectory(prefix="qdock_ligand_prepare_") as staging:
            shutil.copy2(source, Path(staging) / source.name)
            subprocess.run(
                ["prepare_ligand", "-l", source.name, "-o", str(destination)],
                cwd=staging, check=True,
            )

    def _write_coordinate_pdb(self):
        # Coordinate transport for ProDy, without chemical retyping/reordering.
        elements = {"A": "C", "NA": "N", "OA": "O", "SA": "S", "HD": "H", "HS": "H"}
        with open(self.pdbqt_path) as source, open(self.pdb_path, "w") as target:
            for line in source:
                if line.startswith(("ATOM  ", "HETATM")):
                    atom_type = line.split()[-1]
                    element = elements.get(atom_type, atom_type)
                    target.write(line[:66].ljust(76) + element.rjust(2) + "\n")
            target.write("TER\nEND\n")
