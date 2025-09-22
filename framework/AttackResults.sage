from dataclasses import dataclass, field
from pickle import dump, load
# from json import dump, load
from pathlib import Path
from typing import List
from sage.matrix.matrix import Matrix

@dataclass
class AttackResults:
    bkz: int = 0
    secret_vec: Matirx = None
    basis_vecs: field(default_factory=list)

def save_results(state: AttackResults, filepath: Path):
    temp_filepath = filepath.with_suffix(".tmp")

    with open(temp_filepath, "wb") as f:
        dump(state, f)

    temp_filepath.replace(filepath)

def load_results(filepath: Path) -> AttackResults:
    if not file_path.is_file():
        return AttackResults
    
    with open(filepath, "rb") as f:
        return load(f)
    
