from dataclasses import dataclass, field
from pickle import dump as p_dump, load as p_load
# from json import dump, load
from pathlib import Path
from typing import List
from sage.all import Matrix

@dataclass
class AttackResults:
    basis_vecs: field(default_factory=list)
    secret_vec: Matrix = None
    bkz: int = 0

def save_results(state: AttackResults, filepath: Path):
    with open(filepath, "wb") as f:
        p_dump(state, f)
    # temp_filepath = filepath.with_suffix(".tmp")

    # with open(temp_filepath, "wb") as f:
    #     p_dump(state, f)

    # temp_filepath.replace(filepath)

def load_results(filepath: Path) -> AttackResults:
    if not file_path.is_file():
        return AttackResults
    
    with open(filepath, "rb") as f:
        return p_load(f)
    
