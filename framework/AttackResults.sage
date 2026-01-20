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
    error_vec: Matrix = None
    # bkz: int = 0

def save_results(state: AttackResults, filepath: Path):
    # np.save()
    with open(filepath, "wb") as f:
        p_dump(state, f)


# (save_results)
# obj[0] = (error_vec | secret_vec | 0)
# obj[1] = basis[0]
# obj[2] = basis[1]
# ...
# np.save(obj, "file.npz")
# ... (load_results)
# e <= obj
# s <= obj
# basis <= obj

def load_results(filepath: Path) -> AttackResults:
    # np.load("...npz")
    if not file_path.is_file():
        return AttackResults
    
    with open(filepath, "rb") as f:
        return p_load(f)
    
def get_secret() -> Matrix:
    pass

def get_error() -> Matrix:
    pass

def retrieve_shortvectors() -> List[np.Matrix]:
    # find v s.t. nonzeros(v) <= 2 and ||v||_2 < val
    # [0 ... 0 x1 0 ... 0 x2 0 ... 0]
    pass