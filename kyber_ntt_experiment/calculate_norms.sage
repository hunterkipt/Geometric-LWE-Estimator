
import numpy as np
import os.path
from shutil import rmtree
from pathlib import Path
import json

def compute_norms(vecs):
    # filter
    norms = []
    for vec in vecs:
        if (vec[0][-1] == 0 and sum(map(lambda x: 1 if x != 0 else 0, vec[0])) <= 4):
            continue
        norms.append(float(vec[0].norm()))

    return norms

def mkdir(path: str, clear=True) -> Path:
    p = Path(path)
    if clear and p.exists():
        rmtree(p)
    p.mkdir(parents=True, exist_ok=not clear)
    return p


if __name__ == "__main__":
    mkdir("out")
    for g in range(39, 65):
        filename = f"results_{g}.json"
        if not os.path.exists(filename):
            continue
        try:
            with open(filename, "r") as fp:
                data = json.load(fp)
            if data is None:    # skips files that do not exist
                continue
            for i, exp in enumerate(data):
                if exp.get("s-norms") is not None:  # skips already computed norms
                    continue
                seed = exp["seed"]
                vecs = load(f"vecs-{g}/basis_{seed}.sobj")
                secret, bases = vecs[0], vecs[1:]
                norms = compute_norms(bases)
                data[i]["s-norm"] = float(secret.norm())
                data[i]["b-norms"] = norms
            with open(f"out/{filename}", "w") as fp:
                json.dump(data, fp, indent=4)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"failed on file {filename}")


