load("../framework/AttackResults.sage")
load("../framework/utils.sage")

from pathlib import Path
import sys
import os

if len(sys.argv) < 2:
    printf("Specify path to pickle")
    printf("Usage: sage {sys.argv[0]} <path>")
    sys.exit(1)

path = sys.argv[1]
result = load_results(path)

q = 3329
F = GF(q)

# Properties
print(f"BKZ {result.bkz}")
print(f"Basis has {len(result.basis)} entries, each dimension {result.secret.dimensions()}")
print("Secret is")
print(result.secret)
print()

# Basis Vectors
print("Basis vectors are as follows")
for i, vec in enumerate(result.basis):
    vec_cen = vec.apply_map(recenter)
    print(f"Solution {i} (norm {vec_cen.norm} {vec_cen}")

print()
print("End of Results")
print()
