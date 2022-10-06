
# Sage reset. Clears active variables and resets execution.
reset()

## Imports ##
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = [5, 3]
mpl.rcParams['font.size'] = int(10)

import numpy as np
import bitstring
from frodokem import FrodoKEM
from time import process_time
from multi_map import multi_map
from random import randint, seed
from numpy.random import seed as np_seed
from numpy.random import default_rng

## Uncomment the following, if you want to use prediction classes ##
# load("../framework/DBDD_predict_diag.sage") ##
# load("../framework/DBDD_predict.sage") ##

## Sage loads ##
load("../framework/DBDD.sage")
load("../framework/EBDD.sage")
load("../framework/EBDD_non_homo.sage")
load("../framework/utils.sage")
load("../framework/proba_utils.sage")
load("../framework/instance_gen.sage")
load("../framework/LWE.sage")

## Function Definitions ##
def get_most_effective_hint(dbdd, V, L):
    num = (dbdd.mu[0,:-1]*V.T - L).numpy()
    denom = np.sqrt((V*dbdd.S[:-1,:-1]*V.T).numpy().diagonal())
    objective = num / denom
    max_ind = np.argmax(objective, axis=None)
    if objective[0, max_ind] < 0:
        return None, None

    return V[max_ind, :], L[0, max_ind], max_ind

def check_failure(secret, v, l):
    if scal(v*secret.T) < l:
        return True

    return False

def find_failures(dbdd, V, L):
    secret = dbdd.u[0,:-1]
    failures = []
    for i in range(V.nrows()):
        if check_failure(secret, V[i, :], L[0, i]):
            failures.append(i)

    print(f"num failures: {len(failures)}")
    print(failures)
    return failures

def count_unconstrained_queries(dbdd, V, L, num_failures):
    rng = default_rng()
    secret = dbdd.u[0,:-1]
    found_failures = 0
    queries = 0

    # Shuffle indices to fairly count queries
    indices = np.arange(V.nrows())
    rng.shuffle(indices)
    for i in indices:
        queries += 1
        if check_failure(secret, V[i, :], L[0, i]):
            found_failures += 1

        if found_failures == num_failures:
            break
    
    return queries

def count_constrained_queries(dbdd, V, L, limit):
    secret = dbdd.u[0,:-1]
    queries = 0

    # Build dataset of alpha values
    alphas = None
    S = dbdd.dim()*dbdd.S[:-1,:-1]
    mu = dbdd.mu[0,:-1]
    for j in range(0, V.nrows(), 5000):
        print(f"{j}", end="\r")
        if j+5000 >= V.nrows():
            v = V[j:, :]
            l = L[0, j:]
        
        else:
            v = V[j:j+5000, :]
            l = L[0,j:j+5000]

        num = (mu*v.T - l).numpy()
        denom = np.sqrt((v*S*v.T).numpy().diagonal())
        alphas = (num / denom).flatten() if alphas is None else np.concatenate((alphas, num / denom), axis=None)

    # Use alpha values as keys for dictionary
    print("Building alpha dict:")
    alpha_dict = dict()
    for i in range(V.nrows()):
        alpha_dict[alphas[i]] = i

    # Sort keys by decreasing alpha, use constraints to limit which cts to query
    used_keys = []
    for key in sorted(alpha_dict.keys(), reverse=True):
        if key < limit[0] or key > limit[1]:
            continue

        queries += 1
        print(f"query {queries}", end='\r')
        i = alpha_dict[key]
        used_keys.append(i)
        if check_failure(secret, V[i, :], L[0, i]):
            dbdd.integrate_ineq_hint(V[i, :], L[0, i])
            break
    
    print("")
    return queries, V.delete_rows(used_keys), L.delete_columns(used_keys)
        


## Set precision for the FPLLL library ##
FPLLL.set_precision(200)

# set_random_seed(0)
# np_seed(seed=0)
# seed(0)

# Set to 0 if you don't want output from the framework
verbosity = 1

# Set Frodo-640 parameters
n = 640
m = n 
q = 2**15

frodo_distribution = [9288, 8720, 7216, 5264, 3384, 1918,
                    958, 422, 164, 56, 17, 4, 1]
D_s = get_distribution_from_table(frodo_distribution, 2**16)
D_e = D_s

keyfile = "ctexts100k_key.txt"

## Load PK from ctexts file ##
with open(keyfile, 'r') as keys:
    for index, line in enumerate(keys):
        if index == 0:
            sk = line
        
        if index == 1:
            pk = line
            break

sk = sk.strip().split(", ")[1:]
pk = pk.strip().split(", ")[1:]

# Turn sk, pk into a bytes object from hex string array
sk = bytes([int(s, 16) for s in sk])
pk = bytes([int(p, 16) for p in pk])

# Create FrodoKem object and extract A, B from pk; S from sk
kem = FrodoKEM()
_, _, _, S, _ = kem.kem_sk_unpack(sk)
A, B = kem.kem_pk_unpack(pk)

# Derive E from A, B, S
A = matrix(A)
S = matrix(S)
B = matrix(B)
E = ((B - A*S) % q).apply_map(recenter)

b = matrix(B.column(0)) - vec([256] + [0]*(m-1))
s = matrix(S.column(0))
e_vec = matrix(E.column(0)) - vec([256] + [0]*(m-1))

# Set initial e[0] to 0
threshold_offset = e_vec[0, 0] #+ e_vec[0, 1] + e_vec[0, 2]

Ap = A[1:, :]
b = b[0, 1:]
e_vec = e_vec[0, 1:]
m -= 1

assert b == (s*Ap.T + e_vec) % q

print(f"A:: nrows: {Ap.nrows()}, ncols: {Ap.ncols()}")
print(f"B:: nrows: {B.nrows()}, ncols: {B.ncols()}")

# Build Embedding Samples
lwe_instance = LWE(n, q, m, D_e, D_s, A=Ap, s=s, e_vec=e_vec)
_, var = average_variance(D_s)
d = n + m
ell = RR(sqrt(d * var))


dbdd = lwe_instance.embed_into_DBDD()
#print(dbdd.volumes())
# ebdd = lwe_instance.embed_into_EBDD_nist()
# ebdd.mean_update(vec([128, 64, 64] + [0]*(n + m - 3)))
ellip_norm = 1280
dbdd2_means = None

V = None
L = []

print("Loading Ciphertext DB...")
V = load("ctexts100k.sobj")
L = load("thresholds100k.sobj")

# Search V for failing ciphertexts
failures = find_failures(dbdd, V, L)
num_failures = len(failures)

failing_cts = V[failures]
failing_thresholds = L[0, failures]

unconst_q = count_unconstrained_queries(dbdd, V, L, num_failures)

# Count constrained queries
const_q = 0
for i in range(num_failures):
    print(f"Finding failure {i}...")
    local_q, V, L = count_constrained_queries(dbdd, V, L, (0, .1075))
    print(f"done. took {local_q} queries.")
    const_q += local_q


print(f"Unconstrained queries: {unconst_q}, Constrained queries: {const_q}")

print("==========================")
print("Finished All Experiments!")
print("==========================")