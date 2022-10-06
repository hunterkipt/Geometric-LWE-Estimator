
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

## Set precision for the FPLLL library ##
FPLLL.set_precision(200)

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

ctfile = "ctexts100k.log"

## Load PK from ctexts file ##
with open(ctfile, 'r') as ctexts:
    for index, line in enumerate(ctexts):
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

## Open ctext file again. Create Hints ##
with open(ctfile, 'r') as ctexts:
    for index, line in enumerate(ctexts):
        if index < 2:
            continue
        
        # Create variables
        print(f"Processing line {index}", end="\r")
        mu_enc, Sp = line.strip().split("\\\\")
        mu_enc = [int(mu) for mu in mu_enc.split(",")]
        Sp = [int(s) for s in Sp.split(",")]
        Ep = Sp[640*8:2*640*8]
        Epp = Sp[2*640*8:]
        Sp = Sp[:640*8]

        # Cast variables to matrix
        Sp = matrix(np.array(Sp).reshape(8, 640) % q)
        Ep = matrix(np.array(Ep).reshape(8, 640) % q)
        Epp = matrix(np.array(Epp).reshape(8, 8) % q)

        # In normal Frodo Operation Bp stored as Ep. Rederive Ep
        Ep = ((Ep - Sp*A) % q)

        # Recenter Sp, Ep, Epp as they are error matrices
        Sp = Sp.apply_map(recenter)
        Ep = Ep.apply_map(recenter)
        Epp = Epp.apply_map(recenter)

        # Extract first row of Sp, Ep, and first coordinate of Epp
        sp = matrix(Sp.row(0))[0, 1:]
        ep = matrix(Ep.row(0))
        epp = Epp[0, 0]

        # create hint
        v = concatenate([sp, -ep])
        l = (q/8 - epp) - (256 + threshold_offset)*12

        # stack hints
        if V is None:
            V = -v

        else:
            V = V.stack(-v)
        
        L.append(-l)

L = matrix(L)

save(V, "ctexts100k")
save(L, "thresholds100k")

print("Loading Ciphertext DB...")
V = load("ctexts100k.sobj")
L = load("thresholds100k.sobj")

# Search V for failing ciphertexts
failures = find_failures(dbdd, V, L)
num_failures = len(failures)

failing_cts = V[failures]
failing_thresholds = L[0, failures]

# Integrate the failing ciphertexts into the instance
hists = dict()
for i in range(len(failures)):
    print(f"Failure: {i}")

    v, l, max_ind = get_most_effective_hint(dbdd, failing_cts, failing_thresholds)
    dbdd.integrate_ineq_hint(v, l)

    # Remove hint vector from dataset
    failing_cts = failing_cts.delete_rows([max_ind])
    failing_thresholds = failing_thresholds.delete_columns([max_ind])

    # Build dataset of alpha values
    alphas = None
    S = (n+m+1)*dbdd.S[:-1,:-1]
    mu = dbdd.mu[0,:-1]
    for j in range(0, V.nrows(), 5000):
        if j+5000 >= V.nrows():
            v = V[j:, :]
            l = L[0, j:]
        
        else:
            v = V[j:j+5000, :]
            l = L[0,j:j+5000]

        num = (mu*v.T - l).numpy()
        denom = np.sqrt((v*S*v.T).numpy().diagonal())
        alphas = num / denom if alphas is None else np.concatenate((alphas, num / denom), axis=None)

    # concatenate alphas from remaining failing cts
    num = (mu*failing_cts.T - failing_thresholds).numpy()
    denom = np.sqrt((failing_cts*S*failing_cts.T).numpy().diagonal())
    alphas = np.concatenate((alphas, num / denom), axis=None)

    # Create histogram from alpha values
    print(f"Building Histogram {i}")
    hists[i] = np.histogram(alphas, bins=50, range=(np.float64(-1), np.float64(1)))
    freq, bins = hists[i]
    print(freq)
    fig, ax = plt.subplots(dpi=300)
    ax.hist(bins[:-1], bins, weights=freq, rwidth=0.85, alpha=0.8)
    ax.set_xlim(-0.05, 0.25)
    ax.set_ylim(0, 100000)
    ax.set_xlabel("alpha")
    ax.set_ylabel("frequency")
    ax.set_title(f"{i+1} Hint(s)")
    ax.grid()
    fig.tight_layout()
    plt.savefig(f"../results/histogram{i}.png", format="png")


print("==========================")
print("Finished All Experiments!")
print("==========================")