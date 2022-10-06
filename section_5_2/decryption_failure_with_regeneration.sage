
# Sage reset. Clears active variables and resets execution.
reset()

## Imports ##
import sys
import numpy as np
from frodokem import FrodoKEM
import bitstring
from time import process_time
from random import randint, seed
from numpy.random import seed as np_seed
from scipy.stats import truncnorm

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
    S = dbdd.S[:-1,:-1] * dbdd.dim()
    denom = np.sqrt((V*S*V.T).numpy().diagonal())
    objective = num / denom
    max_ind = np.argmax(objective, axis=None)
    print(objective[0, max_ind])


    if objective[0, max_ind] < 0:
        return None, None, None

    if objective[0, max_ind] > 1:
        return None, None, max_ind

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
max_regen = 10

frodo_distribution = [9288, 8720, 7216, 5264, 3384, 1918,
                    958, 422, 164, 56, 17, 4, 1]
D_s = get_distribution_from_table(frodo_distribution, 2**16)
D_e = D_s

if len(sys.argv) < 2:
    raise ValueError("Not enough command line args. Please enter the ctfile location")

ctfile = sys.argv[1]

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

assert E[0,0] > 256

b = matrix(B.column(0)) - vec([256] + [0]*(m-1))
s = matrix(S.column(0))
e_vec = matrix(E.column(0)) - vec([256] + [0]*(m-1))

# Set initial e[0], e[1], e[2] to 0
threshold_offset = e_vec[0, 0] #+ e_vec[0, 1] + e_vec[0, 2]

Ap = A[1:, :]
b = b[0, 1:]
e_vec = e_vec[0, 1:]
m -= 1

assert b == (s*Ap.T + e_vec) % q

print(f"A:: nrows: {Ap.nrows()}, ncols: {Ap.ncols()}")
print(f"B:: nrows: {B.nrows()}, ncols: {B.ncols()}")

# Build Embedding Samples
lwe_instance = LWE(n, q, m, D_e, D_s, A=Ap, s=s, e_vec=e_vec, verbosity=0)

_, var = average_variance(D_s)
d = n + m
ell = RR(sqrt(d * var))


#print(dbdd.volumes())
# ebdd = lwe_instance.embed_into_EBDD_nist()
# ebdd.mean_update(vec([128, 64, 64] + [0]*(n + m - 3)))
ellip_norm = 1280
dbdd2_means = None

V = None
L = []

# ## Open ctext file again. Create Hints ##
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
            V = v

        else:
            V = V.stack(v)
      
        L.append(l)

print("")
L = matrix(L)
save(V, ctfile)
save(L, ctfile + "_thresholds")

V = load(ctfile + ".sobj")
L = load(ctfile + "_thresholds" + ".sobj")

# Ensure all hints in the DB are decryption failures
valid_hints = (concatenate([e_vec, s])*V.T - L).numpy().flatten() >= 0
V = matrix(V.numpy()[valid_hints])
L = vec(L.numpy()[0, valid_hints])

# Compute threshold for checking if <\mu, s> >= <s, s>
a = ((q/8 - (256 + threshold_offset)*12) / (RR(sqrt(m+n+1))*2.8*2.8))
lp = truncnorm.mean(a, np.inf)*2.8*2.8*RR(sqrt(n+m+1))


print(V.nrows(), L.ncols())

V_comb = None
L_comb = []
hints_used = []

for i in range(max_regen):
    try:
        v_comb = load(ctfile + f"v_comb_{i}.sobj")
        l_comb = load(ctfile + f"l_comb_{i}.sobj")
        hints_used = load(ctfile + "hints_used.sobj")
        if V_comb is None:
            V_comb = v_comb

        else:
            V_comb = V_comb.stack(v_comb)

        L_comb.append(l_comb)

    except:
        break

for j in range(i, max_regen):
    print(f"Regeneration: {j}")
    print("==========================")
    dbdd = lwe_instance.embed_into_DBDD()

    for k in range(j):
        print(dbdd.leak(V_comb[k,:]), L_comb[k])
        dbdd.integrate_ineq_hint(V_comb[k,:], L_comb[k])

    print("Unoptimized Regeneration Result")	
    print(f"Final DBDD Beta: {dbdd.beta}")
    ellip_norm = dbdd.ellip_norm()
    print(f"Final DBDD Ellipsoid Norm: {ellip_norm}")
    dbdd.S *= ellip_norm / dbdd.dim()
    dbdd.estimate_attack(silent=True)
    print(f"Final DBDD Beta (adjusted): {dbdd.beta}")
    dbdd.S *= dbdd.dim() / ellip_norm
    dbdd.estimate_attack(silent=True)
    print("==========================")

    num_hints = 0
    # Integrate decryption failure hints hints in optimal fashion
    while True:
        if np.mean(dbdd.mu[0,:-1]*V.T) >= (lp)*1.05:
            break

        v, l, max_ind = get_most_effective_hint(dbdd, -V, -L)
        if v is None:
            break
        
        hints_used.append(max_ind)
        # print(dbdd.leak(v), l)
        # v, l = ebdd.convert_hint(v, l)
        dbdd.integrate_ineq_hint(v, l)
        # print(np.mean(dbdd.mu[0,:-1]*V.T), lp)
        # print(dbdd.mu*dbdd.u.T)
        num_hints += 1
        if num_hints % 20 == 0:
            print(f"Hints integrated: {num_hints}")
    
    # Regeneration reached saturation, the combined hint is the mean of the ellipsoid
    v_comb = -dbdd.mu[0, :-1] 
    l_comb = -2.8*2.8*(m+n)
    print(v_comb*concatenate([e_vec, s]).T, l_comb)

    # Save combined hint for later use
    save(hints_used, ctfile + "hints_used")
    save(v_comb, ctfile + f"v_comb_{j}")
    save(l_comb, ctfile + f"l_comb_{j}")

    # Store combined hint for next regeneration
    if V_comb is None:
        V_comb = v_comb

    else:
        V_comb = V_comb.stack(v_comb)

    L_comb.append(l_comb)
    print(f"Done. Hints used: {len(set(hints_used))}, Beta = {dbdd.beta}, ellip norm = {dbdd.ellip_norm()}")
    print("==========================")


# All regenerations finished. Compute final optimized regeneration.
L_comb = matrix(L_comb)
dbdd = lwe_instance.embed_into_DBDD()
while True:
    v, l, max_ind = get_most_effective_hint(dbdd, V_comb, L_comb)
    if v is None and max_ind is None:
        break

    elif v is None and max_ind is not None:
        V_comb = V_comb.delete_rows([max_ind])
        L_comb = L_comb.delete_columns([max_ind])
        continue

    dbdd.integrate_ineq_hint(v, l)

print(f"Final DBDD Beta: {dbdd.beta}")
ellip_norm = dbdd.ellip_norm()
print(f"Final DBDD Ellipsoid Norm: {ellip_norm}")
dbdd.S *= ellip_norm / dbdd.dim()
dbdd.estimate_attack(silent=True)
print(f"Final DBDD Beta (adjusted): {dbdd.beta}")
dbdd.S *= dbdd.dim() / ellip_norm
print(f"Unique hints used: {len(set(hints_used))}")
print("==========================")
print("Finished All Experiments!")
print("==========================")
