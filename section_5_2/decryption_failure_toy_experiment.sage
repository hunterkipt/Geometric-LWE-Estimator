
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
load("../framework/utils.sage")
load("../framework/proba_utils.sage")
load("../framework/LWE.sage")

# Set to True on first run
extract_hints = True

## Function Definitions ##
def get_most_effective_hint(dbdd, V, L):
    num = (dbdd.mu[0,:-1]*V.T - L).numpy()
    S = dbdd.S[:-1,:-1] * dbdd.dim()
    denom = np.sqrt((V*S*V.T).numpy().diagonal())
    objective = num / denom
    max_ind = np.argmax(objective, axis=None)
    # print(objective[0, max_ind])


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

# Set Frodo-80 parameters
n = 80
m = n 
q = 2**11
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
kem = FrodoKEM(variant="FrodoKEM-80-AES")
_, _, _, S, _ = kem.kem_sk_unpack(sk)
A, B = kem.kem_pk_unpack(pk)

# Derive E from A, B, S
A = matrix(A)
S = matrix(S)
B = matrix(B)
E = ((B - A*S) % q).apply_map(recenter)

# Extract first column of the Frodo80 secret and error

b = matrix(B.column(0))
s = matrix(S.column(0))
e_vec = matrix(E.column(0))


print(f"A:: nrows: {A.nrows()}, ncols: {A.ncols()}")
print(f"B:: nrows: {B.nrows()}, ncols: {B.ncols()}")

# Build Embedding Samples
lwe_instance = LWE(n, q, m, D_e, D_s, A=A, s=s, e_vec=e_vec, verbosity=0)

# Create Approximate hint Covariance matrix
_, var = average_variance(D_s)
d = n + m
ell = RR(sqrt(d * var))

covh = RR(var* ell**4 / (q/8)**2) * identity_matrix(d)

if extract_hints:
    V = None
    L_pos = []
    L_neg = []

    # ## Open ctext file again. Create Hints ##
    with open(ctfile, 'r') as ctexts:
        for index, line in enumerate(ctexts):
            if index < 2:
                continue
        
            # Create variables
            # print(f"Processing line {index}", end="\r")
            mu_enc, Sp = line.strip().split("\\\\")
            mu_enc = [int(mu) for mu in mu_enc.split(",")]
            Sp = [int(s) for s in Sp.split(",")]
            Ep = Sp[n*8:2*n*8]
            Epp = Sp[2*n*8:]
            Sp = Sp[:n*8]

            # Cast variables to matrix
            Sp = matrix(np.array(Sp).reshape(8, n) % q)
            Ep = matrix(np.array(Ep).reshape(8, n) % q)
            Epp = matrix(np.array(Epp).reshape(8, 8) % q)

            # In normal Frodo Operation Bp stored as Ep. Rederive Ep
            Ep = ((Ep - Sp*A) % q)

            # Recenter Sp, Ep, Epp as they are error matrices
            Sp = Sp.apply_map(recenter)
            Ep = Ep.apply_map(recenter)
            Epp = Epp.apply_map(recenter)

            # Extract first row of Sp, Ep, and first coordinate of Epp
            sp = matrix(Sp.row(0))
            ep = matrix(Ep.row(0))
            epp = Epp[0, 0]

            # create hint
            v = concatenate([sp, -ep])
            l_pos = (q/8 - epp)
            l_neg = (q/8 + epp)

            # stack hints
            if V is None:
                V = v

            else:
                V = V.stack(v)
        
            L_pos.append(l_pos)
            L_neg.append(l_neg)

    # print("")
    L_pos = matrix(L_pos)
    L_neg = matrix(L_neg)

    # Ensure all hints in the DB are decryption failures
    valid_pos_hints = (concatenate([e_vec, s])*V.T - L_pos).numpy().flatten() >= 0
    valid_neg_hints = (concatenate([e_vec, s])*(-V.T) - L_neg).numpy().flatten() >= 0

    V_pos = matrix(V.numpy()[valid_pos_hints])
    L_pos = vec(L_pos.numpy()[0, valid_pos_hints])

    V_neg = matrix((-V).numpy()[valid_neg_hints])
    L_neg = vec(L_neg.numpy()[0, valid_neg_hints])

    V = V_pos.stack(V_neg)
    L = L_pos.augment(L_neg)

    save(V, ctfile)
    save(L, ctfile + "_thresholds")

else:
    V = load(ctfile + ".sobj")
    L = load(ctfile + "_thresholds" + ".sobj")


# Select the first 20 hints to use as hint DB

V_20 = matrix(V.numpy()[0:20, :])
L_20 = vec(L.numpy()[0, 0:20])

dbdd_approx = lwe_instance.embed_into_DBDD()
dbdd_ineq = lwe_instance.embed_into_DBDD()

beta_estimate_approx, _ = dbdd_approx.estimate_attack(silent=True, probabilistic=True)
beta_estimate_ineq, _ = dbdd_ineq.estimate_attack(silent=True, probabilistic=True)
beta_approx, _ = dbdd_approx.attack()
beta_ineq, _ = dbdd_ineq.attack()
dbdd_approx.estimate_attack(silent=True)
dbdd_ineq.estimate_attack(silent=True)
print("Number of Hints, Type of Hints, Ellipsoid norm, Estimated Beta, Actual Beta")
print(f"{0}, Approximate Hints, {dbdd_approx.ellip_norm() / dbdd_approx.dim()}, {beta_estimate_approx}, {beta_approx}")
print(f"{0}, Inequality Hints, {dbdd_ineq.ellip_norm() / dbdd_ineq.dim()}, {beta_estimate_ineq}, {beta_ineq}")


for i in range(20):
    v = -V_20[i, :]
    l = -L_20[0, i]
    dbdd_approx.integrate_approx_hint_fulldim((ell**2 / (l))*(v), covh)
    # v_ebdd, l_ebdd = dbdd_ineq.convert_hint_e_to_c(v, l)
    v_ineq, l_ineq, max_ind = get_most_effective_hint(dbdd_ineq, -V_20, -L_20)
    
    if v_ineq is not None:
        dbdd_ineq.integrate_ineq_hint(v_ineq, l_ineq)
    

    beta_estimate_approx, _ = dbdd_approx.estimate_attack(silent=True, probabilistic=True)
    beta_estimate_ineq, _ = dbdd_ineq.estimate_attack(silent=True, probabilistic=True)
    beta_approx, _ = dbdd_approx.attack()
    beta_ineq, _ = dbdd_ineq.attack()
    dbdd_approx.estimate_attack(silent=True)
    dbdd_ineq.estimate_attack(silent=True)

    print(f"{i+1}, Approximate Hints, {dbdd_approx.ellip_norm() / dbdd_approx.dim()}, {beta_estimate_approx}, {beta_approx}")
    print(f"{i+1}, Inequality Hints, {dbdd_ineq.ellip_norm() / dbdd_ineq.dim()}, {beta_estimate_ineq}, {beta_ineq}")
        
        
