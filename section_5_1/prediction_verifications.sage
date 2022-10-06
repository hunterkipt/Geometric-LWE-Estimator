from multiprocessing import Pool
from map_drop import map_drop
from numpy.random import seed as np_seed
from numpy.linalg import eigvalsh
from numpy import allclose
import sys
load("../framework/LWE.sage")
load("../framework/LWR.sage")

Derr = build_centered_binomial_law(6)
Dcomb_hint = build_Gaussian_law(9, 20)
modulus = 11
ineq_offset = 2

try:
    N_tests = int(sys.argv[1])
    threads = int(sys.argv[2])
    T_hints = sys.argv[3].strip()
except:
    N_tests = 5
    threads = 1
    T_hints = "Perfect"


def v(i):
    return canonical_vec(d, i)


qvec_donttouch = 0


def randv():
    vv = v(randint(qvec_donttouch, d - 1))
    vv -= v(randint(qvec_donttouch, d - 1))
    vv += v(randint(qvec_donttouch, d - 1))
    vv -= v(randint(qvec_donttouch, d - 1))
    vv += v(randint(qvec_donttouch, d - 1))
    return vv


def one_experiment(id, aargs):
    (N_hints, T_hints) = aargs
    mu, variance = average_variance(Derr)
    mu_comb_hint, var_comb_hint = average_variance(D_s)
    set_random_seed(id)
    np_seed(seed=id)
    
    lwe_instance = LWE(n, q, m, D_e, D_s, verbosity=0)
    # lwr_instance = LWR(n, p, q, m, D_s, verbosity=1)
    dbdd = lwe_instance.embed_into_DBDD()
    ebdd = lwe_instance.embed_into_EBDD()
    # print(ebdd.ellip_norm(), ebdd.ellip_scale)
    alpha = 0
    for j in range(N_hints):
        vv = randv()
        # print(vv)
        l = dbdd.leak(vv)
        vv_ebdd, l_ebdd = ebdd.convert_hint_e_to_c(vv, l)
        norm = sqrt(scal(vv_ebdd * ebdd.S * ebdd.dim() * vv_ebdd.T))
        alpha += abs(scal(vv_ebdd * ebdd.mu.T) - l_ebdd) / norm

        if T_hints == "Perfect":
            dbdd.integrate_perfect_hint(vv, l)
            ebdd.integrate_perfect_hint(vv_ebdd, l_ebdd)
            

        if T_hints == "Approx":
            dbdd.integrate_approx_hint(vv, dbdd.leak(vv) +
                                       draw_from_distribution(Derr),
                                       variance)
            ebdd.integrate_approx_hint(vv_ebdd, ebdd.leak(vv_ebdd) +
                                         draw_from_distribution(Derr),
                                         variance)
        if T_hints == "Modular":
            dbdd.integrate_modular_hint(vv, dbdd.leak(vv) % modulus,
                                        modulus, smooth=True)
            ebdd.integrate_modular_hint(vv_ebdd, ebdd.leak(vv_ebdd) % modulus,
                                          modulus, smooth=True)

        if T_hints == "Inequality":
            # Ensure inequality always <
            sign = -1 if l > 0 else 1

            dbdd.integrate_ineq_hint(sign*vv, sign*l + ineq_offset)
            vv_ebdd, l_ebdd = ebdd.convert_hint(sign*vv, sign*l + ineq_offset)
            ebdd.integrate_ineq_hint(vv_ebdd, l_ebdd)

        if T_hints == "Combined":
            # Create ellipsoid that is guaranteed to contain the secret
            # Scale radius of ellipsoid to expected length of the gaussian error vec
            offset = vec([draw_from_distribution(D_s) for _ in range(n+m)])/RR(sqrt(N_hints))
            comb_hint_S = diagonal_matrix((n+m)*[var_comb_hint*(n+m)/N_hints])
 
            dbdd.integrate_combined_hint(dbdd.u[0, :-1] + offset, comb_hint_S)
            ebdd.integrate_combined_hint(ebdd.u + offset, comb_hint_S)
            break

    if T_hints == "Perfect":
        ebdd.apply_perfect_hints()

    dbdd.integrate_q_vectors(q)
    ebdd.integrate_q_vectors(q)
    
    alpha /= N_hints

    beta_pred_ebdd, _ = ebdd.estimate_attack(probabilistic=True, silent=True)
    beta_pred_dbdd, _ = dbdd.estimate_attack(probabilistic=True, silent=True)
    beta_dbdd, _ = dbdd.attack()
    beta_ebdd, _ = ebdd.attack()
    
    return (beta_pred_dbdd, beta_dbdd, beta_pred_ebdd, beta_ebdd, alpha)


dic = {" ": None}


def validation_prediction(N_tests, N_hints, T_hints):
    # Estimation
    import datetime
    ttt = datetime.datetime.now()
    res = map_drop(N_tests, threads, one_experiment, (N_hints, T_hints))
    # Print results
    for r in res:
        print("%d,\t %.3f,\t %.3f,\t %.3f,\t %.3f,\t %.3f" %
          (N_hints, r[0], r[1], r[2], r[3], r[4]))

    # Print Averages
    print("== Averages ==")
    beta_pred_dbdd = RR(sum([r[0] for r in res])) / N_tests
    beta_dbdd = RR(sum([r[1] for r in res])) / N_tests
    beta_pred_ebdd = RR(sum([r[2] for r in res])) / N_tests
    beta_ebdd = RR(sum([r[3] for r in res])) / N_tests
    alpha = RR(sum(r[4] for r in res)) / N_tests
    print("%d,\t %.3f,\t %.3f,\t %.3f,\t %.3f,\t %.3f\t" %
          (N_hints, beta_pred_dbdd, beta_dbdd, beta_pred_ebdd, beta_ebdd, alpha), end=" \t")
    print("Time:", datetime.datetime.now() - ttt)
    return beta_pred_dbdd


logging("Number of threads : %d" % threads, style="DATA")
logging("Number of Samples : %d" % N_tests, style="DATA")
logging("     Validation tests     ", style="HEADER")

n = 70
m = n
q = 3301
p = 211
sigma_se = 4.472135955
D_s = build_Gaussian_law(sigma_se, 50)
D_e = D_s
d = m + n

print(f"\n{T_hints}")

print("hints,\t pred_DBDD,\t DBDD,\t pred_EBDD,\t EBDD,\t alpha")
for h in range(1, 500):
    beta_pred = validation_prediction(N_tests, h, T_hints)  # Line 0
    if beta_pred < 3:
        break

# print("\n \n Modular")

# print("hints,\t real,\t pred_full, \t pred_light,")
# for h in range(2, 200, 2):
#     beta_pred = validation_prediction(N_tests, h, "Modular")  # Line 0
#     if beta_pred < 3:
#         break

# print("\n \n Approx")

# print("hints,\t real,\t pred_full, \t pred_light,")
# for h in range(4, 200, 4):
#     beta_pred = validation_prediction(N_tests, h, "Approx")  # Line 0
#     if beta_pred < 3:
#         break
