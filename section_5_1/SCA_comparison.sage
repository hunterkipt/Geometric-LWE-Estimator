load("../framework/LWE.sage")
load("../sage_notebook/optimize_r.sage")

from scipy.io import loadmat
import random
from copy import deepcopy
from multiprocessing import Pool
from map_drop import map_drop
from numpy.random import seed as np_seed

"""  Example
Uncomment the following to get an example
of the detailed computation (without redundancy)
"""
#demo = True
#logging("--- Demonstration mode (no redundancy of the attacks) ---")

prob_limit = 0.77
verbose=False

try:
    nb_tests_per_params = int(sys.argv[1])
    threads = int(sys.argv[2])
except:
    nb_tests_per_params = 1 #50
    threads = 1

def simu_measured(secret, measures):
    """
    This function simulates the information gained by
    Bos et al attack. For a given secret, the simulation
    is outputting a score table at random from the data of Bos et al.
    :secret: an integer being the secret value
    :measures: the Bos et al. data
    :measurement: a score table
    """
    if secret in measures.keys():
        measurement = random.choice(measures[secret])
    else: 
        measurement = None
    return measurement

def measurement_to_aposteriori(measurement, frodo_distribution):
    """
    This fonction transforms a score table into an aposteriori distribution.
    According to the matlab code of Bos et al. attack,
    the score table is proportional
    to the logarithm of the aposteriori probability. We thus apply the exponential
    and renormalize.
    :measurement: a score table
    :frodo_distribution: the frodo distribution being used
    :apost_dist: a dictionnary of the aposteriori distribution
    """
    if measurement is not None:
        s = sum([exp(meas) for meas in measurement])
        measurement = [exp(meas)/s for meas in measurement]
        apost_dist = {}
        for key_guess in range(-len(frodo_distribution) + 1, len(frodo_distribution)):
            apost_dist[key_guess] = measurement[key_guess + len(frodo_distribution) - 1]
    else:
        apost_dist = None
    return apost_dist

def gen_aposteriori_ellip(dbdd, measured, frodo_distribution):
    """
    This function generates the aposteriori ellipsoid from the measured SCA data from
    the Bos et al. attack. It also produces the dictionary of guesses which can be made.
    :dbdd: the dbdd instance
    :measured: table representing the (simulated) information
    given by Bos et al attack
    :frodo_distribution: the frodo distribution being used
    :mu: the mean of the dbdd instance
    :S: the variance of the dbdd instance
    :guess_dict: the dictionary of coordinates values to guess and their probabilities
    :sorted_guesses: the sorted guess dictionary
    """
    n = dbdd.embedded_instance.n
    m = dbdd.embedded_instance.m
    Id = identity_matrix(n + m)

    guess_dict = {}
    mu = zero_matrix(1, n + m)
    S = zero_matrix(1, n + m)
    for i in range(n):
        apost_dist = measurement_to_aposteriori(measured[i], frodo_distribution)            
        # Use aposteriori distribution to find highest prob coordinate
        # Save in guess dict for guessing steps.
        best_guess = max(apost_dist, key=apost_dist.get)
        guess_dict[m+i] = (best_guess, apost_dist[best_guess])

        # find mean and variance of aposteriori dist, and incorporate into ellip 
        av, var = average_variance(apost_dist)
        av = float(av)
        var = float(var)

        if var == 0:
            guess_dict[m+i] = (av, 1)

        v = vec(Id[m+i])
        mu += av*v
        S += var*v
        
    # guess_dict.items -> (index, (guess_value, guess_prob))
    sorted_guesses = sorted(guess_dict.items(), key=lambda kv: kv[1][1], reverse=True)
    sorted_guesses = [sorted_guesses[i][0] for i in range(len(sorted_guesses)) if sorted_guesses[i][1][1] != 1.]

    return mu, S, guess_dict, sorted_guesses
    
def extend_ellipsoid(dbdd, mu, S):
    """
    Extend the ellipsoid from n-dimensions to the full n+m+1 dimensional ellipsoid
    :dbdd: the dbdd instance being used (for the dimension information)
    :mu: The 1 by n dimensional matrix for the ellipsoid mean
    :S: the n by n dimensional diagonal matrix for the ellipsoid variance
    """
    n = dbdd.embedded_instance.n
    m = dbdd.embedded_instance.m

    mu_e, s_e = average_variance(dbdd.embedded_instance.D_e)
    new_S1 = block4(diagonal_matrix([0] * m), zero_matrix(ZZ, m, n), zero_matrix(ZZ, n, m), S)
    new_S = block4(new_S1, zero_matrix(ZZ, n + m, 1), zero_matrix(ZZ, 1, n + m), zero_matrix(ZZ, 1, 1))
    rest_S = diagonal_matrix(m * [s_e] + n * [0] + [0])
        
    hybrid_S = (new_S/n + rest_S)
    diagonal_S = np.array([RR(hybrid_S[i, i]) for i in range(hybrid_S.nrows())])
 
    new_mu = concatenate([[mu_e]*m, mu, [1]])

    return new_mu, diagonal_S

def original_SCA_attack(dbdd_SCA, mu_SCA, S_SCA):
    """
    Performs the original dbdd attack using approximate hint 
    :dbdd_SCA: the dbdd instance where hints will be integrated
    :mu_SCA: the mean of the side channel distribution
    :S_SCA: the variance of the side channel distribution
    """
    n = dbdd_SCA.embedded_instance.n
    m = dbdd_SCA.embedded_instance.m
    Id = identity_matrix(n + m)

    if verbose:
        logging("     Original attack estimation     ", style="HEADER")

    for i in range(n):
        av = mu_SCA[0,i+m]
        var = S_SCA[0,i+m]
        v = vec(Id[m+i])
        if var > 0:
            dbdd_SCA.integrate_approx_hint(v, av, var, aposteriori=True, estimate=False)
        if var == 0:
            dbdd_SCA.integrate_perfect_hint(v, av, estimate=False)

    return dbdd_SCA

def intersect_SCA_ellipsoid(dbdd_int, mu_SCA, S_SCA, guess_dict, sorted_guesses, optimized=False, approximation=False):
    """
    Intersects the posterior (SCA) ellipsoid with the prior (LWE) ellipsoid. Can use the optimized method
    for intersection, or the alternative approach
    :dbdd_int: the dbdd instance for the intersected ellipoid
    :mu_SCA: the mean of the SCA ellipsoid
    :S_SCA: the 1 by n+m+1 dimensional matrix representing the diagonal of the SCA ellipsoid
    :guess_dict: the dictionary of coordinates values to guess and their probabilities
    :sorted_guesses: the sorted guess dictionary
    :optimized: whether or not to use the optimized method for ellipsoid intersection
    """
    intersect_indices = []
    mu_s, s_s = average_variance(dbdd_int.embedded_instance.D_s)
    n = dbdd_int.embedded_instance.n
    m = dbdd_int.embedded_instance.m
    intersect_dim = n

    # Removes the coordinates which will be guessed
    Id = identity_matrix(n + m)
    proba_success = 1.
    guesses = 0
    coordinates_guessed = vec(zeros(n+m))

    if verbose:
        if optimized:
            if approximation:
                logging("   Approximation attack estimation   ", style="HEADER")
            else:
                logging("     Optimized attack estimation     ", style="HEADER")
        else:
            logging("     Heuristic attack estimation     ", style="HEADER")

    for i in sorted_guesses:
        if proba_success >= prob_limit:
            v = vec(Id[i])
            guesses += 1
            guess_prob = guess_dict[i][1]
            proba_success *= guess_prob 
            coordinates_guessed = coordinates_guessed + v

    zero_coords = 0
    for i in range(n):
        if coordinates_guessed[0][i+n] != 1 and S_SCA[0,i] == 0:
            print("0 coordinate ", i, S_SCA[0,i], mu_SCA[0,i])
            zero_coords += 1

    intersect_dim = n - guesses - zero_coords
    full_mu_SCA = mu_SCA
    full_S_SCA = S_SCA
    mu_SCA = matrix(mu_SCA[0,0:intersect_dim])
    S_SCA = matrix(S_SCA[0,0:intersect_dim])

    j = 0
    for i in range(n):
        if coordinates_guessed[0][i+n] != 1 and full_S_SCA[0,i] != 0:
            mu_SCA[0,j] = full_mu_SCA[0,i]
            S_SCA[0,j] = full_S_SCA[0,i]
            j += 1

    # Find the indices to intersect on
    if optimized:
        betas = [-1]*intersect_dim
        cs = mu_SCA[0]
        for i in range(intersect_dim):
            if s_s > S_SCA[0,i] and S_SCA[0,i] != 0:
                betas[i] = s_s / S_SCA[0,i]
        intersect_indices = find_optimal_indices(betas, cs, s_s, approximation)
    else:
        for i in range(intersect_dim):
            if mu_SCA[0,i]*mu_SCA[0,i] > s_s - S_SCA[0,i]:
                intersect_indices += [i]

    if verbose:
        logging("Number of elements in intersect_indices: %4d"% len(intersect_indices))

    #Create the matrices for the intersection
    S_SCA = diagonal_matrix(S_SCA[0])*(intersect_dim)
    mu_LWE = matrix(mu_SCA)
    S_LWE = matrix(S_SCA)

    for i in intersect_indices:
        S_LWE[i,i] = s_s*intersect_dim
        mu_LWE[0,i] = mu_s

    #Calculate the new ellispoid
    try:
        new_mu_sub, new_S_sub = ellipsoid_intersection(mu_LWE, S_LWE, mu_SCA, S_SCA, tolerance=1.48e-08) #<=1
    except ValueError:
        new_mu_sub = mu_SCA
        new_S_sub = S_SCA
        logging("Ellipsoid Intersection FAILED. Resorting to posterior ellipsoid.")
        if optimized:
            if approximation:
                logging("Approximate Intersection Used")
            else:
                logging("Optimized Attack Used")        
        else:
            logging("Hueristic Attack Used")
        logging("Indicies %4d"% len(intersect_indicies))

    det_SCA = logdet(S_SCA)
    det_int = logdet(new_S_sub)
    if verbose:
        logging("ln det(SCA matrix): %3.2f"% det_SCA)
        logging("ln det(LWE matrix): %3.2f"% logdet(S_LWE))
        logging("ln det(intersected matrix): %3.2f"% det_int)
    det_ratio = det_int - det_SCA

    # Add back in the coordinates which will be guessed
    mu_SCA = full_mu_SCA
    S_SCA = diagonal_matrix(full_S_SCA[0])*(n)
    j = 0
    for i in range(n):
        if coordinates_guessed[0][i+n] != 1 and full_S_SCA[0,i] != 0:
            mu_SCA[0,i] = new_mu_sub[0,j]
            S_SCA[i,i] = new_S_sub[j,j]
            j += 1
    new_mu_sub = mu_SCA
    new_S_sub = S_SCA

    # Increase dimension back up to n+m+1 dimensions
    mu_SCA, S_SCA = extend_ellipsoid(dbdd_int, new_mu_sub, new_S_sub)
    dbdd_int.mu = mu_SCA
    dbdd_int.S = S_SCA
    
    ellipsoid_norm = 0
    for i in range(n+m+1):
        ellipsoid_norm += (dbdd_int.u[0,i] - mu_SCA[0,i]) * S_SCA[i] * (dbdd_int.u[0,i] - mu_SCA[0,i])
    if ellipsoid_norm > n+m+1:
        logging("Secret not contained in ellipsoid %3.2f"% ellipsoid_norm)

    #integrate perfect hints where there are zero coordinates
    for i in range(n):
        if coordinates_guessed[0][i+n] != 1 and S_SCA[m+i] == 0:
            dbdd_int.integrate_perfect_hint(vec(Id[m+i]), mu_SCA[0,i+m], estimate=False)
    
    return dbdd_int, det_ratio

def integrate_hints_guesses(dbdd_to_guess, guess_dict, sorted_guesses):
    """
    Integrate guesses into the dbdd instance, and then q-vectors
    :dbdd_to_guess: the dbdd instance to integrate guesses with
    :guess_dict: the dictionary of coordinates values to guess and their probabilities
    :sorted_guesses: the sorted guess dictionary
    :beta: the bikz after integrating q-vectors but without guesses
    :proba_success: the probability that all of the integrated guesses are correct
    :guesses: the number of guesses made
    """
    n = dbdd_to_guess.embedded_instance.n
    m = dbdd_to_guess.embedded_instance.m
    Id = identity_matrix(n + m)    

    # Calculate the bikz without the guesses
    duplicate_dbbd_to_guess = deepcopy(dbdd_to_guess)
    duplicate_dbbd_to_guess.integrate_q_vectors(q, min_dim = n + 1, indices=range(m))
    (beta, _) = duplicate_dbbd_to_guess.estimate_attack()


    # Test to make sure that the guesses are accurate for the experiment
    secret = dbdd_to_guess.u
    proba_success = 1.
    guesses = 0
    j = 0
    coordinates_guessed = vec(zeros(n+m))
    wrong_coordinates = 0   

    for i in sorted_guesses:
        if proba_success >= prob_limit:
            v = vec(Id[i])
            guess_val = guess_dict[i][0]
            guess_prob = guess_dict[i][1]
            proba_success *= guess_prob  
            if guess_val != secret[0][i-m]:
                #print("Incorrect secret", i, guess_val, secret[0][i-m])
                wrong_coordinates += 1
       
    logging("Wrong guesses: %4d" % wrong_coordinates)

    # Integrate perfect hints on coordinates which can be guessed
    proba_success = 1.
    guesses = 0

    for i in sorted_guesses:
        j += 1
        if proba_success >= prob_limit:
            v = vec(Id[i])
            guess_val = guess_dict[i][0]
            guess_prob = guess_dict[i][1]
            if dbdd_to_guess.integrate_perfect_hint(v, guess_val, force = False):
                guesses += 1
                proba_success *= guess_prob
                coordinates_guessed = coordinates_guessed + v
        
    # Calculate the determinant of the matrix after guesses
    S_matrix = dbdd_to_guess.S[n:n+m]
    S_after_guesses = diagonal_matrix(S_matrix[0:n-guesses])
    j = 0
    for i in range(n):
        if coordinates_guessed[0][i+n] != 1:
            S_after_guesses[j,j] = S_matrix[i]
            j += 1
    
    if verbose:
        logging("ln det(matrix after guesses): %3.2f"% logdet(S_after_guesses))

    # Check the bikz before q-vectors, then integrate q-vectors
    beta_after_guesses, _ = dbdd_to_guess.estimate_attack()
    if verbose:
        logging("Beta after guesses before hints: %3.2f"% beta_after_guesses)
    dbdd_to_guess.integrate_q_vectors(q, min_dim = n + 1, indices=range(m))
    
    return dbdd_to_guess, beta, proba_success, guesses


def estimate_SCA(dbdd, measured, frodo_distribution):
    """ 
    This function evaluates the security loss after Bos et al attack.
    Comuptes the estimated bikz after the original dbdd attack, and the ellipsoid
    attack using both optimized and non-optimized methods for intersection
    :dbdd: instance of the class DBDD
    :measured: table representing the (simulated) information
    given by Bos et al attack
    :frodo_distribution: the frodo distribution being used
    """

    mu_SCA, S_SCA, guess_dict, sorted_guesses = gen_aposteriori_ellip(dbdd, measured, frodo_distribution)

    # integrate aposteriori ellipsoid into dbdd instance
    dbdd_SCA = dbdd.embedded_instance.embed_into_DBDD_predict_diag()
    #dbdd_SCA_Ellip = dbdd.embedded_instance.embed_into_DBDD_predict_diag()
    dbdd_int = dbdd.embedded_instance.embed_into_DBDD_predict_diag() 
    dbdd_opt = dbdd.embedded_instance.embed_into_DBDD_predict_diag()
    dbdd_app = dbdd.embedded_instance.embed_into_DBDD_predict_diag()


    # dbdd_SCA_Ellip.mu, dbdd_SCA_Ellip.S = extend_ellipsoid(dbdd_SCA, mu_SCA[0,n:n+m], diagonal_matrix(S_SCA[0,n:n+m][0])*(n))
    # dbdd_SCA_Ellip.estimate_attack(silent=True)
    # dbdd_SCA_Ellip, beta_SCA_Ellip, proba_success_SCA_Ellip, guess_num_SCA_Ellip = integrate_hints_guesses(dbdd_SCA_Ellip, guess_dict, sorted_guesses)

    dbdd_SCA = original_SCA_attack(dbdd_SCA, mu_SCA, S_SCA)
    dbdd_SCA.estimate_attack(silent=True)
    dbdd_SCA, beta_SCA, proba_success_SCA, guess_num_SCA = integrate_hints_guesses(dbdd_SCA, guess_dict, sorted_guesses)

    dbdd_int, int_ratio = intersect_SCA_ellipsoid(dbdd_int, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], guess_dict, sorted_guesses, False, False)
    dbdd_int.estimate_attack(silent=True)
    dbdd_int, beta_int, proba_success_int, guess_num_int = integrate_hints_guesses(dbdd_int, guess_dict, sorted_guesses)
   
    dbdd_opt, opt_ratio = intersect_SCA_ellipsoid(dbdd_opt, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], guess_dict, sorted_guesses, True, False)
    dbdd_opt.estimate_attack(silent=True)
    dbdd_opt, beta_opt, proba_success_opt, guess_num_opt = integrate_hints_guesses(dbdd_opt, guess_dict, sorted_guesses)

    dbdd_app, app_ratio = intersect_SCA_ellipsoid(dbdd_app, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], guess_dict, sorted_guesses, True, True)
    dbdd_app.estimate_attack(silent=True)
    dbdd_app, beta_app, proba_success_app, guess_num_app = integrate_hints_guesses(dbdd_app, guess_dict, sorted_guesses)

    return (beta_SCA, dbdd_SCA.beta, guess_num_SCA, proba_success_SCA), (beta_int, dbdd_int.beta, guess_num_int, proba_success_int, int_ratio), (beta_opt, dbdd_opt.beta, guess_num_opt, proba_success_opt, opt_ratio), (beta_app, dbdd_app.beta, guess_num_app, proba_success_app, app_ratio)

def one_experiment(id, aargs):
    """ 
    Runs one experiment of the SCA attack for each method
    :id: the id of this instance
    :aargs: the parameters for the experiment
    """
    set_random_seed(id) #FIX RANDOMIZATION

    if verbose:
        logging("Instance %4d" % id)
    n, m, q, D_s, frodo_distribution, measures = aargs

    lwe_instance = LWE(n, q, m, D_s, D_s, verbosity=0)
    dbdd = lwe_instance.embed_into_DBDD_predict_diag()

    measured = [simu_measured(dbdd.u[0, i], measures) for i in range(n)]
    results_SCA, results_int, results_opt, results_app = estimate_SCA(dbdd, measured, frodo_distribution)
    
    if verbose:
        logging("DBDD Attack:                 %3.2f bikz %3.2f bikz %4d %0.2f" % (results_SCA), style="HEADER")
        logging("Intersection Attack:         %3.2f bikz %3.2f bikz %4d %0.2f %1.4f ratio" % (results_int), style="HEADER")
        logging("Optimal Intersection Attack: %3.2f bikz %3.2f bikz %4d %0.2f %1.4f ratio" % (results_opt), style="HEADER")
        logging("Approx Intersection Attack:  %3.2f bikz %3.2f bikz %4d %0.2f %1.4f ratio" % (results_app), style="HEADER")

    return results_SCA, results_int, results_opt, results_app

    
def run_experiment():
    """
    Runs the experiment with all parameter sets and with nb_tests_per_params instances for each set of parameters
    """
    # Estimation
    import datetime
    ttt = datetime.datetime.now()

    global n, m, q

    for params in ['CCS2']:#, 'CCS2', 'CCS3']:
        logging("Set of parameters: " + params)

        if params == 'NIST3':
            # NIST1 FRODOKEM-640
            n = 640
            m = 640
            q = 2**15
            frodo_distribution = [9456, 8857, 7280, 5249, 3321,
                                1844, 898, 384, 144, 47, 13, 3]
            D_s = get_distribution_from_table(frodo_distribution, 2 ** 16)
            # We used the following seeds for generating Bos et al. data
            # These seeds were generated with the matlab code genValues.m
            sca_seeds = [42, 72, 163, 175, 301, 320, 335, 406, 430, 445]
            param = 4

        elif params == 'NIST2':
            # NIST2 FRODOKEM-976
            n = 976
            m = 976
            q = 65536
            frodo_distribution = [11278, 10277, 7774, 4882, 2545, 1101,
                                396, 118, 29, 6, 1]
            D_s = get_distribution_from_table(frodo_distribution, 2 ** 16)
            sca_seeds = [74, 324, 337, 425, 543, 1595, 1707, 2026, 2075, 2707]
            param = 5

        elif params == 'CCS1':
            n = 352
            m = 352
            q = 2 ** 11
            frodo_distribution = [22528, 15616, 5120, 768]
            D_s = get_distribution_from_table(frodo_distribution, 2 ** 16)
            sca_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            param = 0

        elif params == 'CCS2':
            n = 592
            m = 592
            q = 2 ** 12
            frodo_distribution = [25120, 15840, 3968, 384, 16]
            D_s = get_distribution_from_table(frodo_distribution, 2 ** 16)
            sca_seeds = [2, 3, 4, 5, 6, 7, 8, 9, 14, 16]
            param = 1


        elif params == 'CCS3':
            n = 752
            m = 752
            q = 2 ** 15
            frodo_distribution = [19296, 14704, 6496, 1664, 240, 16]
            D_s = get_distribution_from_table(frodo_distribution, 2 ** 16)
            sca_seeds = [2, 4, 5, 7, 9, 10, 12, 13, 14, 15]
            param = 2

        elif params == 'CCS4':
            n = 864
            m = 864
            q = 2 ** 15
            frodo_distribution = [19304, 14700, 6490, 1659, 245, 21, 1]
            D_s = get_distribution_from_table(frodo_distribution, 2 ** 16)
            sca_seeds = [256, 393, 509, 630, 637, 652, 665, 1202, 1264, 1387]
            param = 3


        """  Original Security   """
        lwe_instance = LWE(n, q, m, D_s, D_s, verbosity=0)
        dbdd = lwe_instance.embed_into_DBDD_predict_diag()

        dbdd.integrate_q_vectors(q, indices=range(n, n + m))
        (beta, _) = dbdd.estimate_attack()
        logging("Attack without hints:  %3.2f bikz" % beta, style="HEADER")

        """  Refined Side channel attack  """

        # Reading the score tables from Bos et al. attack
        scores = []
        correct = []
        for seed in sca_seeds:
            for i in range(1, 9):
                data = loadmat('../sage_notebook/Scores_tables_SCA/' + params +
                            '/apostandcorrect' + str(param + 1) +
                            '_seed' + str(seed) +
                            'nbar' + str(i) + '.mat')
                scores += list(data['apostdist'])
                correct += list(data['correct'])
        measures = {}
        # the score tables are stored according to the secret coefficient. We use them
        # for generating the measurement.
        for key_guess in range(-len(frodo_distribution)+1, len(frodo_distribution)):
            measures[key_guess] = [scores[ind] for ind in
                    [i for i, d in enumerate(correct) if
                    recenter(d) == key_guess]]


        parameters = (n, m, q, D_s, frodo_distribution, measures)

        res = map_drop(nb_tests_per_params, threads, one_experiment, (parameters))

        for r in res:
            logging("DBDD Attack:                 %3.2f bikz %3.2f bikz %4d %0.2f bikz" % (r[0]), style="HEADER")
            logging("Intersection Attack:         %3.2f bikz %3.2f bikz %4d %0.2f bikz %1.4f ratio" % (r[1]), style="HEADER")
            logging("Optimal Intersection Attack: %3.2f bikz %3.2f bikz %4d %0.2f bikz %1.4f ratio" % (r[2]), style="HEADER")
            logging("Approx Intersection Attack:  %3.2f bikz %3.2f bikz %4d %0.2f bikz %1.4f ratio" % (r[3]), style="HEADER")

        logging("FINAL RESULTS for parameters: " + params, style="HEADER")
        results_SCA = [sum([r[0][i] for r in res]) / nb_tests_per_params for i in range(4)]
        results_int = [sum([r[1][i] for r in res]) / nb_tests_per_params for i in range(5)]
        results_opt = [sum([r[2][i] for r in res]) / nb_tests_per_params for i in range(5)]
        results_app = [sum([r[3][i] for r in res]) / nb_tests_per_params for i in range(5)]

        logging("DBDD Attack with hints:                             %3.2f bikz" % results_SCA[0], style="HEADER")
        logging("DBDD Attack with hints and guesses:                 %3.2f bikz" % results_SCA[1], style="HEADER")
        logging("DBDD Attack Number of guesses:                     %4d" % results_SCA[2], style="HEADER")

        logging("Intersection Attack with hints:                     %3.2f bikz" % results_int[0], style="HEADER")
        logging("Intersection Attack with hints and guesses:         %3.2f bikz"% results_int[1], style="HEADER")
        logging("Intersection Attack Number of guesses:             %4d" % results_int[2], style="HEADER")
        logging("Intersection Attack determinant ratio:             %1.4f" % results_int[4], style="HEADER")
            
        logging("Optimal Intersection Attack with hints:             %3.2f bikz" % results_opt[0], style="HEADER")
        logging("Optimal Intersection Attack with hints and guesses: %3.2f bikz"% results_opt[1], style="HEADER")
        logging("Optimal Intersection Attack Number of guesses:     %4d" % results_opt[2], style="HEADER")
        logging("Optimal Intersection Attack determinant ratio:     %1.4f" % results_opt[4], style="HEADER")

        logging("Approx Intersection Attack with hints:              %3.2f bikz" % results_app[0], style="HEADER")
        logging("Approx Intersection Attack with hints and guesses:  %3.2f bikz"% results_app[1], style="HEADER")
        logging("Approx Intersection Attack Number of guesses:      %4d" % results_app[2], style="HEADER")
        logging("Approx Intersection Attack determinant ratio:      %1.4f" % results_app[4], style="HEADER")

        logging("Success probability                                 %3.2f" %results_int[3], style="HEADER")
    
    print("Time:", datetime.datetime.now() - ttt)


    return 


run_experiment()
