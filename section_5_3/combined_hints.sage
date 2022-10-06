load("../framework/LWE.sage")

from scipy.io import loadmat
import random
from copy import deepcopy
from multiprocessing import Pool
from map_drop import map_drop
from numpy.random import seed as np_seed
from numpy.random import default_rng

verbose=True
prob_limit = 0.77
guesses_before_intersect = False
rng = default_rng()

try:
    nb_tests_per_params = int(sys.argv[1])
    threads = int(sys.argv[2])
except:
    nb_tests_per_params = 1 #50
    threads = 1

try:
    bosetal_sigma = sys.argv[3]
except:
    bosetal_sigma = "0.0045"
print("Bos et al Sigma = ", bosetal_sigma)

def simu_measured(secret, measures):
    """
    This function simulates the information gained by
    Bos et al attack. For a given secret, the simulation
    is outputting a score table at random from the data of Bos et al.
    :secret: an integer being the secret value
    :measures: the Bos et al. data
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
            guess_dict[m+i] = (best_guess, 1)

        v = vec(Id[m+i])
        mu += av*v
        S += var*v
        
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
    s_e = 0.25

    # Create n+m+1 dimensional diagonalmatrix, with zeros in the first m diagonal coordinates,
    # the S matrix in the next n diagonal coordinates, then a 1 in the n+m+1 diagonal coordinate
    new_S1 = block4(diagonal_matrix([0] * m), zero_matrix(ZZ, m, n), zero_matrix(ZZ, n, m), S)
    new_S = block4(new_S1, zero_matrix(ZZ, n + m, 1), zero_matrix(ZZ, 1, n + m), zero_matrix(ZZ, 1, 1))
    
    # Create non-rank scaled ellipsoid with s_e in the first n diagonal coordinates, zeros elsewhere
    rest_S = diagonal_matrix(m * [s_e] + n * [0] + [0])
        
    # Combine the two matrices, then flatten the result
    hybrid_S = (new_S + rest_S)
    diagonal_S = np.array([RR(hybrid_S[i, i]) for i in range(hybrid_S.nrows())])
 
    # Extend ellipsoid mean to n + m + 1 dimensions
    new_mu = concatenate([[mu_e]*m, mu, [1]])

    return new_mu, diagonal_S

def original_SCA_attack(dbdd_SCA, mu_SCA, S_SCA, sorted_guesses, guess_dict):
    """
    Performs the original dbdd attack using approximate hints 
    :dbdd_SCA: the dbdd instance where hints will be integrated
    :mu_SCA: the mean of the side channel distribution
    :S_SCA: the variance of the side channel distribution
    :sorted_guesses: the sorted guess dictionary
    :guess_dict: the dictionary of coordinates values to guess and their probabilities
    """
    n = dbdd_SCA.embedded_instance.n
    m = dbdd_SCA.embedded_instance.m
    Id = identity_matrix(m+n)

    if verbose:
        logging("     Original attack estimation     ", style="HEADER")

    diagonal_S_SCA = diagonal_matrix(S_SCA[0])*(n)
    for i in range(n):
        if diagonal_S_SCA[i,i] == 0:
            diagonal_S_SCA[i,i] = 1
    det_SCA = logdet(diagonal_S_SCA)

    # Integrate perfect hints on coordinates which are immediately guessed
    proba_success = 1.
    guesses = 0
    for i in sorted_guesses:
        if proba_success >= 1-1/n/2:
            v = vec(Id[i])
            guess_val = guess_dict[i][0]
            guess_prob = guess_dict[i][1]
            if dbdd_SCA.integrate_perfect_hint(v, guess_val, force = False):
                guesses += 1
                proba_success *= guess_prob
        else:
            av = mu_SCA[0,i-m]
            var = S_SCA[0,i-m]
            v = vec(Id[i])
            if var > 0:
                dbdd_SCA.integrate_approx_hint(v, av, var, aposteriori=True, estimate=False)

    return dbdd_SCA, (det_SCA, 0, 0)

def intersect_SCA_ellipsoid(dbdd_int, mu_SCA, S_SCA, guess_dict, sorted_guesses, all_coordinates_attack=False, nointersection=False, knownscaling=False):
    """
    Intersects the SC ellipsoid with the DF ellipsoid. Can use the optimized method
    for intersection, or the alternative approach
    :dbdd_int: the dbdd instance for the intersected ellipoid
    :mu_SCA: the mean of the SCA ellipsoid
    :S_SCA: the 1 by n+m+1 dimensional matrix representing the diagonal of the SCA ellipsoid
    :guess_dict: the dictionary of coordinates values to guess and their probabilities
    :sorted_guesses: the sorted guess dictionary
    :all_coordinates_attack: whether or not to use the all-coordinates method for ellipsoid intersection
    :nointersection: the baseline case
    :knownscaling: whether or not the exact ellipsoid norms are assumed to be known
    """
    # Initialize variables
    n = dbdd_int.embedded_instance.n
    m = dbdd_int.embedded_instance.m

    # Print attack type
    if verbose:
        if all_coordinates_attack:
            if knownscaling:
                logging("  Known All Coordinates Attack Estimation  ", style="HEADER")
            else:
                logging(" Unknown All Coordinates Attack Estimation ", style="HEADER")
        elif nointersection:
                logging("         Baseline Attack Estimation        ", style="HEADER")
        else:
            if knownscaling:
                logging("    Known Conditional Attack Estimation    ", style="HEADER")
            else:
                logging("   Unknown Conditional Attack Estimation   ", style="HEADER")

    # Preprocessing
    intersect_indices, guesses, coordinates_guessed, mu_decfail_full, S_decfail_full, full_mu_SCA, full_S_SCA, baseline_norm_p, decfail_norm_p, S_baseline, mu_baseline, S_decfail, mu_decfail, valid_instance, S_baseline_n_g, mu_baseline_n_g = process_before_intersection(dbdd_int, mu_SCA, S_SCA, guess_dict, sorted_guesses, all_coordinates_attack, nointersection, knownscaling, n, m)

    #Calculate the new ellispoid
    try:
        if nointersection:
            S_int = matrix(S_baseline)
            mu_int = matrix(mu_baseline[0])
        else:
            mu_int, S_int = ellipsoid_intersection(mu_decfail, S_decfail, mu_baseline, S_baseline, tolerance=1.48e-08) #<=1
    except ValueError:
        # If the intersection fails, resort to the baseline ellipsoid
        mu_int = matrix(mu_baseline[0])
        S_int = matrix(S_baseline)
        if not nointersection:
            logging("Ellipsoid Intersection FAILED. Resorting to baseline ellipsoid.")
            if all_coordinates_attack:
                logging("All Coordinates Attack Used")     
            else:
                logging("Conditional Attack Used")
            logging("Indices %4d"% len(intersect_indices))

    # Postprocessing
    secret, S_baseline, mu_baseline, S_decfail, mu_decfail, S_int, mu_int, norm_scaling = process_after_intersection(dbdd_int, all_coordinates_attack, nointersection, knownscaling, intersect_indices, n, m, guesses, coordinates_guessed, full_mu_SCA, full_S_SCA, baseline_norm_p, decfail_norm_p, S_baseline, mu_baseline, S_decfail, mu_decfail, S_baseline_n_g, mu_baseline_n_g, S_int, mu_int)

    # Calculates and pritns the determinants and norms of the ellipsoids
    det_baseline, det_int, vol_change = print_norm_det_data(nointersection, knownscaling, intersect_indices, n, m, guesses, coordinates_guessed, S_decfail_full, mu_decfail_full, secret, S_baseline, mu_baseline, S_decfail, mu_decfail, S_int, mu_int)


    # Make initial guesses (which are correct with probability 1-1/n/2)
    proba_success = 1.
    guesses = 0
    Id = identity_matrix(n + m)
    for i in sorted_guesses:
        if proba_success >= 1-1/n/2:
            v = vec(Id[i])
            guess_val = guess_dict[i][0]
            guess_prob = guess_dict[i][1]
            if dbdd_int.integrate_perfect_hint(v, guess_val, force = False):
                S_int[i-m,i-m] = 0
                S_baseline[i-m,i-m] = 0
                guesses += 1
                proba_success *= guess_prob


    # Increase dimension back up to n+m+1 dimensions
    mu_int, S_int = extend_ellipsoid(dbdd_int, mu_int, S_int/(n))

    # Scales error coordinates as well in the case where the ellipsoid norms are not known. This calibrates the beta estimates correctly according to eq 11 in the paper
    if not nointersection and not knownscaling:
        S_int, det_baseline, det_int, vol_change = fix_ellipsoid_scaling(dbdd_int, n, m, coordinates_guessed, S_decfail_full, S_baseline, mu_baseline, S_int, norm_scaling)
        
    dbdd_int.mu = mu_int
    dbdd_int.S = S_int

    return valid_instance, dbdd_int, (det_baseline, vol_change, det_int)

def process_before_intersection(dbdd_int, mu_SCA, S_SCA, guess_dict, sorted_guesses, all_coordinates_attack, nointersection, knownscaling, n, m):
    """
    Creates the two ellipsoids which will be intersected.
    This includes removing the coordinates that will be guessed, choosing the set P and scaling the ellipsoids

    :dbdd_int: the dbdd instance for the intersected ellipoid
    :mu_SCA: the mean of the SCA ellipsoid
    :S_SCA: the 1 by n+m+1 dimensional matrix representing the diagonal of the SCA ellipsoid
    :guess_dict: the dictionary of coordinates values to guess and their probabilities
    :sorted_guesses: the sorted guess dictionary
    :all_coordinates_attack: whether or not to use the all-coordinates method for ellipsoid intersection
    :nointersection: the baseline case
    :knownscaling: whether or not the exact ellipsoid norms are assumed to be known
    :n: the LWE n parameter
    :m: the LWE m parameter
    """
    intersect_indices = []

    # Identify which coordinates will be guessed
    Id = identity_matrix(n + m)
    proba_success = 1.
    guesses = 0
    coordinates_guessed = vec(zeros(n+m))
    for i in sorted_guesses:
        if proba_success >= prob_limit:
            v = vec(Id[i])
            guesses += 1
            guess_prob = guess_dict[i][1]
            proba_success *= guess_prob 
            coordinates_guessed = coordinates_guessed + v
    if verbose:
        logging("Guesses: %4d"% guesses)

    intersect_dim = n - guesses

    # Creates the DF matrix
    s_s = 0.25#25 # CCS1
    #s_s = 2 # NIST1
    if verbose:
        logging("sigma_df value is: %3.4f"% s_s)
    S_decfail_full = diagonal_matrix([s_s]*(n))
    S_decfail_full = S_decfail_full*(n)

    # Remove guessed coordinates from the matrix
    full_mu_SCA = matrix(mu_SCA[0])
    full_S_SCA = matrix(S_SCA[0])
    mu_SCA = matrix(mu_SCA[0,0:intersect_dim])
    S_SCA = matrix(S_SCA[0,0:intersect_dim])
    secret = dbdd_int.u[0,n:n+intersect_dim]
    j = 0
    for i in range(n):
        if coordinates_guessed[0][i+m] != 1:
            mu_SCA[0,j] = full_mu_SCA[0,i]
            S_SCA[0,j] = full_S_SCA[0,i]
            secret[0,j] = dbdd_int.u[0,n+i]
            j += 1

    # Find the indices to intersect on
    if all_coordinates_attack:
        for i in range(intersect_dim):
            intersect_indices += [i]
    elif nointersection:
        intersect_indices = []
    else:
        for i in range(intersect_dim):
            if S_SCA[0,i] > s_s/5 and S_SCA[0,i] < s_s:
                intersect_indices += [i]
    if verbose:
        logging("Number of elements in intersect_indices: %4d"% len(intersect_indices))
    
    # Samples the DF mean and creates the DF and baseline ellipsoids
    mu_decfail_full, baseline_norm_p, decfail_norm_p, S_baseline, mu_baseline, S_decfail, mu_decfail, valid_instance = sample_df_ellipsoid(dbdd_int, mu_SCA, S_SCA, nointersection, knownscaling, intersect_indices, n, m, coordinates_guessed, S_decfail_full, full_mu_SCA, full_S_SCA, secret)

    # Remove coordinates not in the set P
    S_decfail_n_g = 0
    mu_decfail_n_g = 0
    S_baseline_n_g = 0
    mu_baseline_n_g = 0
    if not all_coordinates_attack and not nointersection:
        S_decfail_n_g = matrix(S_decfail)
        mu_decfail_n_g = matrix(mu_decfail[0])
        S_baseline_n_g = matrix(S_baseline)
        mu_baseline_n_g = matrix(mu_baseline[0])
        S_decfail = matrix(S_decfail_n_g[0:len(intersect_indices), 0:len(intersect_indices)])
        S_baseline = matrix(S_decfail)
        mu_decfail = mu_decfail_n_g[0,0:len(intersect_indices)]
        mu_baseline = matrix(mu_decfail)

        j = 0
        for i in intersect_indices:
            S_decfail[j,j] = S_decfail_n_g[i,i]
            mu_decfail[0,j] = mu_decfail_n_g[0,i]
            S_baseline[j,j] = S_baseline_n_g[i,i]
            mu_baseline[0,j] = mu_baseline_n_g[0,i]
            j += 1

    # Scale the ellipsoids before intersection
    if not nointersection:
        if knownscaling:
            S_decfail = S_decfail*decfail_norm_p
            S_baseline = S_baseline*baseline_norm_p
        else:
            S_decfail = S_decfail*0.85

    return intersect_indices, guesses, coordinates_guessed, mu_decfail_full, S_decfail_full, full_mu_SCA, full_S_SCA, baseline_norm_p, decfail_norm_p, S_baseline, mu_baseline, S_decfail, mu_decfail, valid_instance, S_baseline_n_g, mu_baseline_n_g

def sample_df_ellipsoid(dbdd_int, mu_SCA, S_SCA, nointersection, knownscaling, intersect_indices, n, m, coordinates_guessed, S_decfail_full, full_mu_SCA, full_S_SCA, secret):
    """
    Samples the mean of the DF ellipsoid, and calculates the baseline ellipsoid. Ensures that the secret is within the DF ellipsoid

    :dbdd_int: the dbdd instance for the intersected ellipoid
    :mu_SCA: the mean of the SCA ellipsoid
    :S_SCA: the 1 by n+m+1 dimensional matrix representing the diagonal of the SCA ellipsoid
    :nointersection: the baseline case
    :knownscaling: whether or not the exact ellipsoid norms are assumed to be known
    :intersect_indices: a list of the indices in [0,n) which form the set P
    :n: the LWE n parameter
    :m: the LWE m parameter
    :coordinates_guessed: a list of the indices in [0,n+m) which are guessed
    :S_decfail_full: the covriance matrix of the decryption failure ellipsoid on all n coordinates
    :full_mu_SCA: the mean of the side channel ellipsoid on all n secret coordinates
    :full_S_SCA: the covariance matrix of the side channel ellipsoid for all n secret coordinates
    :secret: the n-dimensional LWE secret
    """

    ellipsoid_norm_decfail = matrix([[2]])
    S_SCA = S_SCA*(n)
    while ellipsoid_norm_decfail[0,0] > 1:
        mu_decfail_full = matrix(dbdd_int.u[0,n:n+m])
        random_vector = rng.multivariate_normal([0]*n, S_decfail_full/n)
        mu_decfail_full = matrix(mu_decfail_full + random_vector)

        # Create the DF matrix for the intersection
        mu_decfail_short = matrix(mu_SCA[0])
        S_decfail_short = matrix(S_SCA[0])
        baseline_norm_p = 0
        decfail_norm_p = 0
        j = 0
        p_ind = 0
        for i in range(n):
            if coordinates_guessed[0][i+m] != 1:
                if S_SCA[0,j] > S_decfail_full[i,i]:
                    S_decfail_short[0,j] = S_decfail_full[i,i]
                    mu_decfail_short[0,j] = mu_decfail_full[0,i]
                if p_ind < len(intersect_indices) and intersect_indices[p_ind] == j:
                    S_decfail_short[0,j] = S_decfail_full[i,i]
                    mu_decfail_short[0,j] = mu_decfail_full[0,i]
                    decfail_norm_p += float((secret[0,j] - mu_decfail_short[0,j]))**2/(S_decfail_short[0,j])
                    if S_SCA[0,j] > S_decfail_full[i,i]:
                        baseline_norm_p += float((secret[0,j] - mu_decfail_full[0,i])**2/S_decfail_full[i,i])
                    else:
                        baseline_norm_p += float((secret[0,j] - mu_SCA[0,j])**2/S_SCA[0,j])
                    p_ind += 1
                j += 1

        # Ensure that the DF ellipsoid norm is below 1
        ellipsoid_norm_decfail = (mu_decfail_full - dbdd_int.u[0,n:n+m]) * S_decfail_full.inverse() * (mu_decfail_full - dbdd_int.u[0,n:n+m]).transpose()
        if len(intersect_indices) > 0 and verbose:
           logging("DF norm: %1.10f, just P: %1.10f"%(ellipsoid_norm_decfail[0,0], decfail_norm_p*n/len(intersect_indices)))
    
    # Create baseline ellipsoid
    j = 0
    for i in range(n):
        if coordinates_guessed[0][i+m] != 1:
            if S_SCA[0,j] > S_decfail_full[i,i]:    
                S_SCA[0,j] = S_decfail_full[i,i]
                mu_SCA[0,j] = mu_decfail_full[0,i]
                full_S_SCA[0,i] = S_decfail_full[i,i]/n
                full_mu_SCA[0,i] = mu_decfail_full[0,i]
            j += 1
 
    S_baseline = diagonal_matrix(S_SCA[0])
    mu_baseline = mu_SCA

    # Calculate the norm of the SCA ellipsoid
    baseline_norm = (secret - mu_baseline) * S_baseline.inverse() * (secret - mu_baseline).transpose()
    if verbose:
        logging("Baseline Norm: %1.10f, Baseline Norm_p: %1.10f"% (baseline_norm[0][0], baseline_norm_p))
    
    S_decfail = diagonal_matrix(S_decfail_short[0])
    mu_decfail = matrix(mu_decfail_short[0])

    if verbose:
        logging("DF norm: %1.10f, from P: %1.10f"%(ellipsoid_norm_decfail[0][0], decfail_norm_p))

    valid_instance = True
    if not nointersection and not knownscaling and decfail_norm_p/baseline_norm_p > 0.9:
        logging("DF and baseline norms do not satisfy the bound.")
        valid_instance = False

    if not nointersection and knownscaling and decfail_norm_p >= baseline_norm_p:
        logging("DF norm is higher than baseline norm.")
        valid_instance = False
    
    return mu_decfail_full, baseline_norm_p, decfail_norm_p, S_baseline, mu_baseline, S_decfail, mu_decfail, valid_instance

def process_after_intersection(dbdd_int, all_coordinates_attack, nointersection, knownscaling, intersect_indices, n, m, guesses, coordinates_guessed, full_mu_SCA, full_S_SCA, baseline_norm_p, decfail_norm_p, S_baseline, mu_baseline, S_decfail, mu_decfail, S_baseline_n_g, mu_baseline_n_g, S_int, mu_int):
    """
    Converts the intersected ellipsoid back to n-dimensions, and fixes the scaling (more is done at the end of fix_ellipsoid_scaling in the unknown norm case)

    :dbdd_int: the dbdd instance for the intersected ellipoid
    :mu_SCA: the mean of the SCA ellipsoid
    :S_SCA: the 1 by n+m+1 dimensional matrix representing the diagonal of the SCA ellipsoid
    :all_coordinates_attack: whether or not to use the all-coordinates method for ellipsoid intersection
    :nointersection: the baseline case
    :knownscaling: whether or not the exact ellipsoid norms are assumed to be known
    :intersect_indices: a list of the indices in [0,n) which form the set P
    :n: the LWE n parameter
    :m: the LWE m parameter
    :guesses: the number of guesses which will be made
    :coordinates_guessed: a list of the indices in [0,n+m) which are guessed
    :full_mu_SCA: the mean of the side channel ellipsoid on all n secret coordinates
    :full_S_SCA: the covariance matrix of the side channel ellipsoid for all n secret coordinates
    :baseline_norm_p: the norm of the baseline ellipsoid restricted to the set P
    :decfail_norm_p: the norm of the decryption failure ellipsoid restricted to the set P
    :S_baseline: the P-dimensional covariance matrix for the baseline ellipsoid
    :mu_baseline: the P-dimensional mean for the baseline ellipsoid
    :S_decfail: the P-dimensional covariance matrix for the decryption failure ellipsoid
    :mu_decfail: the P-dimensional mean  for the decryption failure ellipsoid
    :S_baseline_n_g: the covariance matrix of the baseline ellipsoid on all non-guessed coordinates
    :mu_baseline_n_g: the covariance matrix of the baseline ellipsoid on all non-guessed coordinates
    :S_int: the P-dimensional covariance matrix of the intersected ellipsoid
    :mu_int: the P-dimensional mean of the intersected ellipsoid
    """
    
    # Scale back down in the known norm case
    if knownscaling:
        S_int = S_int/baseline_norm_p
        S_decfail = S_decfail/decfail_norm_p
        S_baseline = S_baseline/baseline_norm_p

    # Add back in coordinates not in P
    if not all_coordinates_attack and not nointersection:
        S_decfail = S_baseline_n_g
        mu_decfail = mu_baseline_n_g
        S_baseline = S_baseline_n_g
        mu_baseline = mu_baseline_n_g

        S_int_p = matrix(S_int)
        mu_int_p = matrix(mu_int[0])
        S_int = matrix(S_baseline)
        mu_int = matrix(mu_baseline[0])

        j = 0
        for i in intersect_indices:
            S_int[i,i] = S_int_p[j,j]
            mu_int[0,i] = mu_int_p[0,j]
            j += 1

    # Add back in guessed coordinates
    mu_baseline2 = full_mu_SCA[0,0:n]
    mu_decfail2 = matrix(mu_baseline2)
    mu_int2 = matrix(mu_baseline2)
    S_baseline2 = full_S_SCA[0,0:n]
    S_baseline2 = diagonal_matrix(S_baseline2[0])*(n)
    S_decfail2 = matrix(S_baseline2)
    S_int2 = matrix(S_baseline2)
    secret = dbdd_int.u[0,n:n+m]
    j = 0
    for i in range(n):
        if coordinates_guessed[0][i+m] != 1:
            mu_decfail2[0,i] = mu_decfail[0,j]
            mu_int2[0,i] = mu_int[0,j]
            S_decfail2[i,i] = S_decfail[j,j]
            S_int2[i,i] = S_int[j,j]
            j += 1
    mu_baseline = matrix(mu_baseline2)
    mu_decfail = matrix(mu_decfail2)
    mu_int = matrix(mu_int2)
    S_baseline = matrix(S_baseline2)
    S_decfail = matrix(S_decfail2)
    S_int = matrix(S_int2)

    # Calculate the intersected norm on the coordinates in P
    norm_scaling = 1
    if not nointersection:
        int_norm_p = 0
        j = 0
        p_ind = 0
        for i in range(n):
            if coordinates_guessed[0][i+m] != 1:
                if p_ind < len(intersect_indices) and intersect_indices[p_ind] == j:
                    int_norm_p += float((secret[0,i] - mu_int[0,i])**2/S_int[i,i])
                    p_ind += 1
                j += 1
        logging("Intersected Norm_p: %1.10f"% int_norm_p)

        # Fix scaling in the unknown norm case
        if not knownscaling:
            norm_scaling = (2*n-guesses-n*(baseline_norm_p - int_norm_p))/(2*n - guesses)
            logging("Scaling by %1.5f to correct norm"% norm_scaling)

            for i in range(n):
                if coordinates_guessed[0][i+m] != 1:
                    S_int[i,i] = S_int[i,i]*norm_scaling

            int_norm_p = 0
            j = 0
            p_ind = 0
            for i in range(n):
                if coordinates_guessed[0][i+m] != 1:
                    if p_ind < len(intersect_indices) and intersect_indices[p_ind] == j:
                        int_norm_p += float((secret[0,i] - mu_int[0,i])**2/S_int[i,i])
                        p_ind += 1
                    j += 1
            logging("Intersected Norm_p: %1.10f"% int_norm_p)

    return secret, S_baseline, mu_baseline, S_decfail, mu_decfail, S_int, mu_int, norm_scaling

def print_norm_det_data(nointersection, knownscaling, intersect_indices, n, m, guesses, coordinates_guessed, S_decfail_full, mu_decfail_full, secret, S_baseline, mu_baseline, S_decfail, mu_decfail, S_int, mu_int):
    """
    Calculates and prints the ellipsoid norms and determinants of the ellipsoids before and after intersection

    :nointersection: the baseline case
    :knownscaling: whether or not the exact ellipsoid norms are assumed to be known
    :intersect_indices: a list of the indices in [0,n) which form the set P
    :n: the LWE n parameter
    :m: the LWE m parameter
    :guesses: the number of guesses which will be made
    :coordinates_guessed: a list of the indices in [0,n+m) which are guessed
    :S_decfail_full: the covariance matrix of the n-dimensional decryption failure ellipsoid
    :mu_decfail_full: the mean of the n-dimensional decryption failure ellipsoid
    :secret: the n-dimensional LWE secret
    :S_baseline: the n-dimensional covariance matrix for the baseline ellipsoid
    :mu_baseline: the n-dimensional mean for the baseline ellipsoid
    :S_decfail: the n-dimensional covariance matrix for the decryption failure ellipsoid
    :mu_decfail: the n-dimensional mean  for the decryption failure ellipsoid
    :S_int: the n-dimensional covariance matrix of the intersected ellipsoid
    :mu_int: the n-dimensional mean of the intersected ellipsoid
    """
    # Print out all ellipsoids on all coordinates for debugging
    if verbose:
        j = 0
        k = 0
        logging("guesses: %4d"% guesses)
        for i in range(n):
            if coordinates_guessed[0,i+m] != 1:
                if j < len(intersect_indices) and intersect_indices[j] == k:
                    logging("DF: %3.6f %3.6f  Post: %3.6f %3.6f  DF_p: %3.6f %3.6f  Int: %3.6f %3.6f  Intersect: Yes Guess: %2.1f %2d %3d %1d"%(mu_decfail_full[0,i], S_decfail_full[i,i], mu_baseline[0,i], S_baseline[i,i], mu_decfail[0,i], S_decfail[i,i], mu_int[0,i], S_int[i,i], coordinates_guessed[0,i+m], secret[0,i], i, j))
                    j += 1
                    break
                    j += 10000 #prints only first coordinate of intersection
                k += 1

    # Calculate the determinants before and after intersection
    det_baseline = 1
    det_int = 1
    vol_change = 0
    if nointersection or knownscaling:
        det_baseline = logdet(S_baseline)
        det_int = logdet(S_int)
        vol_change = (det_baseline - det_int)/2
        if verbose:
            logging("ln det(DF matrix [n coords]): %3.2f"% logdet(S_decfail_full))
            logging("ln det(Baseline matrix [n coords]): %3.2f"% det_baseline)
            logging("ln det(Intersected matrix [n coords]): %3.2f"% det_int)       

    # Check if the secret is contained in the ellipsoid
    ellipsoid_norm_decfail = (mu_decfail_full - secret) * S_decfail_full.inverse() * (mu_decfail_full - secret).transpose()
    ellipsoid_norm_baseline = (secret - mu_baseline) * S_baseline.inverse() * (secret - mu_baseline).transpose()
    ellipsoid_norm_int = 0
    secret_norm = secret * secret.transpose()
    for i in range(n):
        ellipsoid_norm_int += (secret[0,i] - mu_int[0,i])**2 / S_int[i,i]
    if verbose:
        logging("Secret norm: %4.4f"% secret_norm[0,0])      
        logging("DF norm: %1.10f"% ellipsoid_norm_decfail[0][0])
        logging("Baseline norm: %1.10f"% ellipsoid_norm_baseline[0][0])
        logging("Intersected norm: %1.10f"% ellipsoid_norm_int)
    elif ellipsoid_norm_int[0][0] > 1 or ellipsoid_norm_decfail[0][0] > 1 or ellipsoid_norm_baseline[0][0] > 1:
        logging("SECRET NOT CONTAINED IN ELLIPSOID. DF norm: %1.3f Post norm: %1.3f Int norm: %1.3f"% (ellipsoid_norm_decfail[0,0], ellipsoid_norm_baseline[0,0], ellipsoid_norm_int))

    return det_baseline, det_int, vol_change

def fix_ellipsoid_scaling(dbdd_int, n, m, coordinates_guessed, S_decfail_full, S_baseline, mu_baseline, S_int, norm_scaling):
    """
    Fixes the ellipsoid scaling in the case where the ellipsoid norms are not known. This calibrates the beta estimates correctly according to eq 11 in the paper

    :dbdd_int: the dbdd instance for the intersected ellipoid
    :n: the LWE n parameter
    :m: the LWE m parameter
    :coordinates_guessed: a list of the indices in [0,n+m) which are guessed
    :S_decfail_full: the n-dimensional decryption failure ellipsoid
    :S_baseline: the n-dimensional covariance matrix for the baseline ellipsoid
    :mu_baseline: the n-dimensional mean for the baseline ellipsoid
    :S_int: the n+m+1-dimensional covariance matrix of the intersected ellipsoid
    :norm_scaling: the factor to scale the ellipsoid norm by
    """

    # Scales error coordinates as well in the case where the ellipsoid norms are not known. This calibrates the beta estimates correctly according to eq 11 in the paper
    for i in range(m):
        S_int[i] = S_int[i] * norm_scaling

    mu_baseline, S_baseline = extend_ellipsoid(dbdd_int, mu_baseline, S_baseline/(n))

    det_baseline = logdet(diagonal_matrix((S_baseline[0:n+m] + coordinates_guessed)[0]))
    det_int = logdet(diagonal_matrix((S_int[0:n+m] + coordinates_guessed)[0]))
    vol_change = (det_baseline - det_int)/2
    if verbose:
        logging("ln det(DF matrix): %3.2f"% logdet(S_decfail_full))
        logging("ln det(Baseline matrix): %3.2f"% det_baseline)
        logging("ln det(Intersected matrix): %3.2f"% det_int)
    
    return S_int, det_baseline, det_int, vol_change

def integrate_hints_guesses(dbdd_to_guess, guess_dict, sorted_guesses):
    """
    Integrate guesses into the dbdd instance, and then q-vectors
    :dbdd_to_guess: the dbdd instance to integrate guesses with
    :guess_dict: the dictionary of coordinates values to guess and their probabilities
    :sorted_guesses: the sorted guess dictionary
    """
    n = dbdd_to_guess.embedded_instance.n
    m = dbdd_to_guess.embedded_instance.m
    Id = identity_matrix(n + m)    

    # Calculate the bikz without the guesses
    duplicate_dbbd_to_guess = deepcopy(dbdd_to_guess)
    duplicate_dbbd_to_guess.integrate_q_vectors(q, min_dim = m + 1, indices=range(n))
    (beta_without_guesses, _) = duplicate_dbbd_to_guess.estimate_attack()
    if verbose:
        logging("Beta with hints before guesses: %3.2f"% beta_without_guesses)

    # Integrate perfect hints on coordinates which can be guessed
    secret = dbdd_to_guess.u
    proba_success = 1.
    guesses = 0
    wrong_coordinates = 0
    for i in sorted_guesses:
        if dbdd_to_guess.S[i] == 0 or proba_success > 1-1/n/2: # These perfect hints are already integrated
            guesses += 1
            guess_prob = guess_dict[i][1]
            proba_success *= guess_prob
        elif proba_success >= prob_limit:
            v = vec(Id[i])
            guess_val = guess_dict[i][0]
            guess_prob = guess_dict[i][1]
            if dbdd_to_guess.integrate_perfect_hint(v, guess_val, force = False):
                guesses += 1
                proba_success *= guess_prob
                # Test to make sure that the guesses are accurate for the experiment  
                if guess_val != secret[0][i]:
                    wrong_coordinates += 1
    logging("Wrong guesses: %4d" % wrong_coordinates)

    # Check the bikz before q-vectors, then integrate q-vectors
    beta_after_guesses, _ = dbdd_to_guess.estimate_attack()
    dbdd_to_guess.integrate_q_vectors(q, min_dim = m + 1, indices=range(n))
    if verbose:
        logging("Beta after guesses before hints: %3.2f"% beta_after_guesses)
        logging("Beta after hints and guesses: %3.2f"% dbdd_to_guess.beta)
    
    return dbdd_to_guess, (beta_without_guesses, proba_success, guesses)


def estimate_SCA(dbdd, measured, frodo_distribution):
    """ 
    This function evaluates the security loss after Bos et al attack.
    Comuptes the estimated bikz after the original dbdd attack, and the ellipsoid
    attack using the original dbdd and baseline methods, and the two intersection methods.
    :dbdd: instance of the class DBDD
    :measured: table representing the (simulated) information
    given by Bos et al attack
    :frodo_distribution: the frodo distribution being used
    """

    mu_SCA, S_SCA, guess_dict, sorted_guesses = gen_aposteriori_ellip(dbdd, measured, frodo_distribution)

    # integrate aposteriori ellipsoid into dbdd instance
    dbdd_SCA = dbdd.embedded_instance.embed_into_DBDD_predict_diag()
    dbdd_baseline = dbdd.embedded_instance.embed_into_DBDD_predict_diag()
    dbdd_conditional = dbdd.embedded_instance.embed_into_DBDD_predict_diag() 
    dbdd_all_coords = dbdd.embedded_instance.embed_into_DBDD_predict_diag()
    dbdd_known_conditional = dbdd.embedded_instance.embed_into_DBDD_predict_diag() 
    dbdd_known_all_coords = dbdd.embedded_instance.embed_into_DBDD_predict_diag()
        
    # The original dbdd attack using the side channel data
    dbdd_SCA, SCA_volumes = original_SCA_attack(dbdd_SCA, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], sorted_guesses, guess_dict)
    dbdd_SCA.estimate_attack(silent=False)
    dbdd_SCA, SCA_guess_data = integrate_hints_guesses(dbdd_SCA, guess_dict, sorted_guesses)
    SCA_beta = dbdd_SCA.beta
    
    # The baseline attack
    _, dbdd_baseline, baseline_volumes = intersect_SCA_ellipsoid(dbdd_baseline, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], guess_dict, sorted_guesses, False, True, False)
    dbdd_baseline.estimate_attack(silent=False)
    dbdd_baseline, baseline_guess_data = integrate_hints_guesses(dbdd_baseline, guess_dict, sorted_guesses)
    baseline_beta = dbdd_baseline.beta

    # The attack intersecting on all coordinates, assuming that the norm values are not known
    valid_all_coords, dbdd_all_coords, all_coords_volumes = intersect_SCA_ellipsoid(dbdd_all_coords, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], guess_dict, sorted_guesses, True, False, False)
    dbdd_all_coords.estimate_attack(silent=False)
    dbdd_all_coords, all_coords_guess_data = integrate_hints_guesses(dbdd_all_coords, guess_dict, sorted_guesses)
    all_coords_beta = dbdd_all_coords.beta

    # The attack intersecting on all coordinates, assuming that the norm values are known
    valid_known_all_coords, dbdd_known_all_coords, known_all_coords_volumes = intersect_SCA_ellipsoid(dbdd_known_all_coords, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], guess_dict, sorted_guesses, True, False, True)
    dbdd_known_all_coords.estimate_attack(silent=False)
    dbdd_known_all_coords, known_all_coords_guess_data = integrate_hints_guesses(dbdd_known_all_coords, guess_dict, sorted_guesses)
    known_all_coords_beta = dbdd_known_all_coords.beta
  
    # The attack intersecting on the coordinates satisfying the condition, assuming that the norm values are not known
    valid_conditional, dbdd_conditional, conditional_volumes = intersect_SCA_ellipsoid(dbdd_conditional, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], guess_dict, sorted_guesses, False, False, False)
    dbdd_conditional.estimate_attack(silent=False)
    dbdd_conditional, conditional_guess_data = integrate_hints_guesses(dbdd_conditional, guess_dict, sorted_guesses)
    conditional_beta = dbdd_conditional.beta

    # The attack intersecting on the coordinates satisfying the condition, assuming that the norm values are known
    valid_known_conditional, dbdd_known_conditional, known_conditional_volumes = intersect_SCA_ellipsoid(dbdd_known_conditional, mu_SCA[0,n:n+m], S_SCA[0,n:n+m], guess_dict, sorted_guesses, False, False, True)
    dbdd_known_conditional.estimate_attack(silent=False)
    dbdd_known_conditional, known_conditional_guess_data = integrate_hints_guesses(dbdd_known_conditional, guess_dict, sorted_guesses)
    known_conditional_beta = dbdd_known_conditional.beta

    return (np.array([SCA_beta]), SCA_volumes, SCA_guess_data), (np.array([baseline_beta]), baseline_volumes, baseline_guess_data), (np.array([conditional_beta]), conditional_volumes, conditional_guess_data, valid_conditional), (np.array([all_coords_beta]), all_coords_volumes, all_coords_guess_data, valid_all_coords), (np.array([known_conditional_beta]), known_conditional_volumes, known_conditional_guess_data, valid_known_conditional), (np.array([known_all_coords_beta]), known_all_coords_volumes, known_all_coords_guess_data, valid_known_all_coords) 

def one_experiment(id, aargs):
    """ 
    Runs one experiment of the SCA attack for each method.
    Samples the LWE instance and side channel distribution
    :id: the id of this instance
    :aargs: the parameters for the experiment
    """

    set_random_seed()
    np_seed()
    random.seed()

    if verbose:
        logging("Instance %4d" % id)
    n, m, q, D_s, frodo_distribution, measures = aargs

    lwe_instance = LWE(n, q, m, D_s, D_s, verbosity=0)
    
    dbdd = lwe_instance.embed_into_DBDD_predict_diag()
    
    # Ensures that the norm of the SCA ellipsoid is below 1. Resamples the SC distribution if it is.
    SCA_ellipsoid_norm = matrix([[2]])
    while SCA_ellipsoid_norm[0][0] > 1:
        measured = [simu_measured(dbdd.u[0, n+i], measures) for i in range(n)]
        
        mu_SCA, S_SCA, _, _ = gen_aposteriori_ellip(dbdd, measured, frodo_distribution)
        mu_SCA = mu_SCA[0,n:n+m]
        S_SCA = S_SCA[0,n:n+m]*(n)
        SCA_ellipsoid_norm = (lwe_instance.s - mu_SCA) * diagonal_matrix(S_SCA[0]).inverse() * (lwe_instance.s - mu_SCA).transpose()
        if verbose:
            logging("Side Channel Attack ellipsoid norm: %2.4f"% SCA_ellipsoid_norm[0][0])
    
    
    results_SCA, results_baseline, results_conditional, results_all_coords, results_known_conditional, results_known_all_coords = estimate_SCA(dbdd, measured, frodo_distribution)
    
    if verbose:
        logging("DBDD Attack:                           %3.2f bikz %3.2f bikz %4d %0.2f, volume: %3.2f" % (results_SCA[2][0], results_SCA[0][0], results_SCA[2][2], results_SCA[2][1], results_SCA[1][0]), style="HEADER")
        logging("Baseline Attack:                       %3.2f bikz %3.2f bikz %4d %0.2f, volume: %3.2f" % (results_baseline[2][0], results_baseline[0][0], results_baseline[2][2], results_baseline[2][1], results_baseline[1][0]), style="HEADER")
        logging("All Coordinates Attack:                %3.2f bikz %3.2f bikz %4d %0.2f, volumes: %3.2f %3.2f %3.2f" % (results_all_coords[2][0], results_all_coords[0][0], results_all_coords[2][2], results_all_coords[2][1], results_all_coords[1][0], results_all_coords[1][1], results_all_coords[1][2]), style="HEADER")
        logging("All Coordinates Attack (known norm):   %3.2f bikz %3.2f bikz %4d %0.2f, volumes: %3.2f %3.2f %3.2f" % (results_known_all_coords[2][0], results_known_all_coords[0][0], results_known_all_coords[2][2], results_known_all_coords[2][1], results_known_all_coords[1][0], results_known_all_coords[1][1], results_known_all_coords[1][2]), style="HEADER")
        logging("Conditional Attack:                    %3.2f bikz %3.2f bikz %4d %0.2f, volumes: %3.2f %3.2f %3.2f" % (results_conditional[2][0], results_conditional[0][0], results_conditional[2][2], results_conditional[2][1], results_conditional[1][0], results_conditional[1][1], results_conditional[1][2]), style="HEADER")
        logging("Conditional Attack (known norm):       %3.2f bikz %3.2f bikz %4d %0.2f, volumes: %3.2f %3.2f %3.2f\n" % (results_known_conditional[2][0], results_known_conditional[0][0], results_known_conditional[2][2], results_known_conditional[2][1], results_known_conditional[1][0], results_known_conditional[1][1], results_known_conditional[1][2]), style="HEADER")

    return results_SCA, results_baseline, results_conditional, results_all_coords, results_known_conditional, results_known_all_coords

    
def run_experiment():
    """
    Runs the experiment with all parameter sets and with nb_tests_per_params instances for each set of parameters.

    """
    # Estimation
    import datetime
    ttt = datetime.datetime.now()

    global n, m, q

    for params in ['CCS1']:#'CCS1']:#, 'CCS2', 'CCS3', 'CCS4', 'NIST1', 'NIST2']:
        logging("Set of parameters: " + params)

        if params == 'NIST1':
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

        # Calculates the original security for this instance
        """  Original Security   """
        lwe_instance = LWE(n, q, m, D_s, D_s, verbosity=0)
        dbdd = lwe_instance.embed_into_DBDD_predict_diag()

        dbdd.integrate_q_vectors(q, indices=range(n, n + m))
        (beta, _) = dbdd.estimate_attack()
        logging("Decryption Failure attack without hints:  %3.2f bikz" % beta, style="HEADER")


        """  Refined Side channel attack  """

        # Reading the score tables from Bos et al. attack
        scores = []
        correct = []
        for seed in sca_seeds:
            for i in range(1, 9):
                data = loadmat('./Scores_tables_SCA/' + params + '/' +
                            bosetal_sigma + '/apostandcorrect' + str(param + 1) +
                            '_seed' + str(seed) +
                            'nbar' + str(i) + '.mat')
                scores += list(data['apostdist'])
                correct += list(data['correct'])
        measures = {}
        # the score tables are stored according to the secret coefficient. We use them
        # for generating the measurement.
        for key_guess in range(-len(frodo_distribution)+1, len(frodo_distribution)):
            inds = []
            for i, d in enumerate(correct):
                darray = zeros((1,1))
                darray[0,0] = d
                if recenter(darray) == key_guess:
                    inds += [i]
            measures[key_guess] = [scores[ind] for ind in inds]


        parameters = (n, m, q, D_s, frodo_distribution, measures)

        # Run the experiments
        res = map_drop(nb_tests_per_params, threads, one_experiment, (parameters))
        save(res, "full_experimental_results.sobj")

        # Print out summary data from each experiment
        total_tests_run = nb_tests_per_params
        valid_conditional_instances = 0
        valid_all_coords_instances = 0
        valid_known_conditional_instances = 0
        valid_known_all_coords_instances = 0
        for r in res:
            logging("DBDD Attack:                           %3.2f bikz %3.2f bikz %4d %0.2f, volume: %3.2f" % (r[0][2][0], r[0][0][0], r[0][2][2], r[0][2][1], r[0][1][0]), style="HEADER")
            logging("Baseline Attack:                       %3.2f bikz %3.2f bikz %4d %0.2f, volume: %3.2f" % (r[1][2][0], r[1][0][0], r[1][2][2], r[1][2][1], r[1][1][0]), style="HEADER")
            logging("All Coordinates Attack:                %3.2f bikz %3.2f bikz %4d %0.2f, volumes: %3.2f %3.2f %3.2f" % (r[3][2][0], r[3][0][0], r[3][2][2], r[3][2][1], r[3][1][0], r[3][1][1], r[3][1][2]), style="HEADER")
            logging("All Coordinates Attack (known norm):   %3.2f bikz %3.2f bikz %4d %0.2f, volumes: %3.2f %3.2f %3.2f" % (r[5][2][0], r[5][0][0], r[5][2][2], r[5][2][1], r[5][1][0], r[5][1][1], r[5][1][2]), style="HEADER")
            logging("Conditional Attack:                    %3.2f bikz %3.2f bikz %4d %0.2f, volumes: %3.2f %3.2f %3.2f" % (r[2][2][0], r[2][0][0], r[2][2][2], r[2][2][1], r[2][1][0], r[2][1][1], r[2][1][2]), style="HEADER")
            logging("Conditional Attack (known norm):       %3.2f bikz %3.2f bikz %4d %0.2f, volumes: %3.2f %3.2f %3.2f\n" % (r[4][2][0], r[4][0][0], r[4][2][2], r[4][2][1], r[4][1][0], r[4][1][1], r[4][1][2]), style="HEADER")
            if r[2][3]:
                valid_conditional_instances += 1
            if r[3][3]:
                valid_all_coords_instances += 1
            if r[4][3]:
                valid_known_conditional_instances += 1
            if r[5][3]:
                valid_known_all_coords_instances += 1

        logging("Total tests: %d"% total_tests_run)
        logging("Valid All coordinates instances: %d"% valid_all_coords_instances)
        logging("Valid All coordinates known instances: %d"% valid_known_all_coords_instances)
        logging("Valid Conditional instances: %d"% valid_conditional_instances)
        logging("Valid Conditional known instances: %d"% valid_known_conditional_instances)
                
        # Calculates the average values for each method                
        logging("AVERAGE RESULTS for parameters: " + params, style="HEADER")
        result_ranges = [1, 3, 3]
        results_SCA = zero_matrix(QQ,3,3)
        results_baseline = zero_matrix(QQ,3,3)
        results_conditional = zero_matrix(QQ,3,3)
        results_all_coords = zero_matrix(QQ,3,3)
        results_known_conditional = zero_matrix(QQ,3,3)
        results_known_all_coords = zero_matrix(QQ,3,3)
        
        for i in range(3):
            for j in range(result_ranges[i]):
                results_SCA[i,j] = sum([r[0][i][j] for r in res])/nb_tests_per_params
                results_baseline[i,j] = sum([r[1][i][j] for r in res])/nb_tests_per_params
                results_conditional[i,j] = sum([r[2][i][j] for r in res])/nb_tests_per_params
                results_all_coords[i,j] = sum([r[3][i][j] for r in res])/nb_tests_per_params
                results_known_conditional[i,j] = sum([r[4][i][j] for r in res])/nb_tests_per_params
                results_known_all_coords[i,j] = sum([r[5][i][j] for r in res])/nb_tests_per_params

        #Print the average value for each method
        logging("DBDD Attack with hints:                                        %3.2f bikz" % results_SCA[2][0], style="HEADER")
        logging("DBDD Attack with hints and guesses:                            %3.2f bikz" % results_SCA[0][0], style="HEADER")
        logging("DBDD Attack Number of guesses:                                %4d" % results_SCA[2][2], style="HEADER")
        logging("DBBD Attack Success probability:                               %3.2f\n" %results_SCA[2][1], style="HEADER")

        logging("Baseline Attack with hints:                                    %3.2f bikz" % results_baseline[2][0], style="HEADER")
        logging("Baseline Attack with hints and guesses:                        %3.2f bikz" % results_baseline[0][0], style="HEADER")
        logging("Baseline Attack Number of guesses:                            %4d" % results_baseline[2][2], style="HEADER")
        logging("Baseline Attack Success probability:                           %3.2f\n" %results_baseline[2][1], style="HEADER")
        
        logging("All Coordinates Attack with hints:                             %3.2f bikz" % results_all_coords[2][0], style="HEADER")
        logging("All Coordinates Attack with hints and guesses:                 %3.2f bikz" % results_all_coords[0][0], style="HEADER")
        logging("All Coordinates Attack Number of guesses:                     %4d" % results_all_coords[2][2], style="HEADER")
        logging("All Coordinates Attack Success probability:                    %3.2f" %results_all_coords[2][1], style="HEADER")
        logging("All Coordinates Attack Change in (log) Volume:                 %3.2f" %results_all_coords[1][1], style="HEADER")
        logging("All Coordinates Attack Baseline log det:                       %3.2f" %results_all_coords[1][0], style="HEADER")
        logging("All Coordinates Attack Intersected log det:                    %3.2f\n" %results_all_coords[1][2], style="HEADER")
        
        logging("All Coordinates Attack (known norm) with hints:                %3.2f bikz" % results_known_all_coords[2][0], style="HEADER")
        logging("All Coordinates Attack (known norm) with hints and guesses:    %3.2f bikz" % results_known_all_coords[0][0], style="HEADER")
        logging("All Coordinates Attack (known norm) Number of guesses:        %4d" % results_known_all_coords[2][2], style="HEADER")
        logging("All Coordinates Attack (known norm) Success probability:       %3.2f" %results_known_all_coords[2][1], style="HEADER")
        logging("All Coordinates Attack (known norm) Change in (log) Volume:    %3.2f" %results_known_all_coords[1][1], style="HEADER")
        logging("All Coordinates Attack (known norm) Baseline log det:          %3.2f" %results_known_all_coords[1][0], style="HEADER")
        logging("All Coordinates Attack (known norm) Intersected log det:       %3.2f\n" %results_known_all_coords[1][2], style="HEADER")

        logging("Conditional Attack with hints:                                 %3.2f bikz" % results_conditional[2][0], style="HEADER")
        logging("Conditional Attack with hints and guesses:                     %3.2f bikz" % results_conditional[0][0], style="HEADER")
        logging("Conditional Attack Number of guesses:                         %4d" % results_conditional[2][2], style="HEADER")
        logging("Conditional Attack Success probability:                        %3.2f" %results_conditional[2][1], style="HEADER")
        logging("Conditional Attack Change in (log) Volume:                     %3.2f" %results_conditional[1][1], style="HEADER")
        logging("Conditional Attack Baseline log det:                           %3.2f" %results_conditional[1][0], style="HEADER")
        logging("Conditional Attack Intersected log det:                        %3.2f\n" %results_conditional[1][2], style="HEADER")
        
        logging("Conditional Attack (known norm) with hints:                    %3.2f bikz" % results_known_conditional[2][0], style="HEADER")
        logging("Conditional Attack (known norm) with hints and guesses:        %3.2f bikz" % results_known_conditional[0][0], style="HEADER")
        logging("Conditional Attack (known norm) Number of guesses:            %4d" % results_known_conditional[2][2], style="HEADER")
        logging("Conditional Attack (known norm) Success probability:           %3.2f" %results_known_conditional[2][1], style="HEADER")
        logging("Conditional Attack (known norm) Change in (log) Volume:        %3.2f" %results_known_conditional[1][1], style="HEADER")
        logging("Conditional Attack (known norm) Baseline log det:              %3.2f" %results_known_conditional[1][0], style="HEADER")
        logging("Conditional Attack (known norm) Intersected log det:           %3.2f\n" %results_known_conditional[1][2], style="HEADER")
        
    
    print("Time:", datetime.datetime.now() - ttt)


    return 

run_experiment()
