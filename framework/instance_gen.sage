from random import shuffle, randint

load("../framework/proba_utils.sage")
load("../framework/utils.sage")
load("../framework/DBDD_predict_diag.sage")
load("../framework/DBDD_predict.sage")
load("../framework/DBDD.sage")
load("../framework/EBDD.sage")
load("../framework/EBDD_non_homo.sage")
load("../framework/DBDD_optimized.sage")
load("../framework/ntru.sage")


def initialize_from_LWE_instance(dbdd_class, n, q, m, D_e,
                                 D_s, diag=False, verbosity=1,
                                 A=None, s=None, e_vec=None):
    """
    constructor that builds a DBDD instance from a LWE instance
    :n: (integer) size of the secret s
    :q: (integer) modulus
    :m: (integer) size of the error e
    :D_e: distribution of the error e (dictionnary form)
    :D_s: distribution of the secret s (dictionnary form)
    """
    if verbosity:
        logging("     Build DBDD from LWE     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d" % (n, m, q), style="VALUE")
    # define the mean and sigma of the instance
    mu_e, s_e = average_variance(D_e)
    mu_s, s_s = average_variance(D_s)
    mu = vec(m * [mu_e] + n * [mu_s] + [1])
    S = diagonal_matrix(m * [s_e] + n * [s_s] + [0])
    # draw matrix A and define the lattice
    if A is None:
        A = matrix([[randint(0, q) for _ in range(n)] for _ in range(m)])

    B = build_LWE_lattice(-A, q) # primal
    D = build_LWE_lattice(A/q, 1/q) # dual
    # draw the secrets
    if s is None:
        s = vec([draw_from_distribution(D_s) for _ in range(n)])

    if e_vec is None:
        e_vec = vec([draw_from_distribution(D_e) for _ in range(m)])

    # compute the public value t and build a target
    b = (s * A.T + e_vec) % q

    # A_cen = A.apply_map(recenter) #Make sure here A is centered.
    b_cen = b.apply_map(recenter)

    tar = concatenate([b_cen, [0] * n])
    B = kannan_embedding(B, tar)
    D = kannan_embedding(D, concatenate([-b_cen/q, [0] * n])).T
    u = concatenate([e_vec, s, [1]])
    return A, b, dbdd_class(B, S, mu, None, u, verbosity=verbosity, D=D, Bvol=m*log(q))



def initialize_kannan_ellipsoid_from_LWE(dbdd_class, n, q, m, D_e,
                                  D_s, homogeneous=True,
                                  verbosity=1, A=None, s=None, e_vec=None):
    """
    constructor that builds an EBDD instance from a LWE instance
    :n: (integer) size of the secret s
    :q: (integer) modulus
    :m: (integer) size of the error e
    :D_e: distribution of the error e (dictionnary form)
    :D_s: distribution of the secret s (dictionnary form)
    """
    if verbosity:
        logging("     Build EBDD from LWE     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d" % (n, m, q), style="VALUE")
    
    # Define mean and variance of instance
    _, s_e = average_variance(D_e)
    _, s_s = average_variance(D_s)
    # Draw centered A matrix
    if A is None:
        A = matrix([[randint(0, q - 1) for _ in range(n)] for _ in range(m)])
    
    A = A.apply_map(recenter)
    # Draw secrets
    if s is None:
        s = vec([draw_from_distribution(D_s) for _ in range(n)])

    if e_vec is None:
        e_vec = vec([draw_from_distribution(D_e) for _ in range(m)])
    # Compute public value and derived secret
    b = (s * A.T + e_vec) % q
    b_cen = b.apply_map(recenter)
    c = (s * A.T + e_vec - b_cen)/q

    # Compute Kannan ellipsoid embedding
    mu, S = kannan_ellipsoid(A, b_cen, q, s_s=s_s, s_e=s_e, homogeneous=homogeneous)
    
    if homogeneous:
        B = identity_matrix(n + m + 1)
        u = concatenate([s, c, [-1]])

    else:
        B = identity_matrix(n + m) 
        u = concatenate([s, c]) 

    return A, b_cen, dbdd_class(B, S, mu, u, verbosity=verbosity, ellip_scale=1)

def initialize_kannan_ellipsoid_from_LWE_sca(dbdd_class, n, q, m, D_e,
                                  D_s, homogeneous=True,
                                  verbosity=1, A=None, s=None, e_vec=None):
    """
    constructor that builds an EBDD instance from a LWE instance
    :n: (integer) size of the secret s
    :q: (integer) modulus
    :m: (integer) size of the error e
    :D_e: distribution of the error e (dictionnary form)
    :D_s: distribution of the secret s (dictionnary form)
    """
    if verbosity:
        logging("     Build EBDD from LWE     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d" % (n, m, q), style="VALUE")
    
    # Define mean and variance of instance
    _, s_e = average_variance(D_e)
    _, s_s = average_variance(D_s)
    # Draw centered A matrix
    if A is None:
        A = matrix([[randint(0, q - 1) for _ in range(n)] for _ in range(m)])
    
    A = A.apply_map(recenter)
    # Draw secrets
    if s is None:
        s = vec([draw_from_distribution(D_s) for _ in range(n)])

    if e_vec is None:
        e_vec = vec([draw_from_distribution(D_e) for _ in range(m)])
    # Compute public value and derived secret
    b = (s * A.T + e_vec) % q
    b_cen = b.apply_map(recenter)
    c = (s * A.T + e_vec - b_cen)/q #c on right hand side

    # Compute Kannan ellipsoid embedding
    mu, S, B_Sigma = kannan_ellipsoid_sca(A, b_cen, q, s_s=s_s, s_e=s_e, homogeneous=homogeneous)
    
    if homogeneous:
        B = identity_matrix(n + m + 1)
        u = concatenate([s, c, [-1]])

    else:
        B = identity_matrix(n + m) 
        u = concatenate([s, c]) 
    # print("ellipnorm of secret@initialize: ", scal((u - mu) * matrix(np_inv(S)) * ((u - mu)).T))
    # print("ellipnorm of secret@initialize: ", scal((u - mu) * S.inverse() * ((u - mu)).T))
    
    print("ellipnorm of secret@initialize: ",RR(scal((u - mu) * B_Sigma*B_Sigma.T/(n + m) * ((u - mu)).T)))
    # print("----------------------")
    # print("Bbbbbb@initial: ", matrix(RR,B_Sigma[n - 2 :n + 3,n - 2 :n + 3]), B_Sigma.rank())
    # print("uuuuuuu@initial", matrix(RR,u[0,n - 5 :n + 5]))
    # print("muuuuuu@initial", matrix(RR,mu[0, n - 5 :n + 5]))
    
    return A, b_cen, e_vec, dbdd_class(B, S, mu, u, verbosity=verbosity, ellip_scale=1), B_Sigma


def initialize_ellipsoid_from_LWE(dbdd_class, n, q, m, D_e,
                                  D_s, non_diag_sigma=False, full_sigma=False, homogeneous=True,
                                  verbosity=1, A=None, s=None, e_vec=None):
    """
    constructor that builds an EBDD instance from a LWE instance
    :n: (integer) size of the secret s
    :q: (integer) modulus
    :m: (integer) size of the error e
    :D_e: distribution of the error e (dictionnary form)
    :D_s: distribution of the secret s (dictionnary form)
    """
    if verbosity:
        logging("     Build EBDD from LWE     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d" % (n, m, q), style="VALUE")
    
    # Define mean and variance of instance
    mu_e, s_e = average_variance(D_e)
    mu_s, s_s = average_variance(D_s)
    # Draw centered A matrix
    if A is None:
        A = matrix([[randint(0, q - 1) for _ in range(n)] for _ in range(m)])
    
    A = A.apply_map(recenter)
    # Draw secrets
    if s is None:
        s = vec([draw_from_distribution(D_s) for _ in range(n)])

    if e_vec is None:
        e_vec = vec([draw_from_distribution(D_e) for _ in range(m)])
    # Compute public value and derived secret
    b = (s * A.T + e_vec) % q
    b_cen = b.apply_map(recenter)
    c = (s * A.T + e_vec - b_cen)/q

    # Compute covariance matrix
    S = ellipsoid_embedding(A, q, s_s, s_e, non_diag_sigma=non_diag_sigma,
                            full_sigma=full_sigma, homogeneous=homogeneous)
    
    if homogeneous:
        B = identity_matrix(n + m + 1)
        mu = vec(m * [mu_e] + n * [mu_s] + [0]) # Last coordinate is mean of b
        u = concatenate([s, c, [-1]])

    else:
        B = identity_matrix(n + m) 
        mu = vec(m * [mu_e] + n * [mu_s]) 
        u = concatenate([s, c]) 

    return A, b_cen, dbdd_class(B, S, mu, u, ellip_scale=(n+m+1), verbosity=verbosity)

def initialize_from_LWR_instance(dbdd_class, n, q, p, m, D_s, verbosity=1):
    if verbosity:
        logging("     Build DBDD from LWR     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d \t p=%d" % (n, m, q, p), style="VALUE")
    D_e = build_mod_switching_error_law(q, p)
    # draw matrix A and define the lattice
    A = matrix([[randint(0, q) for _ in range(n)] for _ in range(m)])
    s = vec([draw_from_distribution(D_s) for _ in range(n)])
    B = build_LWE_lattice(-A, q)
    b = q / p * ((p / q) * s * A.T).apply_map(lambda x: x.round(mode='down'))
    e = b - s * A.T
    tar = concatenate([b, [0] * n])
    B = kannan_embedding(B, tar)
    u = concatenate([e, s, [1]])
    # define the mean and sigma of the instance
    mu_e, s_e = average_variance(D_e)
    mu_s, s_s = average_variance(D_s)
    mu = vec(m * [mu_e] + n * [mu_s] + [1])
    S = diagonal_matrix(m * [s_e] + n * [s_s] + [0])
    return A, b, dbdd_class(B, S, mu, u, verbosity=verbosity)


def initialize_round5_instance(dbdd_class, n, q, p, h, m, verbosity=1):
    if verbosity:
        logging("     Build DBDD from round5     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d \t p=%d" % (n, m, q, p), style="VALUE")
    # draw matrix A and define the lattice
    assert (h % 2 == 0), "Round5 requires 2 to divide h"
    A = matrix([[randint(0, q) for _ in range(n)] for _ in range(m)])
    s = h / 2 * [1] + h / 2 * [-1] + (n - h) * [0]
    shuffle(s)
    s = vec(s)
    B = build_LWE_lattice(-A, q)
    b = vec([q / p * (round((p / q) * ((s * A.T)[0][i] % q)) % p)
             for i in range(n)])
    e = vec([((- s * A.T)[0][i] + b[0][i]) % q
             if ((- s * A.T)[0][i] + b[0][i]) % q < q / 2
             else ((- s * A.T)[0][i] + b[0][i]) % q - q
             for i in range(n)])
    tar = concatenate([b, [0] * n])
    B = kannan_embedding(B, tar)
    u = concatenate([e, s, [1]])
    # define the mean and sigma of the instance
    #raise NotImplementedError("Incorrect computation of the variance: Re-implementation needed. ")
    D_s = {-1: RR(h / 2 / n), 0: RR((n - h) / n), 1: RR(h / 2 / n)}
    D_e = build_uniform_law(q / p)
    mu_e, s_e = average_variance(D_e)
    mu_s, s_s = average_variance(D_s)
    mu = vec(m * [mu_e] + n * [mu_s] + [1])
    S = diagonal_matrix(m * [s_e] + n * [s_s] + [0])
    return A, b, dbdd_class(B, S, mu, u, verbosity=verbosity)


def initialize_LAC_instance(dbdd_class, n, q, m, verbosity=1):
    if verbosity:
        logging("     Build DBDD for LAC     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d" % (n, m, q), style="VALUE")
    # draw matrix A and define the lattice
    A = matrix([[randint(0, q) for _ in range(n)] for _ in range(m)])
    B = build_LWE_lattice(-A, q)
    assert (n % 4 == 0) and (m % 4 == 0), "LAC requires 4 to divide n and m"
    s = (n / 4) * [0, 1, 0, -1]
    shuffle(s)
    s = vec(s)
    e = (m / 4) * [0, 1, 0, -1]
    shuffle(e)
    e = vec(e)
    b = (s * A.T + e) % q
    tar = concatenate([b, [0] * n])
    B = kannan_embedding(B, tar)
    u = concatenate([e, s, [1]])
    # define the mean and sigma of the instance
    mu_e, s_e = 0, 1./2
    mu_s, s_s = 0, 1./2
    mu = vec(m * [mu_e] + n * [mu_s] + [1])
    S = diagonal_matrix(m * [s_e] + n * [s_s] + [0])
    return A, b, dbdd_class(B, S, mu, u, verbosity=verbosity)


def initialize_NTRU_instance(dbdd_class, n, q, Df, Dg, verbosity=1):
    if verbosity:
        logging("     Build DBDD from an NTRU instance (h=f/g [q])  ", style="HEADER")
        logging("n=%3d \t \t q=%d" % (n, q), style="VALUE")

    mu_f, s_f = QQ(0), QQ(2*Df/n)
    mu_g, s_g = QQ(0), QQ(2*Dg/n)
    mu = vec(n * [mu_f] + n * [mu_g])
    S = diagonal_matrix(n * [s_f] + n * [s_g])


    if dbdd_class not in [DBDD, DBDD_optimized]:
        return None, None, dbdd_class(None, S, mu, None, Bvol=n*log(q), homogeneous=True, verbosity=verbosity)

    ntru = NTRUEncrypt(n, q, Dg, Df)
    
    h,(f,g) = ntru.gen_keys()
    u = concatenate(vec(f), vec(g))
    H = matrix.circulant(h)
    B = build_LWE_lattice(H, q)

    return B, None, dbdd_class(B, S, mu, u, homogeneous=True, verbosity=verbosity, circulant=True)

