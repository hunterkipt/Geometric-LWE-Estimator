from numpy.random import choice as np_random_choice
from numpy.linalg import slogdet, qr, cholesky, eigvalsh
from numpy.linalg import inv as np_inv
from numpy import array, zeros
from math import pi, exp
from scipy.linalg import ldl

print_style = {'SUCCESS': '\x1b[1;37;42m',
               'FAILURE': '\x1b[1;37;41m',
               'REJECT': '\x1b[3;31m',
               'ACCEPT': '\x1b[3;32m',
               'VALUE': '\x1b[1;33m',
               'DATA': '\x1b[3;34m',
               'WARNING': '\x1b[6;30;43m',
               'ACTION': '\x1b[1;37m',
               'HEADER': '\x1b[4;37m',
               'NORMAL': '\x1b[0m',
               }


def logging(message, style='NORMAL', newline=True):
    if style is not None:
        print(print_style[style], end=' ')
    print(message, end=' ')
    if newline:
        print('\x1b[0m')
    else:
        print('\x1b[0m', end=' ')
    sys.stdout.flush()


def hint_to_string(v, l=None, lit="u", max_coeffs=5):
    s = ""
    count = 0
    for i in range(v.ncols()):
        x = v[0][i]
        if x == 0:
            continue
        count += 1
        if count > max_coeffs:
            s += "+ ... "
            break

        if x == 1:
            if count == 1:
                s += "%s%d " % (lit, i)
            else:
                s += "+ %s%d " % (lit, i)
        elif x == -1:
            s += "- %s%d " % (lit, i)
        elif x > 0 and count == 1:
            s += "%s*%s%d " % (str(x), lit, i)
        elif x > 0:
            s += "+ %s*%s%d " % (str(x), lit, i)
        else:
            s += "- %s*%s%d " % (str(-x), lit, i)

    if l is not None:
        s += "= %s" % str(l)

    return s


ROUNDING_FACTOR = 2**64

# Vectors will consistently be represented as 1*n matrices


def vec(x):
    # Is this a 2-dim object, with only one row
    try:
        x[0][0]
        try:
            x[1]
        except:
            return matrix(x)
        raise ValueError(
            " The object has more than one line: can't convert to vec.")
    except:
        # Then it should be a 1 dim object
        return matrix([x])


def mat(x):
    return matrix(x)


# Convert a 1*1 matrix into a scalar
def scal(M):
    assert M.nrows() == 1 and M.ncols() == 1, "This doesn't seem to be a scalar."
    return M[0, 0]


def average_variance(D):
    mu = 0.
    s = 0.

    for (v, p) in D.items():
        mu += v * p
        s += v * v * p

    s -= mu * mu
    return round_to_rational(mu), round_to_rational(s)


def canonical_vec(dimension, index):
    """Returns the vector [0,0 ... 0] with
    a 1 in a specific index
    :dimension: integer
    :index: integer
    """
    v = [0 for _ in range(dimension)]
    v[index] = 1
    return vec(v)


def is_canonical_direction(v):
    """ Test wether the vector has a cannonical direction, and returns
    its index and length if so, else None, None.
    """
    nz = [x != 0 for x in v]
    if sum(nz) != 1:
        return None

    i = nz.index(True)
    return i


def round_matrix_to_rational(M):
    A = matrix(ZZ, (ROUNDING_FACTOR * matrix(M)).apply_map(round))
    return matrix(QQ, A / ROUNDING_FACTOR)


def round_vector_to_rational(v):
    A = vec(ZZ, (ROUNDING_FACTOR * vec(v)).apply_map(round))
    return vec(QQ, A / ROUNDING_FACTOR)


def round_to_rational(x):
    A = ZZ(round(x * ROUNDING_FACTOR))
    return QQ(A) / QQ(ROUNDING_FACTOR)


def concatenate(L1, L2=None):
    """
    concatenate vecs
    """
    if L2 is None:
        return vec(sum([list(vec(x)[0]) for x in L1], []))

    return vec(list(vec(L1)[0]) + list(vec(L2)[0]))


def draw_from_distribution(D):
    """draw an element from the distribution D
    :D: distribution in a dictionnary form
    """
    X = np_random_choice([key for key in D.keys()],
                         1, replace=True,
                         p=[float(prob) for prob in D.values()])
    return X[0]


def GH_sv_factor_squared(k):
    return ((pi * k)**(1. / k) * k / (2. * pi * e))


def compute_delta(k):
    """Computes delta from the block size k. Interpolation from the following
    data table:
    Source : https://bitbucket.org/malb/lwe-estimator/
    src/9302d4204b4f4f8ceec521231c4ca62027596337/estima
    tor.py?at=master&fileviewer=file-view-default
    :k: integer
    estimator.py table:
    """

    small = {0: 1e20, 1: 1e20, 2: 1.021900, 3: 1.020807, 4: 1.019713, 5: 1.018620,
             6: 1.018128, 7: 1.017636, 8: 1.017144, 9: 1.016652, 10: 1.016160,
             11: 1.015898, 12: 1.015636, 13: 1.015374, 14: 1.015112, 15: 1.014850,
             16: 1.014720, 17: 1.014590, 18: 1.014460, 19: 1.014330, 20: 1.014200,
             21: 1.014044, 22: 1.013888, 23: 1.013732, 24: 1.013576, 25: 1.013420,
             26: 1.013383, 27: 1.013347, 28: 1.013310, 29: 1.013253, 30: 1.013197,
             31: 1.013140, 32: 1.013084, 33: 1.013027, 34: 1.012970, 35: 1.012914,
             36: 1.012857, 37: 1.012801, 38: 1.012744, 39: 1.012687, 40: 1.012631,
             41: 1.012574, 42: 1.012518, 43: 1.012461, 44: 1.012404, 45: 1.012348,
             46: 1.012291, 47: 1.012235, 48: 1.012178, 49: 1.012121, 50: 1.012065}

    if k != round(k):
        x = k - floor(k)
        d1 = compute_delta(floor(k))
        d2 = compute_delta(floor(k) + 1)
        return x * d2 + (1 - x) * d1

    k = int(k)
    if k < 50:
        return small[k]
    else:
        delta = GH_sv_factor_squared(k)**(1. / (2. * k - 2.))
        return delta.n()


def bkzgsa_gso_len(logvol, i, d, beta=None, delta=None):
    if delta is None:
        delta = compute_delta(beta)

    return RR(delta**(d - 1 - 2 * i) * exp(logvol / d))


rk = [0.789527997160000, 0.780003183804613, 0.750872218594458, 0.706520454592593, 0.696345241018901, 0.660533841808400, 0.626274718790505, 0.581480717333169, 0.553171463433503, 0.520811087419712, 0.487994338534253, 0.459541470573431, 0.414638319529319, 0.392811729940846, 0.339090376264829, 0.306561491936042, 0.276041187709516, 0.236698863270441, 0.196186341673080, 0.161214212092249, 0.110895134828114, 0.0678261623920553, 0.0272807162335610, -
      0.0234609979600137, -0.0320527224746912, -0.0940331032784437, -0.129109087817554, -0.176965384290173, -0.209405754915959, -0.265867993276493, -0.299031324494802, -0.349338597048432, -0.380428160303508, -0.427399405474537, -0.474944677694975, -0.530140672818150, -0.561625221138784, -0.612008793872032, -0.669011014635905, -0.713766731570930, -0.754041787011810, -0.808609696192079, -0.859933249032210, -0.884479963601658, -0.886666930030433]
simBKZ_c = [None] + [rk[-i] - sum(rk[-i:]) / i for i in range(1, 46)]

pruning_proba = .5
simBKZ_c += [RR(log(GH_sv_factor_squared(d)) / 2. -
                log(pruning_proba) / d) / log(2.) for d in range(46, 1000)]


def simBKZ(l, beta, tours=1, c=simBKZ_c):

    n = len(l)
    l2 = copy(vector(RR, l))

    for k in range(n - 1):
        d = min(beta, n - k)
        f = k + d
        logV = sum(l2[k:f])
        lma = logV / d + c[d]

        if lma >= l2[k]:
            continue

        diff = l2[k] - lma
        l2[k] -= diff
        for a in range(k + 1, f):
            l2[a] += diff / (f - k - 1)

    return l2


chisquared_table = {i: None for i in range(1000)}


for i in range(1025):
    chisquared_table[i] = RealDistribution('chisquared', i)


def conditional_chi_squared(d1, d2, lt, l2):
    """
    Probability that a gaussian sample (var=1) of dim d1+d2 has length at most
    lt knowing that the d2 first cordinates have length at most l2
    """
    D1 = chisquared_table[d1].cum_distribution_function
    D2 = chisquared_table[d2].cum_distribution_function
    l2 = RR(l2)

    PE2 = D2(l2)
    # In large dim, we can get underflow leading to NaN
    # When this happens, assume lifting is successfully (underestimating security)
    if PE2==0:
        raise ValueError("Numerical underflow in conditional_chi_squared")

    steps = 5 * (d1 + d2)

    # Numerical computation of the integral
    proba = 0.
    for i in range(steps)[::-1]:
        l2_min = i * l2 / steps
        l2_mid = (i + .5) * l2 / steps
        l2_max = (i + 1) * l2 / steps

        PC2 = (D2(l2_max) - D2(l2_min)) / PE2
        PE1 = D1(lt - l2_mid)

        proba += PC2 * PE1

    return proba


def compute_beta_delta(d, logvol, tours=1, interpolate=True, 
                       probabilistic=False, ignore_lift_proba=False, lift_union_bound=False,
                       number_targets=1, verbose=False):
    """
    Computes the beta value for given dimension and volumes
    It is assumed that the instance has been normalized and sphericized, 
    i.e. that the covariance matrices of the secret is the identity
    :d: integer
    :vol: float
    """
    bbeta = None
    pprev_margin = None

    # Keep increasing beta to be sure to catch the second intersection
    if not probabilistic:
        for beta in range(2, d):
            lhs = RR(sqrt(beta))
            rhs = bkzgsa_gso_len(logvol, d - beta, d, beta=beta)

            if lhs < rhs and bbeta is None:
                margin = rhs / lhs
                prev_margin = pprev_margin
                bbeta = beta

            if lhs > rhs:
                bbeta = None
            pprev_margin = rhs / lhs

        if bbeta is None:
            return 9999, 0

        ddelta = compute_delta(bbeta) * margin**(1. / d)
        if prev_margin is not None and interpolate:
            beta_low = log(margin) / (log(margin) - log(prev_margin))
        else:
            beta_low = 0
        assert beta_low >= 0
        assert beta_low <= 1
        return bbeta - beta_low, ddelta

    else:
        remaining_proba = 1.
        average_beta = 0.
        cumulated_proba = 0.

        delta = compute_delta(2)
        l = [log(bkzgsa_gso_len(logvol, i, d, delta=delta)) / log(2)
             for i in range(d)]
        for beta in [x/tours  for x in range(2*tours, d*tours)]:
            l = simBKZ(l, beta, 1)
            proba = 1.
            delta = compute_delta(beta)
            i = d - beta
            proba *= chisquared_table[beta].cum_distribution_function(
                2**(2 * l[i]))

            if not ignore_lift_proba:
                for j in range(2, int(d / beta + 1)):
                    i = d - j * (beta - 1) - 1
                    xt = 2**(2 * l[i])
                    if j > 1:
                        if not lift_union_bound:
                            x2 = 2**(2 * l[i + (beta - 1)])
                            d2 = d - i + (beta - 1)
                            proba *= conditional_chi_squared(beta - 1, d2, xt, x2)
                        else:
                            proba = min(proba, chisquared_table[d-i].cum_distribution_function(xt))

            for t in range(number_targets):
                average_beta += beta * remaining_proba * proba
                cumulated_proba += remaining_proba * proba
                remaining_proba *= 1. - proba

            if verbose:        
                print("β= %d,\t pr=%.4e, \t cum-pr=%.4e \t rem-pr=%.4e"%(beta, proba, cumulated_proba, remaining_proba))

            if remaining_proba < .001:
                average_beta += beta * remaining_proba
                break

        if remaining_proba > .01:
            raise ValueError("This instance may be unsolvable")

        # ddelta = compute_delta(average_beta)
        return average_beta, None


def block4(A, B, C, D):
    return block_matrix([[A, B], [C, D]])


def build_LWE_lattice(A, q):
    """Builds a n*m matrix of the form
    q*I, 0
    A^T, -I
    It corresponds to the LWE matrix
    :A: a m*n matrix
    :q: integer
    """

    (m, n) = A.nrows(), A.ncols()
    lambd = block4(q * identity_matrix(m),
                   zero_matrix(ZZ, m, n),
                   A.T,
                   identity_matrix(n)
                   )
    return lambd


def kannan_embedding(A, target, factor=1):
    """Creates a kannan embedding, i.e. it appends a line and
    a column as follows.
    A, 0
    target, uSVP_embedding_coeff
    :A: a matrix
    :target: a vector
    """
    d = A.nrows()
    lambd = block4(A,
                   zero_matrix(ZZ, d, 1),
                   mat(target),
                   mat([factor])
                   )

    return lambd


def recenter(elt):
    if elt > q / 2:
        return elt - q
    return elt

def get_distribution_from_table(table, multiplicative_factor):
    eta = len(table)
    D_s = {}
    support = set()
    for i in range(eta):
        D_s[i] = table[i] / multiplicative_factor
        D_s[-i] = D_s[i]
        support.add(i)
        support.add(-i)
    _, var = average_variance(D_s)
    assert(abs(sum([D_s[i] for i in support]) - 1) < 1e-5)
    return D_s

def ellipsoid_embedding(A, q, s_s, s_e, non_diag_sigma=False, full_sigma=False, homogeneous=True):
    """
    Construct initial embedding for ellipsoid2 embedding method
    """
    n = A.ncols()
    m = A.nrows()
    zero1 = zero_matrix(ZZ, n, m)
    zero2 = zero_matrix(ZZ, m, n)
    zero3 = zero_matrix(ZZ, m + n, 1)
    zero4 = zero_matrix(ZZ, 1, m + n)

    sigma_s = identity_matrix(n) * s_s
    sigma_1 = identity_matrix(1) * 1

    if non_diag_sigma:
        sigma_c = ((s_s)/(q*q) * A*A.T) + ((1/12) * identity_matrix(m))
    
    else:
        sigma_c = diagonal_matrix(((A * A.T)*s_s/(q*q) + s_e/(q*q)).diagonal())
        sigma_c_list = [scal(vec(A[i])*vec(A[i]).T)*s_s/q**2 + s_e/q**2 for i in range(m)]
        assert sigma_c == diagonal_matrix(sigma_c_list)


    if full_sigma:
        AA = (s_s/q)*A
        upleft_mat = block4(sigma_s*1, AA.T, AA, sigma_c*1)
    
    else:
        upleft_mat = block4(sigma_s*1, zero1, zero2, sigma_c*1)

    if homogeneous:
        return block4(upleft_mat, zero3, zero4, sigma_1)
    
    else:
        return upleft_mat

def kannan_ellipsoid(A, b, q, s_s=1, s_e=1, Sigma_s_e = None, mean_s=None, mean_e=None, homogeneous=True):
    """
    Sigma_s_e: covariance of [s || e]
    """
    n = A.ncols()
    m = A.nrows()
    
    zero1 = zero_matrix(ZZ, m, n)
    zero2 = zero_matrix(ZZ, m + n, 1)
    if mean_s is None:
        mean_s = vec([0] * n)
    if mean_e is None:
        mean_e = vec([0] * m)
    # Create Initial Transformation Matrix and Mean
    upleft_mat = block4(q*identity_matrix(m), zero1, -A.T, identity_matrix(n))
    b_new = b - mean_s * A.T - mean_e

    if not homogeneous:
        upleft_mat_inv = block4(1/q*identity_matrix(m), zero1, A.T/q, identity_matrix(n)) #upleft_mat.inverse()
        mu = concatenate(-b_new/q, mean_s)#b0 * upleft_mat_inv
        if not Sigma_s_e:
            Sigma_s_e = diagonal_matrix(QQ, m*[s_e] + n*[s_s]) 
        BB_inv = upleft_mat_inv.T * Sigma_s_e * upleft_mat_inv * (n + m)
        return mu, BB_inv

    b0 = concatenate(-b, zero_matrix(QQ, 1, n)) #double check if we can use b_new
    B = block4(upleft_mat, zero2, b0, identity_matrix(1))
    B_inv = B.inverse()
    return zero_matrix(QQ, 1, m+n+1), (B_inv.T * B_inv)*(n + m +1)

def kannan_ellipsoid_ldl(A, b, q, s_s=1, s_e=1, Sigma_s_e=None, mean_s=None, mean_e=None, homogeneous=True):
    """
    Sigma_s_e: covariance of [s || e]
    """
    n = A.ncols()
    m = A.nrows()
    
    zero1 = zero_matrix(ZZ, m, n)
    zero2 = zero_matrix(ZZ, m + n, 1)
    if mean_s is None:
        mean_s = vec([0] * n)
    if mean_e is None:
        mean_e = vec([0] * m)
    # Create Initial Transformation Matrix and Mean
    upleft_mat = block4(q*identity_matrix(m), zero1, -A.T, identity_matrix(n))
    b_new = b - mean_s * A.T - mean_e

    if not homogeneous:
        upleft_mat_inv = block4(1/q*identity_matrix(m), zero1, A.T/q, identity_matrix(n)) #upleft_mat.inverse()
        mu = concatenate(-b_new/q, mean_s)#b0 * upleft_mat_inv
        if not Sigma_s_e:
            Sigma_s_e = diagonal_matrix(QQ, n*[s_s] + m*[s_e])

        BB_inv = upleft_mat_inv.T * Sigma_s_e * upleft_mat_inv * (n + m)
        # BB_inv = 2*block4(BB_inv + mu.T*mu, mu.T, mu, matrix([1]))
        # mu = zero_matrix(1, n+m+1)

        lu, d, perm = ldl(BB_inv)
        return mu, BB_inv, matrix(lu[perm, :]), matrix(d)

    b0 = concatenate(-b, zero_matrix(QQ, 1, n)) #double check if we can use b_new
    B = block4(upleft_mat, zero2, b0, identity_matrix(1))
    B_inv = B.inverse()
    return zero_matrix(QQ, 1, m+n+1), (B_inv.T * B_inv)*(n + m +1)

def kannan_ellipsoid_mean_update(A, b, mean_s=None, mean_e=None):
    '''
    when the mean of s and e is nonzero
    output mean of s||c
    '''
    m = A.nrows()
    n = A.ncols()
    if mean_s is None:
        print("None in mean_s@kannan_ellipsoid_mean_update")
        mean_s = vec([0] * n)
    if mean_e is None:
        print("None in mean_e@kannan_ellipsoid_mean_update")
        mean_e = vec([0] * m)
    # print("1@mean_s_c_update")
    zero1 = zero_matrix(ZZ, m, n)
    # print("2@mean_s_c_update")
    # print("dimension match?", b.ncols(), A.nrows())
    b_new = b - mean_s * A.T - mean_e
    # print("3@mean_s_c_update")
    mu = concatenate(mean_s, -b_new/q)
    return mu
def round_to_rational_precision(x, rounding_digit):
    A = ZZ(round(x * rounding_digit))
    return QQ(A) / QQ(rounding_digit)
def round_matrix_to_rational_precision(M, rounding_digit):
    A = matrix(ZZ, (rounding_digit * matrix(M)).apply_map(round))
    return matrix(QQ, A / rounding_digit)
def round_vector_to_rational_precision(v, rounding_digit):
    A = vec(ZZ, (rounding_digit * vec(v)).apply_map(round))
    return vec(QQ, A / rounding_digit)

# def kannan_ellipsoid(A, b, q, s_s=1, s_e=1, homogeneous=True):
#     n = A.ncols()
#     m = A.nrows()
    
#     zero1 = zero_matrix(ZZ, m, n)
#     zero2 = zero_matrix(ZZ, m + n, 1)

#     # Create Initial Transformation Matrix and Mean
#     upleft_mat = block4(identity_matrix(n), -A.T, zero1, q*identity_matrix(m))
#     b0 = concatenate(zero_matrix(QQ, 1, n), -b)
    
#     scaling_mat = diagonal_matrix(QQ, n*[s_s] + m*[s_e])

#     if not homogeneous:
#         upleft_mat_inv = upleft_mat.inverse()
#         mu = b0 * upleft_mat_inv
#         BB_inv = upleft_mat_inv.T * scaling_mat * upleft_mat_inv * (n + m)
#         return mu, BB_inv

#     B = block4(upleft_mat, zero2, b0, identity_matrix(1))
#     B_inv = B.inverse()
#     return zero_matrix(QQ, 1, m+n+1), (B_inv.T * B_inv)*(n + m +1)


