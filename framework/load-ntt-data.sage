import numpy as np
from sage.rings.generic import ProductTree

load("../framework/LWE.sage")
load("../framework/utils.sage")

q = 3329

def bit_reverse_7(x):
    return int(bin(x)[2:].zfill(7)[::-1], 2)

def kyber_ntt(v):
    F = Zmod(3329)
    gen = F(17)

    z = var('z')
    P = PolynomialRing(F, 'z')
    v = P(v)
    tree = ProductTree([P([-gen^(2*bit_reverse_7(i)+1), 0, 1]) for i in range(128)])

    result = []
    for i in tree.remainders(v):
        coeffs = i.coefficients()
        result.append(coeffs[0])
        if len(coeffs) == 1:
            result.append(0)
        else:
            result.append(coeffs[1])
    return result


def pairwise_mult(v, w):
    F = Zmod(3329)
    gen = F(17)

    result = []
    # (a + bx)(c + dx) = ac + (ad + bc)x + bdx^2 
    for i in range(0,128):
        a, b = v[2*i], v[2*i+1]
        c, d = w[2*i], w[2*i+1]

        result.append(a*c + b*d*(gen^(2*bit_reverse_7(i)+1)))
        result.append(b*c+a*d)

    return result


def gen_u_matrix(u):
    F = Zmod(3329)
    gen = F(17)
    block_list = []
    for i in range(0, 128):
        c = u[2*i]
        d = u[2*i + 1]
        block = matrix(F, 2, 2, [[c, d * (gen^(2 * bit_reverse_7(i) + 1))], [d, c]])
        block_list.append(block)

    U = block_diagonal_matrix(block_list)
    return U

def gen_half_ntt_matrix():
    F = GF(3329)
    gen = F(17)
    M = [[None for _ in range(128)] for _ in range(128)]
    for i in range(128):
        for j in range(128):
            i_ = 2*bit_reverse_7(i) + 1
            M[i][j] = gen^(i_ * j)
    M = matrix(M)
    return M

def gen_full_ntt_matrix():
    half = gen_half_ntt_matrix()
    M = [[0 for _ in range(256)] for _ in range(256)]
    for i in range(128):
        for j in range(128):
            M[2*i][2*j] = half[i][j]
            M[2*i+1][2*j+1] = half[i][j]
    M = matrix(M)
    return M


# Get rid of rows which correspond to the `e` vector in a matrix to put it into a proper LWE form
def convert_mat_to_lwe(mat):
    F = GF(3329)
    top = mat[:256,:]
    bottom = 169 * mat[256:,:]  # Apply Montgomery form to s

    rows = top.rows()

    non_identity_rows = matrix(F, 128, 128, [rows[2 * i + 1] for i in range(128)]) # extract all non id rows

    return block_matrix(F, 2, 1, [[non_identity_rows], [bottom]]) # combine all the rows which don't correspond to an identity into one LWE matrix


def generate_ntt_instance(ciphertext_ntt, secret_ntt=None):
    F = GF(3329)
    q = 3329

    print("Starting generation of even and odd LWE matrices for the NTT repr.")

    ct_ntt = [F(i) for i in ciphertext_ntt[0]]

    U = gen_u_matrix(ct_ntt)

    # Inverse. In the first 64 case, this will be the inverse of U but with blocks only for the first 64:
    # [ a b ....... ]
    # [ c d ....... ]
    # [ ... a b ... ]
    # [ ... c d ... ]
    # ...
    U_i = U.pseudoinverse().T

    # The full matrix for each of the even and the odd parts of the secret polynomial
    # Looks like this:
    # [ X . . . . ]
    # [ X . . . . ]
    # [ . X . . . ]
    # [ . X . . . ]
    # [ . . X . . ]
    # [ . . X . . ]
    # ...
    # In the first 64 case, only the first 32 columns (64 rows will have values.
    U_E_full = U_i.matrix_from_columns([2*i for i in range(U_i.dimensions()[1] // 2)]) 
    U_O_full = U_i.matrix_from_columns([2*i + 1 for i in range(U_i.dimensions()[1] // 2)])


    # Pi_(UE) and Pi_(UO)
    # In the first 64 case, will be the 32 dim identity and 0 everywhere else for each one
    # [ 1  ..... ]
    # [ . 1 .... ]
    # [ ... 1 .. ]
    # ...
    # Note that these should be the same - either a U block is 0 or not.
    # Therefore, the projection at a single point is determined by that block, which is identical in both.

    proj_U_E = U_E_full.pseudoinverse() * U_E_full
    proj_U_O = U_O_full.pseudoinverse() * U_O_full

    # NTT matrix for the NTT transformation in the Kyber field.
    V = gen_full_ntt_matrix() 

    # Half NTT for evens / odds (due to the structure of the field)
    V_half = gen_half_ntt_matrix()

    # Projecting the NTT on the coordinates we have
    # In the first 64 case, we will be selecting the first 32 columns of V_half
    V_proj_E = V_half.T * proj_U_E.T
    V_proj_O = V_half.T * proj_U_O.T

    # Generate the block matrices that correspond with the instance.
    # Specifically, we know that 
    # (ntt(s * ct) || s_E) * [ [-U_E], [V_proj_E] ] = ntt(s * ct) * (-U_E) + s_E * V_proj_E
    # = -ntt(s_E) + ntt(s_E) = 0
    block_E = block_matrix(F, 2, 1, [[-U_E_full], [V_proj_E]])
    block_O = block_matrix(F, 2, 1, [[-U_O_full], [V_proj_O]])

    # Row reducing the matrix to get a set of 1s at the top -- this allows us to convert this into an LWE instance.
    # Here, we now get something like following:
    # [ 1 .......... ]
    # [ a .......... ]
    # [ . 1 ........ ]
    # [ . b ........ ]
    # [ . . (keep) . ]
    # [ . . (going). ]
    # [ ------------ ]
    # [ REDUCED MATR ]
    # [ REDUCED MATR ]
    # [ ............ ]
    mat_E = block_E.T.rref().T
    mat_O = block_O.T.rref().T

    # Extracts out the "1" and converts the bottom part to Montgomery form.
    # This gives us a matrix of the form 

    # [ a' .......... ]
    # [ . b' ........ ]
    # [ . . (keep) .. ]
    # [ . . (going).. ]
    # [ ------------- ]
    # [ RED MONT MATR ]
    # [ RED MONT MATR ]
    # [ RED MONT MATR ]

    # By extracting out the 1s, we can combine them to create an "e" term for LWE.
    # Therefore, we now have that ntt(s * ct)_E + [Mat * (ntt(s * ct)_O || s_E)] = 0
    # This is an LWE instance (heuristically, doesn't exactly meet assumptions).
    mat_E = convert_mat_to_lwe(mat_E)
    mat_O = convert_mat_to_lwe(mat_O)

    out_lwe_s_E = None
    out_lwe_e_E = None
    out_lwe_s_O = None
    out_lwe_e_O = None

    # Assert that the LWE instance functions properly.
    if secret_ntt is not None:

        print("Checking validity...")

        s_ntt = [F(i) for i in secret_ntt[0]]

        # Take the evens / odds of the secret.
        s_ntt_E = [s_ntt[2*i] for i in range(128)]
        s_ntt_O = [s_ntt[2*i + 1] for i in range(128)]


        # iNTT the Secret
        s_E = list(list(matrix(F, 1, 128, s_ntt_E) * V_half.T.inverse())[0])
        s_O = list(list(matrix(F, 1, 128, s_ntt_O) * V_half.T.inverse())[0])

        # Inverse montgomery the secret
        prod = [QQ((i*169) % 3329) for i in pairwise_mult(s_ntt, ct_ntt)]


        # Split the product into two parts -- this is because we extracted out the 'e' part into the error term.
        prod_part_e = [prod[2*i] for i in range(128)]
        prod_part_s = [prod[2*i + 1] for i in range(128)]

        # Check the output of the LWE computation.
        out_lwe_s_E = matrix(F, 1, 256, prod_part_s + s_E)
        out_lwe_e_E = matrix(F, 1, 128, prod_part_e)
        output_e = out_lwe_e_E + out_lwe_s_E * mat_E

        out_lwe_s_O = matrix(F, 1, 256, prod_part_s + s_O)
        out_lwe_e_O = matrix(F, 1, 128, prod_part_e)
        output_o = out_lwe_e_O + out_lwe_s_O * mat_O

        for i in output_e[0]:
            assert i == 0

        for j in output_o[0]:
            assert j == 0

        print("Valid!")

    print("Finished generation. ")

    return ((mat_E, out_lwe_s_E, out_lwe_e_E), (mat_O, out_lwe_s_O, out_lwe_e_O))

def embed_instance_into_dbdd(ciphertext_ntt, v_s_E, m_s_E, v_e_E, m_e_E, v_s_O, m_s_O, v_e_O, m_e_O, secret_ntt=None, column_slicing=False, row_slicing=False):

    print("Embedding into DBDD...")

    # Construct NTT -> LWE matrices
    (mat_E, out_lwe_s_E, out_lwe_e_E), (mat_O, out_lwe_s_O, out_lwe_e_O) = generate_ntt_instance(ciphertext_ntt, secret_ntt=secret_ntt)

    # Embed into LWE

    s_E = None
    e_E = None
    s_O = None
    e_O = None

    if (out_lwe_s_E is not None):
        s_E = matrix(QQ, out_lwe_s_E).apply_map(recenter)
    if (out_lwe_e_E is not None):
        e_E = matrix(QQ, out_lwe_e_E).apply_map(recenter)

    if (out_lwe_s_O is not None):
        s_O = matrix(QQ, out_lwe_s_O).apply_map(recenter)
    if (out_lwe_e_O is not None):
        e_O = matrix(QQ, out_lwe_e_O).apply_map(recenter)

    m_E = mat_E
    m_O = mat_O


    # In progress, DnD
    if column_slicing:
        slices = []

        for c in range(len(m_E.columns())):
            if m_E.columns()[c].count(0) == len(m_E.columns()[c]):
                slices.append(c)

        m_E = m_E.delete_columns(slices)
        e_E = e_E.delete_columns(slices)

        slices = []

        for c in range(len(m_O.columns())):
            if m_O.columns()[c].count(0) == len(m_O.columns()[c]):
                slices.append(c)

        m_O = m_O.delete_columns(slices)
        e_O = e_O.delete_columns(slices)

    # Also in progress :D
    if row_slicing:
        slices = []

        for c in range(len(m_E.rows())):
            if m_E.rows()[c].count(0) == len(m_E.rows()[c]):
                slices.append(c)

        m_E = m_E.delete_rows(slices)
        s_E = s_E.delete_columns(slices)

        slices = []

        for c in range(len(m_O.rows())):
            if m_O.rows()[c].count(0) == len(m_O.rows()[c]):
                slices.append(c)

        m_O = m_O.delete_rows(slices)
        s_O = s_O.delete_columns(slices)

    lwe_E = LWE(
        n=256, 
        q=q,
        m=128, 
        D_e=None, 
        D_s=None,
        verbosity=1, 
        A=matrix(QQ, m_E).T, 
        b=matrix(QQ, [0 for i in range(128)]),
        Sigma_s=v_s_E,
        Sigma_e=v_e_E,
        mean_s=m_s_E,
        mean_e=m_e_E,
        s=s_E,
        e_vec=e_E,
    )

    lwe_O = LWE(
        n=256, 
        q=q,
        m=128, 
        D_e=None, 
        D_s=None,
        verbosity=1, 
        A=matrix(QQ, m_O).T, 
        b=matrix(QQ, [0 for i in range(128)]),
        Sigma_s=v_s_O,
        Sigma_e=v_e_O,
        mean_s=m_s_O,
        mean_e=m_e_O,
        s=s_O,
        e_vec=e_O,
    )


    dbdd_E = lwe_E.embed_into_DBDD()
    dbdd_O = lwe_O.embed_into_DBDD()

    print("Finished!")

    # Return dbdd instances

    return (dbdd_E, dbdd_O)


def load_data(filename):

    print("Loading data file...")

    data = np.load(filename, allow_pickle=True)

    skpv = data['skpv']
    bhat = data['bhat']

    ntt_coeff_dist = data['ntt_coeff_dist'][()]
    means, variances = [], []

    sum_v = 0
    for i in range(256):  
        coeff, confidence = max(ntt_coeff_dist[i][0].items(), key=lambda x: x[1])
        coeff_mean = np.average(list(ntt_coeff_dist[i][0].keys()), weights=list(ntt_coeff_dist[i][0].values()))
        variance = 0
        for k, v in ntt_coeff_dist[i][0].items():
            sum_v += v
            variance += (k - coeff_mean)**2 * v
        means.append(coeff_mean)
        variances.append(variance)

    print("Loaded means and variances!")

    return (skpv, bhat, means, variances)


def conv_info(ciphertext_ntt, means_cs, variances_cs, secret_ntt=None):

    print("Converting mean/var dataa into proper representation...")

    # Split the variances into the error part and the part contained in the secret
    variances_cs_s = [variances_cs[2*i + 1] for i in range(128)]
    variances_cs_e = [variances_cs[2*i] for i in range(128)]

    # Split the means into the error part and the part contained in the secret
    means_cs_s = [means_cs[2*i + 1] for i in range(128)]
    means_cs_e = [means_cs[2*i] for i in range(128)]

    # Get variance and mean of actual secret vector
    D_s = build_centered_binomial_law(2)
    m_sec, v_sec = average_variance(D_s)

    # Calculate the variance and mean of the secret
    variance_s = variances_cs_s + [v_sec for i in range(128)]
    mean_s = means_cs_s + [m_sec for i in range(128)]

    # Set the variance and mean of error (just a formality -- its actually just the exact value)
    variance_e = variances_cs_e
    mean_e = means_cs_e

    # Cast everything to a rational
    mean_s = [round(i) for i in mean_s]
    mean_e = [round(i) for i in mean_e]

    variance_s = [QQ(i) if i > (1/100) else QQ(1/100) for i in variance_s]
    variance_e = [QQ(i) if i > (1/100) else QQ(1/100) for i in variance_e]

    # Full list of means / vars
    means_list = mean_e + mean_s
    variances_list = variance_e + variance_s

    print("Done!")

    dE, dO = embed_instance_into_dbdd(ciphertext_ntt, variance_s, mean_s, variance_e, mean_e, variance_s, mean_s, variance_e, mean_e, secret_ntt=secret_ntt)
    return (dE, dO, means_list, variances_list)


def simulation_test(filename, guessable):
    skpv, bhat, means, variances = load_data(filename)
    for i in range(64 - guessable):

        # Simulated variance, much more than normal
        D_s = build_centered_binomial_law(2)
        out_val = draw_from_distribution(D_s)
        mean, var = average_variance(D_s)
        means[i] = bhat[0][i] + out_val
        variances[i] = var

    return skpv, bhat, means, variances


def do_attack(filename):

    F = GF(3329)
    q = 3329

    # skpv, bhat, means, variances = load_data(filename)

    guesses = 35

    skpv, bhat, means, variances = simulation_test(filename, guesses)

    dbdd_E, dbdd_O, means_list, variances_list = conv_info(bhat, means, variances, secret_ntt=skpv)

    dbdd_E.estimate_attack()

    # print("Integrating zeroes ...")

    # for i in range(32, 128):
    #     prod_vec = [0] * (256 + 128)
    #     prod_vec[i] = 1
    #     dbdd_E.integrate_perfect_hint(vec(prod_vec), int(0))

    # for i in range(160, 256):
    #     prod_vec = [0] * (256 + 128)
    #     prod_vec[i] = 1
    #     dbdd_E.integrate_perfect_hint(vec(prod_vec), int(0))

    # print("Done!")

    print("Integrating full guesses ... ")

    for i in range(0, 32):
        if variances_list[i] > 1/10:
            continue

        prod_vec = [0] * (256 + 128)
        prod_vec[i] = 1
        value = means_list[i]

        dbdd_E.integrate_perfect_hint(vec(prod_vec), int(round(value)))


    for i in range(128, 160):
        if variances_list[i] > 1/10:
            continue

        prod_vec = [0] * (256 + 128)
        prod_vec[i] = 1
        value = means_list[i]

        dbdd_E.integrate_perfect_hint(vec(prod_vec), int(round(value)))

    print("Done!")

    print("Integrating pathological short vectors...")

    short_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    for i in range(30): 
        dbdd_E.integrate_short_vector_hint(matrix(QQ, matrix(F, short_vec)).apply_map(recenter))
        short_vec = short_vec[1:] + [0] 


    short_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(30): 
        dbdd_E.integrate_short_vector_hint(matrix(QQ, matrix(F, short_vec)).apply_map(recenter))
        short_vec = short_vec[1:] + [0] 

    print("Done!")

    dbdd_E.estimate_attack()

    dbdd_E.attack()

if __name__ == "__main__":
    do_attack("../ktrace-cca-data/results_exp_2_[(0,)]_0.8_3.npz")
