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


def convert_mat_to_lwe(mat):
    F = GF(3329)
    top = mat[:64,:]
    bottom = mat[64:,:]

    rows = top.rows()

    odds = matrix(F, 32, 32, [rows[2 * i + 1] for i in range(32)])

    return block_matrix(F, 2, 1, [[odds], [bottom]])

def load_ntt_data(filename):

    F = GF(3329)

    data = np.load(filename, allow_pickle=True)

    skpv = data['skpv']

    bhat = data['bhat']

    ct_ntt = [F(i) for i in bhat[0]]

    U = gen_u_matrix(ct_ntt)

    U_64 = U.submatrix(0, 0, 64, 64)

    U_i64 = U_64.inverse().T

    U_E = U_i64.matrix_from_columns([2*i for i in range(32)])

    V = gen_full_ntt_matrix()

    V_half = gen_half_ntt_matrix()

    V_64 = V_half.T.submatrix(0, 0, 128, 32)

    block = block_matrix(F, 2, 1, [[-U_E], [V_64]])

    mat = block.T.rref().T # submatrix

    mat = convert_mat_to_lwe(mat)

    # (shat ** chat) * Ui - s * V = 0
    # (shat ** chat || s) * [ Ui // V ] = 0
    # (shat ** chat || s) * [ 1 // MAT ] = 0

    secret_ntt = [F(i) for i in skpv[0]]


    secret_ntt_even = [secret_ntt[2*i] for i in range(128)]

    secret_poly = list(list(matrix(F, 1, 256, secret_ntt) * V.T.inverse())[0])

    secret_even_poly = list(list(matrix(F, 1, 128, secret_ntt_even) * V_half.T.inverse())[0])


    # combined_secret = matrix(F, 1, 320, pairwise_mult(secret_ntt, ct_ntt)[:64] + secret_poly)

    # zzero = matrix(F, 1, 64, pairwise_mult(secret_ntt, ct_ntt)[:64]) + matrix(F, 1, 256, secret_poly) * mat # = vector of zeroes!!

    mul = pairwise_mult(secret_ntt, ct_ntt)[:64]

    mul_even = [mul[2 * i] for i in range(32)]

    mul_odd = [mul[2 * i + 1] for i in range(32)]

    matrix(F, 1, 32, mul_even) + matrix(F, 1, 160, mul_odd + secret_even_poly) * mat # all zeroes

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

    variances = variances[:64]
    means = means[:64]

    #means = [round((i * (2^16))) % 3329 for i in means]
    print(f'{means = }')

    variances_odd = [variances[2*i + 1] for i in range(32)]
    variances_even = [variances[2*i] for i in range(32)]

    means_even = [means[2*i] for i in range(32)]
    means_odd = [means[2*i + 1] for i in range(32)]

    secret_ciphertext_product = [QQ((i*169) % 3329) for i in pairwise_mult(secret_ntt, ct_ntt)[:64]]

    secret_s = [secret_ciphertext_product[2*i + 1] for i in range(32)] + secret_even_poly
    secret_e = [secret_ciphertext_product[2*i] for i in range(32)]

    #means_even = [secret_ciphertext_product[2*i] for i in range(32)]            # COMMENT THESE OUT
    #means_odd = [secret_ciphertext_product[2*i + 1] for i in range(32)]         # COMMENT THESE OUT

    D_s = build_centered_binomial_law(4)

    m_s, v_s = average_variance(D_s)

    variance_s = variances_odd + [v_s for i in range(128)]

    mean_s = means_odd + [m_s for i in range(128)]

    #print(pairwise_mult(secret_ntt, ct_ntt)[:64])
    print(f'{secret_ciphertext_product = }')
    print()
    print(variance_s)
    print(mean_s)
    print()
    print(variances_even)
    print(means_even)

    q = 3329

    print((matrix(QQ, secret_s) * matrix(QQ, mat) + matrix(QQ, secret_e)) % 3329)

    lwe = LWE(160, q, 32, None, None, 1, matrix(QQ, mat).T, matrix(QQ, [0 for i in range(32)]), Sigma_s = variance_s, Sigma_e = variances_even, mean_s = mean_s, mean_e = means_even, s = matrix(QQ, secret_s), e_vec = matrix(QQ, secret_e))

    ebdd = lwe.embed_into_EBDD()

    ebdd.estimate_attack()

    for i in range(32):
        prod_vec = [0] * 192
        prod_vec[i] = 1
        value = means_even[i]
        ebdd.integrate_perfect_hint(*ebdd.convert_hint_e_to_c(vec(prod_vec), value))

    ebdd.apply_perfect_hints()

    ebdd.estimate_attack()


    ebdd.attack()



if __name__ == "__main__":
    load_ntt_data("../kyber_project/results/results_exp_2_[(0,)]_1.2_1.npz")

#     F = GF(3329)
#     P = PolynomialRing(F, 'z')
#     z = var('z')
#     R = P.quotient(z^256 + 1)
#     NTT = gen_full_ntt_matrix()
#     NTT_inv = NTT.inverse()
#     vv = [F.random_element() for _ in range(256)]
#     ww = [F.random_element() for _ in range(256)]
# 
# 
#     U = gen_u_matrix(kyber_ntt(vv))
# 
# 
    # print(vv)
    # print((NTT_inv * matrix(F, 256, 1, kyber_ntt(vv))).T)

    # print(list(U * matrix(F, 256, 1, kyber_ntt(ww))))

    # print(pairwise_mult(kyber_ntt(ww), kyber_ntt(vv)))

    #print(pairwise_mult(kyber_ntt(vv), kyber_ntt(ww)))
    #print(kyber_ntt(list(R(vv) * R(ww))))


