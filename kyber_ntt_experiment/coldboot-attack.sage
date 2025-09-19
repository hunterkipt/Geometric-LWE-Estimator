load("../framework/LWE.sage")
load("../framework/utils.sage")
from sage.probability.probability_distribution import GeneralDiscreteDistribution

def bit_reverse_7(x):
    return int(bin(x)[2:].zfill(7)[::-1], 2)

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

def sample_err(coeff, decay_dist):
    offset = 0
    bits = bin(coeff)[2:].zfill(12)[::-1]
    for i, b in enumerate(bits):
        if b == '1':
            offset += 2 ^ i * decay_dist.get_random_element()
    return offset

mu2 = 2
rho0 = 0.01
q = 3329
samples = 3

# 1 mu (rho0) var (rho0(1-rho0))
# 2 mu (3 * rho0^2 + 3 * rho0(1-rho0) = 3*rho0) var (5 * rho0(1-rho0) + 9 * rho0^2)

mu = {1: rho0, 2: 3 * rho0}
sigma = {1: rho0 * (1 - rho0), 2: 5 * rho0 + 4 * rho0^2}

F = GF(q)

z = var('z')

P = PolynomialRing(F, x)

QP = P.quotient(x^128 + 1, 'z')

d = QP.degree()

k = 2

# modeling: s_twiddle (W) = delta * V_inv + s

V = gen_half_ntt_matrix()

V_inv = V.inverse()

ring_s = QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(d)])

ring_s_hat = vector(F, ring_s) * V

decay = GeneralDiscreteDistribution([1 - rho0, rho0])

# error sampled on the s coordinates
ring_delta = QP([F(sample_err(coeff, decay)) for coeff in ring_s_hat])

w_hat = vector(ring_s_hat) - vector(ring_delta)

ring_w = QP(list(w_hat * V_inv))

mV_val = []
mD_val = []
for v_elem, d_elem in zip(V_inv, ring_delta):
    for j in range(0, 12, k):
        pos = 12 - k - j
        mV_val += list((2 ^ pos) * v_elem)
        mD_val.append(int(d_elem >> pos) & int((1 << k) - 1))

mV = matrix(F, d * (12 // k), d, mV_val)
mD = matrix(F, 1, d * (12 // k), mD_val)
mW = matrix(F, 1, d, list(ring_w))
mS = matrix(F, 1, d, list(ring_s))

# print("V", mV.nrows(), mV.ncols())
# print("D", mD.nrows(), mD.ncols())
# print("W", mW.nrows(), mW.ncols())
# print("S", mS.nrows(), mS.ncols())

assert((mD * -mV) + mS == mW)

emb_V = -mV.change_ring(QQ)
emb_D = mD.change_ring(QQ)
emb_W = mW.change_ring(QQ)
emb_S = mS.change_ring(QQ)

emb_W = emb_W.apply_map(recenter)

lwe_inst = LWE(
        n = d * (12 // k), 
        q = q, 
        m = d, 
        D_e = None, 
        D_s = None, 
        verbosity=1,
        A = emb_V,
        b = emb_W,
        Sigma_s = [QQ(sigma[k])] * d * (12 // k), 
        Sigma_e = [QQ(mu2)] * d, 
        mean_s = [mu[k]] * d * (12 // k),
        mean_e = [0] * d, 
        s = emb_D, 
        e_vec = emb_S
)

dbdd = lwe_inst.embed_into_DBDD()

dbdd.estimate_attack()

dbdd.attack(randomize=True)

# a * s + e = b % q

# A * s + e = b % q <- Rot(a) here
# bkz on this??
