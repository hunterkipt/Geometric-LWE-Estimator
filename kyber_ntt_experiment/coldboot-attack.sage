load("../framework/LWE.sage")
load("../framework/proba_utils.sage")
load("../framework/utils.sage")
load("../framework/AttackResults.sage")
load("../framework/DBDD.sage")

from sage.probability.probability_distribution import GeneralDiscreteDistribution
from pathlib import Path

def mkdir(path: str, clear=True) -> Path:
    p = Path(path)
    if clear and p.exists():
        rmtree(p)
    p.mkdir(parents=True, exist_ok=not clear)
    return p

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

mu_delta = 0
mu_delta_sq = 0

for i in range(2^12):
    cnt = (bin(i)[2:].zfill(13)).count('1')
    mu_delta += (i * (rho0^cnt) * ((1-rho0)^(12-cnt)))
    mu_delta_sq += (i^2 * (rho0^cnt) * ((1-rho0)^(12-cnt)))

S_delta = mu_delta_sq - (mu_delta ^ 2)

F = GF(q)

z = var('z')

P = PolynomialRing(F, x)

QP = P.quotient(x^128 + 1, 'z')

d = QP.degree()

k = 1

decay = GeneralDiscreteDistribution([1 - rho0, rho0])

# modeling: sample (W) = s * V + Delta (D)

V = gen_half_ntt_matrix()

ring_s = QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(d)])

ring_s_hat = vector(F, ring_s) * V

ring_delta = QP([F(sample_err(coeff, decay)) for coeff in ring_s_hat])

ring_w = QP(list(vector(ring_s_hat) - vector(ring_delta)))

mV_val = [list(v_elem) for v_elem in V]

mV = matrix(F, d, d, mV_val)
mD = matrix(F, 1, d, list(ring_delta))
mW = matrix(F, 1, d, list(ring_w))
mS = matrix(F, 1, d, list(ring_s))

# print("V", mV.nrows(), mV.ncols())
# print("D", mD.nrows(), mD.ncols())
# print("W", mW.nrows(), mW.ncols())
# print("S", mS.nrows(), mS.ncols())

assert((mS * mV) - mD == mW)

print("after the assert")
print("mean", mu_delta)
print("var", S_delta)

emb_V = -mV.change_ring(QQ).T
emb_D = mD.change_ring(QQ)
emb_W = mW.change_ring(QQ)
emb_S = mS.change_ring(QQ)

emb_W = emb_W.apply_map(recenter)

# perform embedding here
mu = vec([QQ(mu_delta)] * d + [QQ(0)] * d + [1])
mu = matrix(QQ, mu)

S = diagonal_matrix(QQ, [QQ(S_delta)] * d + [QQ(mu2)] * d + [0])

B = build_LWE_lattice(-emb_V, q) # primal
D = build_LWE_lattice(emb_V/q, 1/q) # dual

b_cen = emb_W.apply_map(recenter)

tar = concatenate([b_cen, [0] * d])
B = kannan_embedding(B, tar)
D = kannan_embedding(D, concatenate([-b_cen/q, [0] * d])).T

u = concatenate([emb_D, emb_S, [1]])

dbdd_inst = DBDD(
    B, S, mu, None, u, 
    verbosity=1, 
    D=D, 
    Bvol=d*log(q)
)

# lwe_inst = LWE(
#         n = d, 
#         q = q, 
#         m = d, 
#         D_e = None, 
#         D_s = None, 
#         verbosity=1,
#         A = emb_V,
#         b = emb_W,
#         Sigma_s = [QQ(mu2)] * d, 
#         Sigma_e = [QQ(sigma[k])] * d, 
#         mean_s = [0] * d,
#         mean_e = [mu[k]] * d, 
#         s = emb_S, 
#         e_vec = emb_D
# )

dbdd_inst.estimate_attack()

result = dbdd_inst.attack(beta_max=60)

save_results(result, "./out/results.pkl")
