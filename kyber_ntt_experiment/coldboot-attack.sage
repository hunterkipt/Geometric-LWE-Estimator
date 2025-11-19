load("../framework/LWE.sage")
load("../framework/proba_utils.sage")
load("../framework/utils.sage")
load("../framework/AttackResults.sage")
load("../framework/DBDD.sage")

from sage.probability.probability_distribution import GeneralDiscreteDistribution
from pathlib import Path
from functools import reduce

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

def decompose(ring_vector, word=1, factor=12):
    width = factor * word
    assert(width >= ceil(log(3329)/log(2)))
    bits = list(map(lambda x: bin(x)[2:].zfill(width), ring_vector))
    return [
        [
            int(elt[width - (k + 1) * word:width - k * word], 2)
            for k in range(factor)
        ]
        for elt in bits
    ]

mu2 = 2
rho0 = 0.01
q = 3329
samples = 3
num_width = 11
factor = ceil(ceil(log(q)/log(2))/num_width)

mu_delta = 0
S_delta = 0
for i in range(2^num_width):
    cnt = (bin(i)[2:].zfill(num_width)).count('1')
    mu_delta += (i * (rho0^cnt) * ((1-rho0)^(num_width-cnt)))
    S_delta += (i^2 * (rho0^cnt) * ((1-rho0)^(num_width-cnt)))
S_delta = S_delta - (mu_delta ^ 2)
decay = GeneralDiscreteDistribution([1 - rho0, rho0])

F = GF(q)
z = var('z')
P = PolynomialRing(F, x)
QP = P.quotient(x^128 + 1, 'z')
d = QP.degree()

# modeling: sample (W) = s * V + Delta (D)

V = gen_half_ntt_matrix()

ring_s = QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(d)])

ring_s_hat = vector(F, ring_s) * V

ring_delta = QP([F(sample_err(coeff, decay)) for coeff in ring_s_hat])

ring_w = QP(list(vector(ring_s_hat) - vector(ring_delta)))

# Read values
args = parse_args() # gives npz file path
if filepath in args:
    instance = AttackResults(args.filepath)
    ring_s = instance.get_secret()
    ring_s_hat = vector(F, ring_s) * V
    ring_delta = instance.get_error()
    ring_w = QP(list(vector(ring_s_hat) - vector(ring_delta)))
    short_vecs = instance.retrieve_shortvectors()
    num_width = 1

    dbdd_inst = DBDD(**args)

    # hint integration
    for v in short_vecs:
        sv_list = rotations(v):
        for sv in sv_list:
            dbdd_inst.integrate_short_vector_hint(matrix(QQ, matrix(F, sv)).apply_map(recenter))

    dbdd.estimate_attack()

    dbdd.attack()

    

mV_val = [list(v_elem) for v_elem in V]
for i in range(d):
    for j in range(num_width, factor * num_width, num_width):
        mV_val.append(list([0] * (i) + [2 ^ j] + [0] * (d - 1 - i)))

mS_val = list(ring_s)
mD_val = []
for row in decompose(ring_delta, word=num_width, factor=factor):
    mS_val.extend(list(map(lambda x: -x, row[1:])))
    mD_val.append(-1 * row[0])

mV = matrix(F, factor * d, d, mV_val)
mD = matrix(F, 1, d, mD_val)
mW = matrix(F, 1, d, list(ring_w))
mS = matrix(F, 1, factor * d, mS_val)
# print("V", mV.nrows(), mV.ncols())
# print("D", mD.nrows(), mD.ncols())
# print("W", mW.nrows(), mW.ncols())
# print("S", mS.nrows(), mS.ncols())

assert((mS * mV) + mD == mW)
# print("mean", mu_delta)
# print("var", S_delta)

emb_V = mV.change_ring(QQ).T
emb_D = mD.change_ring(QQ)
emb_W = mW.change_ring(QQ)
emb_S = mS.change_ring(QQ)

emb_W = emb_W.apply_map(recenter)

# perform embedding here
mu = vec([QQ(-mu_delta)] * d + [QQ(0)] * d + [QQ(-mu_delta)] * ((factor - 1) * d) + [1])
mu = matrix(QQ, mu)

S = diagonal_matrix(QQ, 
        [QQ(S_delta)] * d + 
        [QQ(mu2)] * d + 
        [QQ(S_delta)] * ((factor - 1) * d) + 
        [0])

B = build_LWE_lattice(-emb_V, q) # primal
D = build_LWE_lattice(emb_V/q, 1/q) # dual

b_cen = emb_W.apply_map(recenter)
tar = concatenate([b_cen, [0] * factor * d])
B = kannan_embedding(B, tar)
D = kannan_embedding(D, concatenate([-b_cen/q, [0] * factor * d])).T
u = concatenate([emb_D, emb_S, [1]])

dbdd_inst = DBDD(
    B, S, mu, None, u, 
    verbosity=1, 
    D=D, 
    Bvol=d*log(q)
)

dbdd_inst.estimate_attack()

# 1100 1100 1100
# [00000000001] [100 1100 1100](11)
# short vectors - 11 bits = up to 2048 [1] [11]
# [qI 0 0]
# [-V_n I_m 0]
# [e 0 1]
# vecs = instance.retrieve_shortvectors()
# vecs = [
#     [8] + [0] * 255 + [13] + [0] * 127,
#     [189] + [0] * 255 + [-109] + [0] * 127
# ]

# for i in range(11 * 127 + 10):
#     sv_list = []
#     for sv in vecs:
#         sv_list.append(
#             reduce(lambda x, y: x + y, [
#                 list(map(int, 
#                     (-1)^(v < 0) * bin(abs(v))[2:].zfill(11) 
#                          if i % 2 == 1 else 
#                     (-1)^(v < 0) * bin(abs(v))[-1]
#                 )) for i, v in enumerate(sv[:-128])
#             ]) + sv[-128:]
#         )

#     for j in range(2):
#         dbdd_inst.integrate_short_vector_hint(matrix(QQ, matrix(F, sv_list[j])).apply_map(recenter))
#         vecs[j] = [0] + vecs[j][:-1]

dbdd_inst.estimate_attack()

result = dbdd_inst.attack(beta_max=60)

save_results(result, "./out/results.pkl")
