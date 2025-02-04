load("../framework/LWE.sage")
load("../framework/utils.sage")

mu2 = 2
q = 524287 # 3329

samples = 4

F = GF(q)

z = var('z')

P = PolynomialRing(F, x)

# QP = P.quotient(x^192 + 4092, 'z')
QP = P.quotient(x^128 + 524288 * x + 524285, 'z')


d = QP.degree()

ring_a = []

for i in range(samples):
    ring_a.append(QP.random_element())

ring_s = QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(d)])

ring_e = []

for i in range(samples):
    ring_e.append(QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(d)]))

ring_b = [ring_a[i] * ring_s + ring_e[i] for i in range(samples)]

pub_mat = []

for j in range(samples):
    temp = []
    for i in range(d):
        temp.append(list(ring_a[j] * QP(f"x^{i}")))
    temp = matrix(F, d, d, temp).T
    pub_mat.append(temp)

mA = block_matrix(F, samples, 1, pub_mat)

mS = matrix(F, d, 1, list(ring_s))

mE = matrix(F, d * samples, 1, list(ring_e[0]) + list(ring_e[1]) + list(ring_e[2]) + list(ring_e[3]))

mB = matrix(F, d * samples, 1, list(ring_b[0]) + list(ring_b[1]) + list(ring_b[2]) + list(ring_b[3]))

assert(mA * mS + mE == mB)

emb_A = mA.change_ring(QQ).T
emb_B = mB.change_ring(QQ).T
emb_S = mS.change_ring(QQ).T
emb_E = mE.change_ring(QQ).T

emb_B = emb_B.apply_map(recenter)

lwe_inst = LWE(
        n = d, 
        q = q, 
        m = d * samples, 
        D_e = None, 
        D_s = None, 
        verbosity=1,
        A = emb_A.T,
        b = emb_B, 
        Sigma_s = [QQ(mu2)] * d, 
        Sigma_e = [QQ(mu2)] * d * samples, 
        mean_s = [0] * d,
        mean_e = [0] * d * samples, 
        s = matrix(QQ, list(ring_s)), 
        e_vec = matrix(QQ, emb_E)
)


dbdd = lwe_inst.embed_into_DBDD()

dbdd.estimate_attack()

dbdd.attack()

# a * s + e = b % q

# A * s + e = b % q <- Rot(a) here
# bkz on this??
