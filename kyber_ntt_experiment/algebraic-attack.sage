load("../framework/LWE.sage")
load("../framework/utils.sage")

mu2 = 2
q = 524287 # 3329

F = GF(q)

z = var('z')

P = PolynomialRing(F, x)

# QP = P.quotient(x^192 + 4092, 'z')
QP = P.quotient(x^128 + 524288 * x + 524285, 'z')

ring_a = [
        QP.random_element(),
        QP.random_element(),
        QP.random_element(),
        QP.random_element()
        ]
ring_s = QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(128)])
ring_e = [
        QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(128)]),
        QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(128)]),
        QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(128)]),
        QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(128)])
        ]

ring_b = [ring_a[i] * ring_s + ring_e[i] for i in range(4)]

pub_mat = []

for j in range(4):
    temp = []
    for i in range(128):
        temp.append(list(ring_a[j] * QP(f"x^{i}")))
    temp = matrix(F, 128, 128, temp).T
    pub_mat.append(temp)

mA = block_matrix(F, 4, 1, pub_mat)

mS = matrix(F, 128, 1, list(ring_s))

mE = matrix(F, 128 * 4, 1, list(ring_e[0]) + list(ring_e[1]) + list(ring_e[2]) + list(ring_e[3]))

mB = matrix(F, 128 * 4, 1, list(ring_b[0]) + list(ring_b[1]) + list(ring_b[2]) + list(ring_b[3]))

assert(mA * mS + mE == mB)

emb_A = mA.change_ring(QQ).T
emb_B = mB.change_ring(QQ).T
emb_S = mS.change_ring(QQ).T
emb_E = mE.change_ring(QQ).T

emb_B = emb_B.apply_map(recenter)

lwe_inst = LWE(
        n = 128, 
        q = q, 
        m = 128 * 4, 
        D_e = None, 
        D_s = None, 
        verbosity=1,
        A = emb_A,
        b = emb_B, 
        Sigma_s = [QQ(mu2)] * 128, 
        Sigma_e = [QQ(mu2)] * 128 * 4, 
        mean_s = [0] * 128,
        mean_e = [0] * 128 * 4, 
        s = matrix(QQ, list(ring_s)), 
        e_vec = matrix(QQ, emb_E)
)


dbdd = lwe_inst.embed_into_DBDD()

dbdd.estimate_attack()

dbdd.attack()

# a * s + e = b % q

# A * s + e = b % q <- Rot(a) here
# bkz on this??
