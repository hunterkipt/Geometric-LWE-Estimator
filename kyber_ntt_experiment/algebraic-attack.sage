load("../framework/LWE.sage")
load("../framework/utils.sage")

mu2 = 2
q = 3329

F = GF(q)

z = var('z')

P = PolynomialRing(F, x)

QP = P.quotient(x^192 + 4092, 'z')

ring_a = QP.random_element()
ring_s = QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(192)])
ring_e = QP([F(randint(0, mu2) - randint(0, mu2)) for _ in range(192)])

ring_b = ring_a * ring_s + ring_e

pub_mat = []

for i in range(192):
    pub_mat.append(list(ring_a * QP(f"x^{i}")))

mA = matrix(F, 192, 192, pub_mat).T

mS = matrix(F, 192, 1, list(ring_s))

mE = matrix(F, 192, 1, list(ring_e))

mB = matrix(F, 192, 1, list(ring_b))

assert(mS.T * mA.T + mE.T == mB.T)

emb_A = matrix(QQ, 192, 192, pub_mat)
emb_B = matrix(QQ, 1, 192, list(ring_b))
emb_S = matrix(QQ, 1, 192, list(ring_s))
emb_E = matrix(QQ, 1, 192, list(ring_e))


lwe_inst = LWE(
        n = 192, 
        q = q, 
        m = 192, 
        D_e = None, 
        D_s = None, 
        verbosity=1,
        A = emb_A,
        b = emb_B, 
        Sigma_s = [QQ(mu2)] * 192, 
        Sigma_e = [QQ(mu2)] * 192, 
        mean_s = [0] * 192,
        mean_e = [0] * 192, 
        s = matrix(QQ, list(ring_s)), 
        e_vec = matrix(QQ, list(ring_e))
)


dbdd = lwe_inst.embed_into_DBDD()

dbdd.estimate_attack()

dbdd.attack()

# a * s + e = b % q

# A * s + e = b % q <- Rot(a) here
# bkz on this??
