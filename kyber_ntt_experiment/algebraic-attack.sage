load("../framework/LWE.sage")
load("../framework/utils.sage")

mu = 2
q = 3329

F = GF(q)

z = var('z')

P = PolynomialRing(F, x)

QP = P.quotient(x^192 + 4092, 'z')

a = QP.random_element()

s = QP([F(randint(0, mu) - randint(0, mu)) for _ in range(192)])

e = QP([F(randint(0, mu) - randint(0, mu)) for _ in range(192)])

b = a * s + e

mat = []

for i in range(192):
    mat.append(list(a * QP(f"x^{i}")))

A = matrix(F, 192, 192, mat).T

S = matrix(F, 192, 1, list(s))

E = matrix(F, 192, 1, list(e))

B = matrix(F, 192, 1, list(b))

assert(S.T * A.T + E.T == B.T)

emb_A = matrix(QQ, 192, 192, mat)
emb_B = matrix(QQ, 1, 192, list(b))
emb_S = matrix(QQ, 1, 192, list(s))
emb_E = matrix(QQ, 1, 192, list(e))


lwe_inst = LWE(
        n = 192, 
        q = q, 
        m = 192, 
        D_e = None, 
        D_s = None, 
        verbosity=1,
        A = emb_A,
        b = emb_B, 
        Sigma_s = [QQ(2)] * 192, 
        Sigma_e = [QQ(2)] * 192, 
        mean_s = [0] * 192,
        mean_e = [0] * 192, 
        s = list(s), 
        e_vec = list(e))


dbdd = lwe_inst.embed_into_DBDD()

dbdd.estimate_attack()

dbdd.attack()

# a * s + e = b % q

# A * s + e = b % q <- Rot(a) here
# bkz on this??
