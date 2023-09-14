

# This file was *autogenerated* from the file test-ntt-ellipsoid.sage
from sage.all_cmdline import *   # import sage library

_sage_const_4 = Integer(4); _sage_const_17 = Integer(17); _sage_const_13 = Integer(13); _sage_const_1 = Integer(1); _sage_const_0 = Integer(0); _sage_const_2 = Integer(2)
load("../framework/LWE.sage")
load("../framework/utils.sage")


n = _sage_const_4 
q = _sage_const_17 
mu = _sage_const_4 
# n = 256
# q = 3329
# mu = 4

F = GF(q)

gen = F(_sage_const_13 )
w = gen
# gen = F(17)
# w= = gen

z = var('z')
P = PolynomialRing(F, 'z')
tree = ProductTree([P([-gen**i, _sage_const_1 ]) for i in range(n)])

QP = P.quotient(tree.root(), 'x')

a = QP.random_element()
ahat = [i for i in tree.remainders(a.lift())]

s = QP([F(randint(_sage_const_0 , mu) - randint(_sage_const_0 , mu)) for _ in range(n)])
shat = [i for i in tree.remainders(s.lift())]

e = QP([F(randint(_sage_const_0 , mu) - randint(_sage_const_0 , mu)) for _ in range(n)])
ehat = [i for i in tree.remainders(e.lift())]

b = a * s + e
bhat = [(ahat[i] * shat[i]) + ehat[i] for i in range(n)]


assert [i for i in tree.remainders(b.lift())] == bhat

shat_m = matrix(QQ, _sage_const_1 , n, shat)
shat_m = shat_m.apply_map(recenter)
shat_mean = [QQ(shat_m[_sage_const_0 ,i]) for i in range(n)]

shat_variance = [_sage_const_2  for i in range(n)]

# Kyber: we don't have a 512th root of unity, so we need two NTT ellipsoids
# roots = [w^(2 * i + 1) for i in range(n)]
roots = [w**i for i in range(n)]


# NTT matrix for circular convolution (works with roots w^(2 * i + 1) for negacyclic
V = matrix(F, [[roots[i]**j for j in range(n)] for i in range(n)]).T # NTT matrix
# V = matrix(F, [[roots[i]^(2 * j + 1) for j in range(2 * n)] for i in range(n)]).T

nVinv = matrix(QQ, n, n, -_sage_const_1  * V.inverse())

# shat * nVinv + s = 0 mod q

D_s = build_centered_binomial_law(_sage_const_4 )

# b or 0
LWE_instance = LWE(n, q, n, D_s, D_s, _sage_const_1 , nVinv, matrix(QQ, [_sage_const_0  for i in range(n)]), Sigma_s=shat_variance, Sigma_e = None, mean_s = shat_mean, mean_e = None)


e_instance = LWE_instance.embed_into_EBDD()

integer_shat = matrix(QQ, _sage_const_1 , n, shat)
integer_shat = integer_shat.apply_map(recenter)

print("Integer shat: ", integer_shat)
hint1 = matrix(QQ, _sage_const_1 , _sage_const_2 *n, _sage_const_0 )
hint2 = matrix(QQ, _sage_const_1 , _sage_const_2 *n, _sage_const_0 ) 
#hint2 = np.zeros((1, 2*n))
hint1[_sage_const_0 ,n] = _sage_const_1 
hint2[_sage_const_0 ,n+_sage_const_1 ] = _sage_const_1 
print("Hint 1: ", hint1)
#hint2[0,1] = 1
out1 = integer_shat[_sage_const_0 , _sage_const_0 ]
out2 = integer_shat[_sage_const_0 , _sage_const_1 ]
#out2 = shat[1]
e_instance.integrate_perfect_hint(matrix(hint1), out1)
e_instance.integrate_perfect_hint(matrix(hint2), out2)

e_instance.apply_perfect_hints()

retvec = e_instance.attack()

print("Returned vector: ", retvec)
print("Original s: ", s)
print("Original s hat: ", shat)

# retvec = matrix(QQ, 1, 2*n, retvec[1][0,:-1])
# 

