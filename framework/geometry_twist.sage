from numpy.linalg import inv as np_inv
# from numpy.linalg import slogdet as np_slogdet
from numpy import array
import numpy as np

load("../framework/utils.sage")
load("../framework/geometry.sage")
# def kannan_ellipsoid_sca(A, b, q, s_s=1, s_e=1, homogeneous=True):
#     n = A.ncols()
#     m = A.nrows()
	
#     zero1 = zero_matrix(ZZ, m, n)
#     zero2 = zero_matrix(ZZ, m + n, 1)

#     upleft_mat = block4((1/sqrt(s_s))*identity_matrix(n), (-1/sqrt(s_e))*A.T, zero1, (q/sqrt(s_e))*identity_matrix(m))
#     b0 = concatenate(zero_matrix(QQ, 1, n), -b/sqrt(s_e))

#     if not homogeneous:
#         upleft_mat = array(upleft_mat, dtype=float)
#         b0 = array(b0, dtype=float)
#         mu = round_matrix_to_rational(b0 @ np_inv(upleft_mat))
#         BB_inv = np_inv(upleft_mat @ upleft_mat.T)
#         # print("Cholesky BB_inv")
#         # scipy.linalg.cholesky(BB_inv, lower=True)
#         Sigma = round_matrix_to_rational(BB_inv)*(n + m)
#         # print("Cholesky Sigma")
#         # scipy.linalg.cholesky(Sigma, lower=True)
#         return mu, Sigma, matrix(upleft_mat)

#     B = block4(upleft_mat, zero2, b0, identity_matrix(1))
#     BB = array(B*B.T, dtype=float)
#     return zero_matrix(QQ, 1, m+n+1), round_matrix_to_rational(np_inv(BB))*(n + m +1), B

def kannan_ellipsoid_sca(A, b, q, s_s=1, s_e=1, homogeneous=True):
	n = A.ncols()
	m = A.nrows()
	
	zero1 = zero_matrix(ZZ, m, n)
	zero2 = zero_matrix(ZZ, m + n, 1)
	
	# Create Initial Transformation Matrix and Mean
	upleft_mat = block4(identity_matrix(n), -A.T, zero1, q*identity_matrix(m))
	b0 = concatenate(zero_matrix(QQ, 1, n), -b)
	
	scaling_mat = diagonal_matrix(QQ, n*[s_s] + m*[s_e])
	###SCA: unscale for B###
	scalingB_mat = diagonal_matrix(QQ, n*[1/sqrt(s_s)] + m*[1/sqrt(s_e)])
	scaled_B = upleft_mat * scalingB_mat

	if not homogeneous:
		# upleft_mat_inv = upleft_mat.inverse()
		upleft_mat_inv = block4(identity_matrix(n), A.T/q, zero1, 1/q*identity_matrix(m))
		mu = concatenate(zero_matrix(QQ, 1, n), -b/q)#b0 * upleft_mat_inv
		BB_inv = upleft_mat_inv.T * scaling_mat * upleft_mat_inv * (n + m)#scaled
		return mu, BB_inv, scaled_B

	B = block4(upleft_mat, zero2, b0, identity_matrix(1))
	B_inv = B.inverse()
	return zero_matrix(QQ, 1, m+n+1), (B_inv.T * B_inv)*(n + m +1), B #temp

def kannan_ellipsoid_sca_general_cov_mean(A, b, q, Sigma_s_e, mean_s=None, mean_e=None, homogeneous=True):
	n_curr = A.ncols()
	m_curr = A.nrows()
	print("dim n_curr, m_curr", n_curr, m_curr)
	
	zero1 = zero_matrix(ZZ, m_curr, n_curr)
	zero2 = zero_matrix(ZZ, m_curr + n_curr, 1)

	if mean_s is None:
		mean_s = vec([0] * n_curr)
	if mean_e is None:
		mean_e = vec([0] * m_curr)

	# Create Initial Transformation Matrix and Mean
	upleft_mat = block4(identity_matrix(n_curr), -A.T, zero1, q*identity_matrix(m_curr)) ##
	# print(1)
	b_new = b - mean_s * A.T - mean_e
	# print(2)
	b0 = concatenate(zero_matrix(QQ, 1, n), -b_new)
	# print(3)
	Sigma_s_e_inv = matrix(np_inv(Sigma_s_e))
	# print("dim Sigma_s_e", Sigma_s_e.nrows(), Sigma_s_e.ncols())

	sqrt_Sigma_s_e_inv = matrix(scipy.linalg.cholesky(Sigma_s_e_inv))

	# print("----------------------")
	# print("sqrt_Sigma_s_e_inv@reinitial: ", matrix(RR,round_matrix_to_rational_precision(sqrt_Sigma_s_e_inv, 10*8)[n_curr - 2 :n_curr + 3,n_curr - 2 :n_curr + 3]), sqrt_Sigma_s_e_inv.rank())
	# print(sqrt_Sigma_s_e_inv.nrows(), sqrt_Sigma_s_e_inv.ncols())
	
	# # diagonal_matrix(QQ, n*[1/sqrt(s_s)] + m*[1/sqrt(s_e)])
	scaled_B = upleft_mat * sqrt_Sigma_s_e_inv
	# print("----------------------")
	# print("scaled_B: ", matrix(RR,round_matrix_to_rational_precision(scaled_B, 10*8)[n_curr - 2 :n_curr + 3,n_curr - 2 :n_curr + 3]), scaled_B.rank(), scaled_B.nrows(), scaled_B.ncols())
	
	#scaled_B = upleft_mat
	if not homogeneous:
		upleft_mat_inv = block4(identity_matrix(n_curr), A.T/q, zero1, 1/q*identity_matrix(m_curr)) #upleft_mat.inverse()
		mu = concatenate(mean_s, -b_new/q)#b0 * upleft_mat_inv
		BB_inv = upleft_mat_inv.T * Sigma_s_e * upleft_mat_inv * (n_curr + m_curr) #<=1
		# print("dim BB_inv", BB_inv.nrows(), BB_inv.ncols())
		tempB = scaled_B * scaled_B.T 
		tempS = upleft_mat * Sigma_s_e.inverse() * upleft_mat.T
		tempSS = tempB.inverse() * (n_curr + m_curr) 
		########
		# print("----------------------")
		# print("tempB: ", matrix(RR,round_matrix_to_rational_precision(tempB, 10*8)[n_curr - 2 :n_curr + 3,n_curr - 2 :n_curr + 3]), tempB.rank())
		# print("tempS: ", matrix(RR,round_matrix_to_rational_precision(tempS, 10*8)[n_curr - 2 :n_curr + 3,n_curr - 2 :n_curr + 3]), tempS.rank())
		# print("BB_inv: ", matrix(RR,round_matrix_to_rational_precision(BB_inv, 10*8)[n_curr - 2 :n_curr + 3,n_curr - 2 :n_curr + 3]), BB_inv.rank())
		# print("tempSS: ", matrix(RR,round_matrix_to_rational_precision(tempSS, 10*8)[n_curr - 2 :n_curr + 3,n_curr - 2 :n_curr + 3]), tempSS.rank())
		#########
		return mu, BB_inv, scaled_B

	B = block4(upleft_mat, zero2, b0, identity_matrix(1))
	B_inv = B.inverse()
	return zero_matrix(QQ, 1, m_curr+n_curr+1), (B_inv.T * B_inv)*(n_curr + m_curr +1), B #temp


# def mean_s_e_update(A, b, q, new_mu_s, new_mu_e):
# 	m = A.nrows()
# 	n = A.ncols()
# 	zero1 = zero_matrix(ZZ, m, n)

#     upleft_mat = block4(identity_matrix(n), -A.T, zero1, q*identity_matrix(m))
#     b_new = b - new_mu_s * A.T - new_mu_e
#     b0 = concatenate(zero_matrix(QQ, 1, n), -b_new)
#     upleft_mat_inv = upleft_mat.inverse()
#     mu_update = b0 * upleft_mat_inv
#     return mu_update

def find_ellipsoid_intersection_sca(mu1, W1, mu2, W2, tolerance=1.48e-08):
	"""
	Ellipsoid intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
	Page 14. Note that for numerical stability, we will minimize the log polynomial instead of
	finding a root. This function only works between two full rank ellipsoids of equal dimension.
	"""
	# Set up q1, q2, W1, W2
	q1 = array(mu1, dtype=float)
	q2 = array(mu2, dtype=float)

	# W1 = np_inv(array(Sigma1, dtype=float))
	# W2 = np_inv(array(Sigma2, dtype=float))
	W1 = array(W1, dtype=float)
	W2 = array(W2, dtype=float)

	# Set up commonly used calculations for speed up
	diff_q = q2 - q1
	diff_W = W2 - W1
	W1q1 = W1 @ q1.T # numpy dot product operator
	W2q2 = W2 @ q2.T
	n = W1.shape[0]

	# Define log_polynomial function
	def log_poly(pi):
		X = pi*W1 + (1 - pi)*W2
		X_inv = np_inv(X)
		q_plus = X_inv @ (pi*W1q1 + (1 - pi)*W2q2)
		alpha = 1 - pi*(q1 @ W1 @ q1.T).item() - (1 - pi)*(q2 @ W2 @ q2.T).item() + (q_plus.T @ X @ q_plus).item()

		_, X_det = slogdet(X)
		poly = alpha * trace(X_inv @ (-diff_W)) - n*(2*(q_plus.T @ (W1q1 - W2q2)).item() + (q_plus.T @ diff_W @ q_plus).item() - (q1 @ W1 @ q1.T).item() + (q2 @ W2 @ q2.T).item())

		return 2*X_det + log(poly)

	# Solve for the solution
	_, pi_sol = find_local_minimum(log_poly, 0, 1, tol=tolerance)

	return round_to_rational(pi_sol)

def ellipsoid_intersection_sca(mu1, W1, mu2, W2, tolerance=1.48e-08):
	"""
	Ellipsoid intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
	Page 14. Note that for numerical stability, we will minimize the  log polynomial instead of
	finding a root. This function only works between two full rank ellipsoids of equal dimension.
	"""
	# Find pi_sol:
	pi_sol = find_ellipsoid_intersection_sca(mu1, W1, mu2, W2, tolerance=tolerance)
	print(f"(Solution) pi = {pi_sol.n()}")
	# Calculate final ellipsoid
	# W1 = Sigma1.inverse()
	# W2 = Sigma2.inverse()
	X = pi_sol*W1 + (1 - pi_sol)*W2
	# print("condition")
	# print(matrix(RDF, X).condition(p=2))
	# print("eigenvalues: ", X.eigenvalues())
  
	X_inv = X.inverse()

	# Compute new mean. Use it to compute alpha scaling factor
	new_mu = (X_inv*(pi_sol*W1*mu1.T + (1 - pi_sol)*W2*mu2.T)).T
	
	alpha = 1 - pi_sol*scal(mu1 * W1 * mu1.T) - (1 - pi_sol)*scal(mu2 * W2 * mu2.T) + scal(new_mu * X * new_mu.T)

	print(f"alpha = {alpha.n()}")
	# print(f"X*X_inv MSE = {(((X*X_inv) - identity_matrix(X.ncols())).norm()**2)/(X.ncols() * X.ncols())}")
	if alpha < 0:
		raise ValueError("Scaling factor alpha < 0!")
	
	new_Sigma = alpha * X_inv

	return round_matrix_to_rational(new_mu), round_matrix_to_rational(new_Sigma)

def intersect_indices_update(guess_indices, intersect_indices, n):
	#n: prev dimension
	indice_map = dict()
	ctr = 0
	for i in range(n):
		if i in guess_indices:
			continue
		else:
			indice_map[i] = ctr
			ctr += 1
	# print(indice_map)
	intersect_update = []
	for j in intersect_indices:
		if j in indice_map:
			intersect_update += [indice_map[j]]
	# print(intersect_update)
	return intersect_update
def value_list_update(guess_indices, val_list):
	list_update = []
	# print(type(val_list))
	# print(type(val_list[0]))
	# print("curr testing...",len(val_list), type(len(val_list)))
	for ind in range(len(val_list)):
		if ind not in guess_indices:
			list_update += [val_list[ind]]
	# print("list_update: ", list_update)
	return list_update
def A_s_update(guess_indices, A, u):
	# print("A_s_update1")
	n = A.ncols()
	# print("A_s_update2 ", guess_indices)
	A_update = A.delete_columns(guess_indices)
	# print("A_s_update3")
	s_update = u[:, :n].delete_columns(guess_indices)
	# print("A_s_update4Sigma_unscaled@reinitial:")
	return A_update, s_update
def cov_update(guess_indices, new_Sigma):
	Sigma_temp = new_Sigma.delete_columns(guess_indices)
	Sigma_update = Sigma_temp.delete_rows(guess_indices)
	print("updated Sigma dimension: col, row: ", Sigma_update.ncols(), Sigma_update.nrows())
	return Sigma_update


def mean_s_c_update(A, b, m, n, mean_s=None, mean_e=None):
	'''
	when the mean of s and e is nonzero
	output mean of s||c
	'''
	if mean_s is None:
		print("None in mean_s@mean_s_c_update")
		mean_s = vec([0] * n)
	if mean_e is None:
		print("None in mean_e@mean_s_c_update")
		mean_e = vec([0] * m)
	# print("1@mean_s_c_update")
	zero1 = zero_matrix(ZZ, m, n)
	# print("2@mean_s_c_update")
	# print("dimension match?", b.ncols(), A.nrows())
	b_new = b - mean_s * A.T - mean_e
	# print("3@mean_s_c_update")
	mu = concatenate(mean_s, -b_new/q)
	return mu
def round_to_rational_precision(x, rounding_digit):
    A = ZZ(round(x * rounding_digit))
    return QQ(A) / QQ(rounding_digit)
def round_matrix_to_rational_precision(M, rounding_digit):
    A = matrix(ZZ, (rounding_digit * matrix(M)).apply_map(round))
    return matrix(QQ, A / rounding_digit)
def round_vector_to_rational_precision(v, rounding_digit):
    A = vec(ZZ, (rounding_digit * vec(v)).apply_map(round))
    return vec(QQ, A / rounding_digit)






def reinitialize_kannan_ellipsoid_from_LWE_sca(dbdd_class, A_prev, u_prev, e_vec, q, guess_indices, 
												new_mu_s, 
												new_mu_e, 
												Sigma_unscaled,
												homogeneous=False,
												verbosity=1):
	"""
	constructor that builds an EBDD instance from a LWE instance
	:n: (integer) size of the secret s
	:q: (integer) modulus
	:m: (integer) size of the error e
	:D_e: distribution of the error e (dictionnary form)
	:D_s: distribution of the secret s (dictionnary form)
	"""
	logging("     Build chopped EBDD from previous LWE     ", style="HEADER")
		# logging("n=%3d \t m=%3d \t q=%d" % (n, m, q), style="VALUE")
	n = A_prev.ncols()
	m = A_prev.nrows()
	
	num_del = len(guess_indices)
	new_n = n - num_del
	rounding_digit = 10**7 #10*15

	# Update A, s
	new_A, new_s = A_s_update(guess_indices, A_prev, u_prev)

	#Update Sigma_s_e
	
	print("----------------------")
	print("Sigma_unscaled@reinitial: ", matrix(RR,round_matrix_to_rational_precision(Sigma_unscaled, rounding_digit)[n - 2 :n + 3,n - 2 :n + 3]), Sigma_unscaled.rank())
	# print("Sigma_unscaled@reinitial: ", matrix(RR,round_matrix_to_rational_precision(Sigma_unscaled, rounding_digit)), Sigma_unscaled.rank())
	zero1 = zero_matrix(ZZ, m, n)
	upleft_mat = block4(identity_matrix(n), -A.T, zero1, q*identity_matrix(m)) #
	Sigma_s_e_p = upleft_mat.T * Sigma_unscaled * upleft_mat # Square norm (s||e) <=dim
	print("----------------------")
	print("Sigma_s_e_p@reinitial: ", matrix(RR,Sigma_s_e_p)[n - 2 :n + 3,n - 2 :n + 3], Sigma_s_e_p.rank())

	Sigma_s_e = round_matrix_to_rational_precision(Sigma_s_e_p, rounding_digit)
	print("----------------------")
	print("Sigma_s_e@reinitial: ", matrix(RR,Sigma_s_e[n - 2 :n + 3,n - 2 :n + 3]), Sigma_s_e.rank())
	# print("Sigma_s_e@reinitial: ", matrix(RR,Sigma_s_e), Sigma_s_e.rank())
	
	new_Sigma_s_e = cov_update(guess_indices, Sigma_s_e)
	print("----------------------")
	print("new_Sigma_s_e@reinitial: ", matrix(RR,new_Sigma_s_e[new_n - 2 :new_n + 3,new_n - 2 :new_n + 3]), new_Sigma_s_e.rank())
	# print("new_Sigma_s_e@reinitial: ", matrix(RR,new_Sigma_s_e), new_Sigma_s_e.rank())
	
	
	c_vec = u_prev[:, n:]
	new_b = (new_s * new_A.T + e_vec) - c_vec * identity_matrix(m) * q #(new_s * new_A.T + e_vec) % q
	# new_b_cen = new_b.apply_map(recenter)
	# new_c = (new_s * new_A.T + e_vec - new_b_cen)/q #c on right hand side

	update_mu_s = new_mu_s.delete_columns(guess_indices)
	# Compute Kannan ellipsoid embedding
	new_mu, new_S, new_B_Sigma = kannan_ellipsoid_sca_general_cov_mean(new_A, new_b, q, new_Sigma_s_e, update_mu_s, new_mu_e, homogeneous=homogeneous)
	print("dimension of new_S@reinitialize_kannan_ellipsoid_from_LWE_sca: ", new_S.nrows(), new_S.ncols())
	# new_n = new_A.ncols()
	new_m = new_A.nrows()
	print("new n, m", new_n, new_m)
	if homogeneous:
		B = identity_matrix(new_n + new_m + 1)
		new_u = concatenate([new_s, c_vec, [-1]])

	else:
		B = identity_matrix(new_n + new_m) 
		new_u = concatenate([new_s, c_vec]) 
	
	print("ellipnorm of secret@reinitialize: ", scal((new_u - new_mu) * matrix(np_inv(new_S)) * ((new_u - new_mu)).T))
	# print("ellipnorm of secret@reinitialize: ", scal((new_u - new_mu) * new_S.inverse() * ((new_u - new_mu)).T))
	
	# print("ellipnorm of secret@reinitialize: ", RR(scal((new_u - new_mu) * new_B_Sigma * new_B_Sigma.T/(new_n + m) * ((new_u - new_mu)).T)))
	# print("----------------------")
	# print("Bbbbbb@reinitial: ", matrix(RR,new_B_Sigma[new_n - 2 :new_n + 3,new_n - 2 :new_n + 3]), new_B_Sigma.rank())
	# # print("Bbbbbb@reinitial: ", matrix(RR,new_B_Sigma), new_B_Sigma.rank())
	
	# print("uuuuuuu@reinitial", matrix(RR,new_u[0, new_n - 5 :new_n + 5]))
	# print("muuuuuu@reinitial", matrix(RR,new_mu[0, new_n - 5 :new_n + 5]))
	
	return new_A, new_b, e_vec, dbdd_class(B, new_S, new_mu, new_u, verbosity=verbosity, scaled=True), new_B_Sigma

def switch_to_kannan_embedding(dbdd_class, A, b, e, q, mu_s, mu_e, ebdd, verbosity=1):
	n = A.ncols()
	m = A.nrows()
	print("n, m", n, m)
	print(mu_s.ncols())
	s = ebdd.u[:, :n]

	B = build_LWE_lattice(-A, q) # primal
	tar = concatenate([b, [0] * n])
	B = kannan_embedding(B, tar)
	print("B dim:", B.nrows(), B.ncols())

	D = build_LWE_lattice(A/q, 1/q)
	D = kannan_embedding(D, concatenate([-b/q, [0] * n])).T
	print("D dim:", D.nrows(), D.ncols())

	zero1 = zero_matrix(ZZ, m, n)
	upleft_mat = block4(identity_matrix(n), -A.T, zero1, q*identity_matrix(m)) #
	print("suppose false@switch_to_kannan_embedding: ", ebdd.scaled)
	assert(ebdd.scaled == False)
	S_s_e_p = upleft_mat.T * ebdd.S * upleft_mat
	print("----------------------")
	print("S_s_e_p@switch: ", matrix(RR,S_s_e_p)[n - 2 :n + 3,n - 2 :n + 3], S_s_e_p.rank())
	# S_s_e = S_s_e_p
	rounding_digit = 10**7
	# print("S_s_e dim:", S_s_e.nrows(), S_s_e.ncols())
	S_s_e = round_matrix_to_rational_precision(S_s_e_p, rounding_digit)
	print("----------------------")
	print("S_s_e@switch: ", matrix(RR,S_s_e[n - 2 :n + 3,n - 2 :n + 3]), S_s_e.rank())
	block11 = S_s_e[:n, :n]
	block12 = S_s_e[:n, n:]
	block21 = S_s_e[n:, :n]
	block22 = S_s_e[n:, n:]

	S_e_s = block4(block22, block21, block12, block11)
	zero_e_s = zero_matrix(ZZ, 1, m + n)
	zero_e_sT = zero_matrix(ZZ, m + n, 1)
	S_1 = zero_matrix(ZZ, 1, 1)
	S = block4(S_e_s, zero_e_sT, zero_e_s, S_1)
	print("S dim:", S.nrows(), S.ncols())

	print("vec_s dim:", mu_s.nrows(), mu_s.ncols())
	print(mu_e.ncols())
	print(mu_s.ncols())
	mu = concatenate([mu_e, mu_s, 1])
	print("mu dim:", mu.nrows(), mu.ncols())
	u = concatenate([e, s, 1])
	# print("is q vector in ellipsoid? ", )
	# print("ellipsoid norm of secret after switching to Kannan: ", scal((ebdd.u - ebdd.mu) * ebdd.S.inverse()/(n + m) * ((ebdd.u - ebdd.mu)).T))
	print("u dim:", u.nrows(), u.ncols())
	return dbdd_class(B, S, mu, u, verbosity=verbosity, D=D, Bvol=m*log(q))

def initialize_from_LWE_instance(dbdd_class, n, q, m, D_e,
                                 D_s, diag=False, verbosity=1,
                                 A=None, s=None, e_vec=None):
    """
    constructor that builds a DBDD instance from a LWE instance
    :n: (integer) size of the secret s
    :q: (integer) modulus
    :m: (integer) size of the error e
    :D_e: distribution of the error e (dictionnary form)
    :D_s: distribution of the secret s (dictionnary form)
    """
    if verbosity:
        logging("     Build DBDD from LWE     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d" % (n, m, q), style="VALUE")
    # define the mean and sigma of the instance
    mu_e, s_e = average_variance(D_e)
    mu_s, s_s = average_variance(D_s)
    mu = vec(m * [mu_e] + n * [mu_s] + [1])
    S = diagonal_matrix(m * [s_e] + n * [s_s] + [0])
    # draw matrix A and define the lattice
    if A is None:
        A = matrix([[randint(0, q) for _ in range(n)] for _ in range(m)])

    B = build_LWE_lattice(-A, q) # primal
    D = build_LWE_lattice(A/q, 1/q) # dual
    # draw the secrets
    if s is None:
        s = vec([draw_from_distribution(D_s) for _ in range(n)])

    if e_vec is None:
        e_vec = vec([draw_from_distribution(D_e) for _ in range(m)])

    # compute the public value t and build a target
    b = (s * A.T + e_vec) % q

    # A_cen = A.apply_map(recenter) #Make sure here A is centered.
    b_cen = b.apply_map(recenter)

    tar = concatenate([b_cen, [0] * n])
    B = kannan_embedding(B, tar)
    D = kannan_embedding(D, concatenate([-b_cen/q, [0] * n])).T
    u = concatenate([e_vec, s, [1]])
    return A, b, dbdd_class(B, S, mu, u, verbosity=verbosity, D=D, Bvol=m*log(q))


def initialize_kannan_ellipsoid_from_LWE_derandom(dbdd_class, n, q, m, D_e,
                                  D_s, homogeneous=True,
                                  verbosity=1, A=None, s=None, e_vec=None):
    """
    constructor that builds an EBDD instance from a LWE instance
    :n: (integer) size of the secret s
    :q: (integer) modulus
    :m: (integer) size of the error e
    :D_e: distribution of the error e (dictionnary form)
    :D_s: distribution of the secret s (dictionnary form)
    """
    if verbosity:
        logging("     Build EBDD from LWE     ", style="HEADER")
        logging("n=%3d \t m=%3d \t q=%d" % (n, m, q), style="VALUE")
    
    # Define mean and variance of instance
    _, s_e = average_variance(D_e)
    _, s_s = average_variance(D_s)
    # Draw centered A matrix
    if A is None:
        A = matrix([[randint(0, q - 1) for _ in range(n)] for _ in range(m)])
    
    # A = A.apply_map(recenter)
    # Draw secrets
    if s is None:
        s = vec([draw_from_distribution(D_s) for _ in range(n)])

    if e_vec is None:
        e_vec = vec([draw_from_distribution(D_e) for _ in range(m)])
    # Compute public value and derived secret
    b = (s * A.T + e_vec) % q
    b_cen = b.apply_map(recenter)
    c = (s * A.T + e_vec - b_cen)/q #c on right hand side

    # Compute Kannan ellipsoid embedding
    mu, S, B_Sigma = kannan_ellipsoid_sca(A, b_cen, q, s_s=s_s, s_e=s_e, homogeneous=homogeneous)
    
    if homogeneous:
        B = identity_matrix(n + m + 1)
        u = concatenate([s, c, [-1]])

    else:
        B = identity_matrix(n + m) 
        u = concatenate([s, c]) 
    # print("ellipnorm of secret@initialize_derandom: ", scal((u - mu) * matrix(np_inv(S)) * ((u - mu)).T))
    # print("ellipnorm of secret@initialize_derandom: ", scal((u - mu) * S.inverse() * ((u - mu)).T))
    
    print("ellipnorm of secret@initialize_derandom: ",RR(scal((u - mu) * B_Sigma*B_Sigma.T/(n + m) * ((u - mu)).T)))
    # print("----------------------")
    # print("Bbbbbb@initial: ", matrix(RR,B_Sigma[n - 2 :n + 3,n - 2 :n + 3]), B_Sigma.rank())
    # print("uuuuuuu@initial", matrix(RR,u[0,n - 5 :n + 5]))
    # print("muuuuuu@initial", matrix(RR,mu[0, n - 5 :n + 5]))

    return A, b_cen, e_vec, dbdd_class(B, S, mu, u, verbosity=verbosity, scaled=True), B_Sigma

def find_ellipsoid_intersection_Sigma(mu1, Sigma1, mu2, Sigma2, tolerance=1.48e-08):
    """
    Ellipsoid intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
    Page 14. Note that for numerical stability, we will minimize the log polynomial instead of
    finding a root. This function only works between two full rank ellipsoids of equal dimension.
    """
    # Set up q1, q2, W1, W2
    q1 = array(mu1, dtype=float)
    q2 = array(mu2, dtype=float)

    W1 = np_inv(array(Sigma1, dtype=float))
    W2 = np_inv(array(Sigma2, dtype=float))

    # Set up commonly used calculations for speed up
    diff_q = q2 - q1
    diff_W = W2 - W1
    W1q1 = W1 @ q1.T # numpy dot product operator
    W2q2 = W2 @ q2.T
    n = W1.shape[0]

    # Define log_polynomial function
    def log_poly(pi):
        X = pi*W1 + (1 - pi)*W2
        X_inv = np_inv(X)
        q_plus = X_inv @ (pi*W1q1 + (1 - pi)*W2q2)
        alpha = 1 - pi*(q1 @ W1 @ q1.T).item() - (1 - pi)*(q2 @ W2 @ q2.T).item() + (q_plus.T @ X @ q_plus).item()

        _, X_det = slogdet(X)
        poly = alpha * trace(X_inv @ (-diff_W)) - n*(2*(q_plus.T @ (W1q1 - W2q2)).item() + (q_plus.T @ diff_W @ q_plus).item() - (q1 @ W1 @ q1.T).item() + (q2 @ W2 @ q2.T).item())

        return 2*X_det + log(poly)

    # Solve for the solution
    _, pi_sol = find_local_minimum(log_poly, 0, 1, tol=tolerance)

    return round_to_rational(pi_sol)
    
def ellipsoid_intersection_Sigma(mu1, Sigma1, mu2, Sigma2, tolerance=1.48e-08):
    """
    Ellipsoid intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
    Page 14. Note that for numerical stability, we will minimize the log polynomial instead of
    finding a root. This function only works between two full rank ellipsoids of equal dimension.
    """
    # Find pi_sol:
    pi_sol = find_ellipsoid_intersection_Sigma(mu1, Sigma1, mu2, Sigma2, tolerance=tolerance)
    print(f"(Solution) pi = {pi_sol.n()}")
    # Calculate final ellipsoid
    W1 = Sigma1.inverse()
    W2 = Sigma2.inverse()
    X = pi_sol*W1 + (1 - pi_sol)*W2
    X_inv = X.inverse()

    # Compute new mean. Use it to compute alpha scaling factor
    new_mu = (X_inv*(pi_sol*W1*mu1.T + (1 - pi_sol)*W2*mu2.T)).T
    
    alpha = 1 - pi_sol*scal(mu1 * W1 * mu1.T) - (1 - pi_sol)*scal(mu2 * W2 * mu2.T) + scal(new_mu * X * new_mu.T)

    print(f"alpha = {alpha.n()}")
    print(f"X*X_inv MSE = {(((X*X_inv) - identity_matrix(X.ncols())).norm('frob')**2)/(X.ncols() * X.ncols())}")
    if alpha < 0:
        raise ValueError("Scaling factor alpha < 0!")
    
    new_Sigma = alpha * X_inv

    return round_matrix_to_rational(new_mu), round_matrix_to_rational(new_Sigma)

	
def ellipsoid_intersection_sca_prev(mu1, W1, mu2, W2, tolerance=1.48e-08):
	"""
	Ellipsoid intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
	Page 14. Note that for numerical stability, we will minimize the log polynomial instead of
	finding a root. This function only works between two full rank ellipsoids of equal dimension.
	"""
	# Find pi_sol:
	pi_sol = find_ellipsoid_intersection_sca(mu1, W1, mu2, W2, tolerance=tolerance)
	print(f"(Solution) pi = {pi_sol.n()}")
	# Calculate final ellipsoid
	# W1 = Sigma1.inverse()
	# W2 = Sigma2.inverse()
	X = pi_sol*W1 + (1 - pi_sol)*W2
	# print("condition")
	# print(matrix(RDF, X).condition(p=2))
	# print("eigenvalues: ", X.eigenvalues())

	X_inv = X.inverse()
	alpha = 1 - pi_sol*(1 - pi_sol)*scal((mu2 - mu1)*(W2*X_inv*W1)*(mu2 - mu1).T)

	print(f"alpha = {alpha.n()}")
	print(f"X*X_inv RMSE = {(((X*X_inv) - identity_matrix(X.ncols())).norm())/(X.ncols() * X.ncols())}")
	if alpha < 0:
		raise ValueError("Scaling factor alpha < 0!")
	
	new_mu = (X_inv*(pi_sol*W1*mu1.T + (1 - pi_sol)*W2*mu2.T)).T
	new_Sigma = alpha * X_inv

	return round_matrix_to_rational(new_mu), round_matrix_to_rational(new_Sigma)

def find_ellipsoid_intersection_sca_prev(mu1, W1, mu2, W2, tolerance=1.48e-08):
	"""
	Ellipsoid intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
	Page 14. Note that for numerical stability, we will minimize the log polynomial instead of
	finding a root. This function only works between two full rank ellipsoids of equal dimension.
	"""
	# Set up q1, q2, W1, W2
	q1 = array(mu1, dtype=float)
	q2 = array(mu2, dtype=float)

	# W1 = np_inv(array(Sigma1, dtype=float))
	# W2 = np_inv(array(Sigma2, dtype=float))
	W1 = array(W1, dtype=float)
	W2 = array(W2, dtype=float)

	# Set up commonly used calculations for speed up
	diff_q = q2 - q1
	diff_W = W2 - W1
	W1q1 = W1 @ q1.T # numpy dot product operator
	W2q2 = W2 @ q2.T
	n = W1.shape[0]

	# Define log_polynomial function
	def log_poly(pi):
		X = pi*W1 + (1 - pi)*W2
		X_inv = np_inv(X)
		alpha = 1 - pi*(1 - pi)*(diff_q @ W2 @ X_inv @ W1 @ diff_q.T).item()
		q_plus = X_inv @ (pi*W1q1 + (1 - pi)*W2q2)

		_, X_det = slogdet(X)
		poly = alpha * trace(X_inv @ (-diff_W)) - n*(2*(q_plus.T @ (W1q1 - W2q2)).item() + (q_plus.T @ diff_W @ q_plus).item() - (q1 @ W1 @ q1.T).item() + (q2 @ W2 @ q2.T).item())

		return 2*X_det + log(poly)

	# Solve for the solution
	_, pi_sol = find_local_minimum(log_poly, 0, 1, tol=tolerance)

	return round_to_rational(pi_sol)

"""
def ellipsoid_intersection_sca(mu1, W1, mu2, W2, tolerance=1e-10):
	
	Ellipsoid intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
	Page 14. Replace q, Q with mu, Sigma for readability.
	
	# # Set up W1, W2
	# try:
	#     W1 = degen_inverse(Sigma1)

	# except AssertionError:
	#     W1 = Sigma1.inverse()

	# try:
	#     W2 = degen_inverse(Sigma2)

	# except AssertionError:
	#     W2 = Sigma2.inverse()


	# Set up commonly used calculations for speed up
	diff_q = mu2 - mu1
	diff_W = W2 - W1
	W1q1 = W1 * mu1.T
	W2q2 = W2 * mu2.T
	n = W1.nrows()

	# Define Lambda functions for calculating polynomial and ellipsoid
	X = lambda pi: pi*W1 + (1 - pi)*W2
	alpha = lambda pi: 1 - pi*(1 - pi) * scal(diff_q * (W2 * X(pi).inverse() * W1) * diff_q.T)
	q_plus = lambda pi: X(pi).inverse() * (pi * W1q1 + (1 - pi) * W2q2)
	Q_plus = lambda pi: alpha(pi) * X(pi).inverse()
	
	# Define polynomial function
	def poly(pi):
		X_det = X(pi).determinant()
		X_det2 = X_det*X_det
		term1 = alpha(pi) * X_det2 * (X(pi).inverse() * (-diff_W)).trace()
		term2 = 2 * scal(q_plus(pi).T * (W1q1 - W2q2))
		term3 = scal(q_plus(pi).T * diff_W * q_plus(pi))
		term4 = scal(mu1 * W1q1)
		term5 = scal(mu2 * W2q2)
		return term1 - n*X_det2*(term2 + term3 - term4 + term5)

	# Solve for the solution, but first check boundary conditions
	poly_0 = poly(0)
	poly_1 = poly(1)

	if abs(poly_0) <= tolerance and abs(poly_1) > tolerance:
		pi_sol = 0
	
	elif abs(poly_0) > tolerance and abs(poly_1) <= tolerance:
		pi_sol = 1

	else:
		pi_sol = bisection_method(poly, 0, 1, tolerance=tolerance)
		
	return round_matrix_to_rational(q_plus(pi_sol).T), round_matrix_to_rational(Q_plus(pi_sol))
	"""
