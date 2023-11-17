from numpy.linalg import inv as np_inv
from numpy.linalg import svd, det
# from numpy.linalg import slogdet as np_slogdet
from numpy import array, trace, log, diag
from numpy import sqrt as np_sqrt
from scipy.linalg import sqrtm
from scipy.optimize import bisect, brenth, minimize_scalar
import numpy as np
import sys
from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction

load("../framework/utils.sage")

def dual_basis(B):
    """
    Compute the dual basis of B
    """
    return B.pseudoinverse().transpose()


def projection_matrix(A):
    """
    Construct the projection matrix orthogonally to Span(V)
    """
    S = A * A.T
    return A.T * S.inverse() * A


def project_against(v, X):
    """ Project matrix X orthonally to vector v"""
    # Pv = projection_matrix(v)
    # return X - X * Pv
    Z = (X * v.T) * v / scal(v * v.T)
    return X - Z


# def make_primitive(B, v):
#     assert False
#     # project and Scale v's in V so that each v
#     # is in the lattice, and primitive in it.
#     # Note: does not make V primitive as as set of vector !
#     # (e.g. linear dep. not eliminated)
#     PB = projection_matrix(B)
#     DT = dual_basis(B).T
#     v = vec(v) * PB
#     w = v * DT
#     den = lcm([x.denominator() for x in w[0]])
#     num = gcd([x for x in w[0] * den])
#     if num==0:
#         return None
#     v *= den/num
#     return v


def vol(B):
    return sqrt(det(B * B.T))


def project_and_eliminate_dep(B, W):
    # Project v on Span(B)
    PB = projection_matrix(B)
    V = W * PB
    rank_loss = V.nrows() - V.rank()

    if rank_loss > 0:
        print("WARNING: their were %d linear dependencies out of %d " %
              (rank_loss, V.nrows()))
        V = V.LLL()
        V = V[rank_loss:]

    return V


def is_cannonical_direction(v):
    v = vec(v)
    return sum([x != 0 for x in v[0]]) == 1


def cannonical_param(v):
    v = vec(v)
    assert is_cannonical_direction(v)
    i = [x != 0 for x in v[0]].index(True)
    return i, v[0, i]


def eliminate_linear_dependencies(B, nb_dep=None, dim=None):
    """
    Transform a lattice generator set into a lattice basis
    :B: Generator set of the lattice
    :nb_dep: Numbers of linear dependencies in B (optional)
    :dim: The rank of the lattice (optional)
    """

    # Get the number of dependencies, if possible
    nrows = B.nrows()
    if (nb_dep is None) and (dim is not None):
        nb_dep = nrows - dim
    assert (dim is None) or (nb_dep + dim == nrows)

    if nb_dep is None or nb_dep > 0:
        # Remove dependencies
        B = B.LLL()
        nb_dep = min([i for i in range(nrows) if not B[i].is_zero()]) \
                if nb_dep is None else nb_dep
        B = B[nb_dep:]

    return B


def lattice_orthogonal_section(D, V, assume_full_rank=False, output_basis=True, square=False):
    """
    Compute the intersection of the lattice L(B)
    with the hyperplane orthogonal to Span(V).
    (V can be either a vector or a matrix)
    INPUT AND OUTPUT DUAL BASIS
    :assume_full_rank: if True, assume V is already in Span(B),
            to avoid projection computation (for optimization purpose)
    :output_basis: if False, return only a lattice generator set instead of a basis,
            to avoid expensive process of eliminating linear dependencies with LLL
            (for optimization purpose)
    :square: if True, run LLL column-wise to produce a lower-dimensional square
             basis.
    Algorithm:
    - project V onto Span(B)
    - project the dual basis onto orth(V)
    - eliminate linear dependencies (LLL)
    - go back to the primal.
    """

    if not assume_full_rank:
        V = project_and_eliminate_dep(D, V)
    r = V.nrows()

    # Project the dual basis orthogonally to v
    PV = projection_matrix(V)
    D = D - D * PV

    # Eliminate linear dependencies
    if output_basis:
        D = eliminate_linear_dependencies(D, nb_dep=r)
        # Create square basis
        if square:
            D = eliminate_linear_dependencies(D.T, nb_dep=r).T

    # Go back to the primal
    return D


def lattice_project_against(B, V, assume_full_rank=False, assume_belonging=False, output_basis=True):
    """
    Compute the projection of the lattice L(B) orthogonally to Span(V). All vectors if V
    (or at least their projection on Span(B)) must belong to L(B).
    :assume_full_rank: if True, assume V is already in Span(B),
            to avoid projection computation (for optimization purpose)
    :assume_belonging: if True, assume V is already in L(B),
            to avoid the verification (for optimization purpose)
    :output_basis: if False, return only a lattice generator set instead of a basis,
            to avoid expensive process of eliminating linear dependencies with LLL
            (for optimization purpose)
    Algorithm:
    - project V onto Span(B)
    - project the basis onto orth(V)
    - eliminate linear dependencies (LLL)
    """
    # Project v on Span(B)
    if not assume_full_rank:
        V = project_and_eliminate_dep(B, V)
    r = V.nrows()

    # Check that V belongs to L(B)
    if not assume_belonging:
        D = dual_basis(B)
        M = D * V.T
        if not lcm([x.denominator() for x in M.list()]) == 1:
            raise ValueError("Not in the lattice")

    # Project the basis orthogonally to v
    PV = projection_matrix(V)
    B = B - B * PV

    # Eliminate linear dependencies
    if output_basis:
        B = eliminate_linear_dependencies(B, nb_dep=r)

    # Go back to the primal
    return B


def lattice_modular_intersection(D, V, k, assume_full_rank=False, output_basis=True):
    """
    Compute the intersection of the lattice L(B) with
    the lattice {x | x*V = 0 mod k}
    (V can be either a vector or a matrix)
    :assume_full_rank: if True, assume V is already in Span(B),
            to avoid projection computation (for optimization purpose)
    :output_basis: if False, return only a lattice generator set instead of a basis,
            to avoid expensive process of eliminating linear dependencies with LLL
            (for optimization purpose)
    Algorithm:
    - project V onto Span(B)
    - append the equations in the dual
    - eliminate linear dependencies (LLL)
    - go back to the primal.
    """
    # Project v on Span(B)
    if not assume_full_rank:
        V = project_and_eliminate_dep(D, V)
    r = V.nrows()
    # append the equation in the dual
    V /= k
    # D = dual_basis(B)
    D = D.stack(V)

    # Eliminate linear dependencies
    if output_basis:
        D = eliminate_linear_dependencies(D, nb_dep=r)

    # Go back to the primal
    return D


def is_diagonal(M):
    if M.nrows() != M.ncols():
        return False
    A = M.numpy()
    return np.all(A == np.diag(np.diagonal(A)))


def logdet(M, exact=False):
    """
    Compute the log of the determinant of a large rational matrix,
    tryping to avoid overflows.
    """
    if not exact:
        MM = array(M, dtype=float)
        _, l = slogdet(MM)
        return l

    a = abs(M.det())
    l = 0

    while a > 2**32:
        l += RR(32 * ln(2))
        a /= 2**32

    l += ln(RR(a))
    return l


def degen_inverse(S, B=None):
    """ Compute the inverse of a symmetric matrix restricted
    to its span
    """
    # Get an orthogonal basis for the Span of B

    if B is None:
        # Get an orthogonal basis for the Span of B
        V = S.echelon_form()
        V = V[:V.rank()]
        P = projection_matrix(V)
    else:
        P = projection_matrix(B)

    # make S non-degenerated by adding the complement of span(B)
    C = identity_matrix(S.ncols()) - P
    Sinv = (S + C).inverse() - C

    assert S * Sinv == P, "Consistency failed (probably not your fault)."
    assert P * Sinv == Sinv, "Consistency failed (probably not your fault)."

    return Sinv


def degen_logdet(S, B=None, eigh=False, assume_full_rank=False):
    """ Compute the determinant of a symmetric matrix
    sigma (m x m) restricted to the span of the full-rank
    rectangular (k x m, k <= m) matrix V
    """
    # Get an orthogonal basis for the Span of B
    if assume_full_rank:
        P = identity_matrix(S.ncols())

    elif not assume_full_rank and B is None:
        # Get an orthogonal basis for the Span of B
        V = S.echelon_form()
        V = V[:V.rank()]
        P = projection_matrix(V)

    else:
        P = projection_matrix(B)
       
    if eigh:
        C = identity_matrix(S.ncols()) - P
        eigs = np.linalg.eigvalsh(array(S + C, dtype=float))
        leigs = np.log(np.where(eigs < 1e-10, 1, eigs))
        l3 = sum(leigs)
    
    else:
        # make S non-degenerated by adding the complement of span(B)
        C = identity_matrix(S.ncols()) - P
        # Check that S is indeed supported by span(B)
        # assert (S - P.T * S * P).norm() < 1e-10

        l3 = logdet(S + C)

    return l3


def square_root_inverse_degen(S, B=None, assume_full_rank=False):
    """ Compute the determinant of a symmetric matrix
    sigma (m x m) restricted to the span of the full-rank
    rectangular (k x m, k <= m) matrix V
    """
    
    if assume_full_rank:
        P = identity_matrix(S.ncols())

    elif not assume_full_rank and B is None:
        # Get an orthogonal basis for the Span of B
        V = S.echelon_form()
        V = V[:V.rank()]
        P = projection_matrix(V)

    else:
        P = projection_matrix(B)

    # make S non-degenerated by adding the complement of span(B)
    C = identity_matrix(S.ncols()) - P
    # Take matrix sqrt via SVD, then inverse
    # S = adjust_eigs(S)
    
    u, s, vh = svd(array(S + C, dtype=float))
    L_inv = np_inv(vh) @ np_inv(np_sqrt(diag(s))) @ np_inv(u)
    # L_inv = np_inv(sqrtm(array(S + C, dtype=float)))
    
    L_inv = np_inv(cholesky(array(S + C, dtype=float))).T
    L_inv = round_matrix_to_rational(L_inv)
    L = L_inv.inverse()


    # scipy outputs complex numbers, even for real valued matrices. Cast to real before rational.
    #L = round_matrix_to_rational(u @ np_sqrt(diag(s)) @ vh)

    return L, L_inv


def check_basis_consistency(B=None, D=None, Bvol=None):
    """ Check if the non-null parameters are consistent between them
    """
    try:
        active_basis = B or D
        if active_basis:
            assert (B is None) or (D is None) or (D.T * B == identity_matrix(active_basis.nrows()))
            if Bvol is not None:
                read_Bvol = logdet(active_basis)
                read_Bvol *= (-1)**(active_basis!=B)
                assert abs(Bvol - read_Bvol) < 1e-6
        return True

    except AssertionError:
        return False


def build_substitution_matrix(V, pivot=None, output_extra_data=True):
    """ Compute the substitution matrix Γ of a linear system X⋅M^T = 0 where we know X⋅V^T.
    After substitution, the smaller linear system X'⋅M'^T = 0 will verify X = X'⋅Γ^T.
    """
    dim = V.ncols()

    # Find a pivot for V
    _, pivot = V.nonzero_positions()[0] if (pivot is None) else (None, pivot)
    assert V[0,pivot] != 0, 'The value of the pivot must be non-zero.'

    # Normalize V according to the pivot
    V1 = - V[0,:pivot] / V[0,pivot]
    V2 = - V[0,pivot+1:] / V[0,pivot]

    # Build the substitution matrix
    Gamma = zero_matrix(QQ, dim,dim-1)
    Gamma[:pivot,:pivot] = identity_matrix(pivot)
    Gamma[pivot,:pivot] = V1
    Gamma[pivot,pivot:] = V2
    Gamma[pivot+1:,pivot:] = identity_matrix(dim-pivot-1)

    if not output_extra_data:
        return (Gamma, None)

    # Compute efficiently
    #  - the determinant of (Gamma.T * Gamma)
    #  - the 'pseudo_inv' := (Gamma.T * Gamma).inv()
    det = 1 + scal(V1*V1.T) + scal(V2*V2.T)
    pseudo_inv = zero_matrix(QQ, dim-1,dim-1)
    pseudo_inv[:pivot,:pivot] = identity_matrix(pivot) - V1.T*V1 / det
    pseudo_inv[pivot:,pivot:] = identity_matrix(dim-pivot-1) - V2.T*V2 / det
    pseudo_inv[:pivot,pivot:] = - V1.T*V2 / det
    pseudo_inv[pivot:,:pivot] = - V2.T*V1 / det

    return (Gamma, (det, pseudo_inv))

def adjust_eigs(A):
    w, v = np.linalg.eigh(A)
    print(f"adjust_eigs:\n{w}")
    w[w < 1e-10] = 1e-10
    print(f"adjust_eigs:\n{w}")
    W = diagonal_matrix(w)
    V = matrix(v)
    w, v = np.linalg.eigh(V * W * V.T)
    print(f"adjust_eigs:\n{w}")
    return V * W * V.T

def solve_diophantine_system_LWE(V, gamma, lwe_instance):
    """ Solves the diophantine system xV.T = gamma for x in
    the LWE lattice.
    :V: Integer coefficient matrix consisting of row vectors.
    :gamma: Integer output constraints.
    :lwe_instance: LWE object
    """
    # First put coefficient matrix into HNF
    H, U = V.T.hermite_form(transformation=True)
    k = matrix(ZZ, H.solve_left(gamma))
    fixed_vars = H.rank()
    # Use free variables to solve for a solution to y [A | I].T = b mod q
    A = lwe_instance.A
    b = lwe_instance.b
    q = lwe_instance.q

    AI = matrix(GF(q), identity_matrix(A.nrows()).augment(A))
    b = matrix(GF(q), b)

    # Remove Non-free params from RHS
    bp = b - k*U*AI.T

    # Solve for free variables over mod q
    residual = matrix(ZZ, (U[fixed_vars:, :]*AI.T).solve_left(bp)).apply_map(recenter)
    # Add residual to the non free parameters to get final solution
    sol = (k + concatenate([matrix(1, fixed_vars), matrix(ZZ, residual)]))*U
    
    assert matrix(ZZ, sol*AI.T) % q == matrix(ZZ, b)
    assert sol*V.T == gamma
    # assert out*A.T - z*b == 0

    return sol

def bisection_method(f, low, high, tolerance=1e-10, max_iter=100):
    if low > high:
        raise ValueError("low point must be less then high point!")
    
    if (f(low) < 0 and f(high) > 0) or (f(low) > 0 and f(high) < 0):
        for i in range(max_iter):
            # print(f"Bisection iteration {i}")#temp too many printout
            mid = (low + high)/2
            if f(mid) == 0 or (high - low) < tolerance:
                return mid

            if (f(mid) > 0) == (f(low) > 0):
                low = mid
            
            else:
                high = mid

        raise ValueError(f"Convergence not achieved in {max_iter} iterations.")

    else:
        raise ValueError(f"Zero does not exist between {low} and {high}.")

def find_ellipsoid_intersection(mu1, Sigma1, mu2, Sigma2, tolerance=1.48e-08):
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
    diff_W = W2 - W1
    W1q1 = W1 @ q1.T # numpy dot product operator
    W2q2 = W2 @ q2.T
    n = W1.shape[0]

    # Define log(polynomial) function
    def poly(pi):
        X = pi*W1 + (1 - pi)*W2
        X_inv = np_inv(X)
        q_plus = X_inv @ (pi*W1q1 + (1 - pi)*W2q2)
        alpha = 1 - pi*(q1 @ W1 @ q1.T).item() - (1 - pi)*(q2 @ W2 @ q2.T).item() + (q_plus.T @ X @ q_plus).item()

        poly = alpha * trace(X_inv @ (-diff_W)) - n*(2*(q_plus.T @ (W1q1 - W2q2)).item() + (q_plus.T @ diff_W @ q_plus).item() - (q1 @ W1 @ q1.T).item() + (q2 @ W2 @ q2.T).item())
        return poly

    def objective(pi):
        X = pi*W1 + (1 - pi)*W2
        X_inv = np_inv(X)
        q_plus = X_inv @ (pi*W1q1 + (1 - pi)*W2q2)
        alpha = 1 - pi*(q1 @ W1 @ q1.T).item() - (1 - pi)*(q2 @ W2 @ q2.T).item() + (q_plus.T @ X @ q_plus).item()
        return logdet(alpha * X_inv)

    # Solve for the solution
    try:
        x = np.linspace(0, 1, 100)
        objectives = np.array([objective(pi) for pi in x])
        #plt.plot(x, objectives)
        #plt.show()
        pi_sol = brenth(poly, 0, 1, xtol=tolerance, rtol=tolerance, maxiter=250)

    except (NotImplementedError, RuntimeError) as e:
        # Brent's method failed. Use bisection
        print(type(e), file=sys.stderr)
        print(e, file=sys.stderr)
        print("Using bisection method as fallback...", file=sys.stderr)
        pi_sol = bisect(poly, 0, 1, xtol=tolerance, rtol=tolerance, maxiter=250)
    
#    except ValueError as e:
#        print(type(e), file=sys.stderr)
#        print(e, file=sys.stderr)
#        print("Using min/max as fallback...", file=sys.stderr)
#
#        if poly(0) >= 0:
#            pi_sol = minimize_scalar(poly, bounds=(0, 1), method="bounded", tol=tolerance).x
#
#        else:
#            pi_sol = minimize_scalar(lambda pi: -1*poly(pi), bounds=(0, 1), method="bounded", tol=tolerance).x
#
#        print(f"poly(pi_sol) = {poly(pi_sol)}", file=sys.stderr)
#        print("poly(1, 0) =", poly(1), poly(0), file=sys.stderr)

    return pi_sol
    
def ellipsoid_intersection(mu1, Sigma1, mu2, Sigma2, tolerance=1.48e-08):
    """
    Ellipsoid intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
    Page 14. Note that for numerical stability, we will minimize the log polynomial instead of
    finding a root. This function only works between two full rank ellipsoids of equal dimension.
    """
    # Find pi_sol:
    # print("this version")
    #print("rank", Sigma1.rank(), Sigma2.rank(), Sigma1.ncols(), Sigma1.nrows())
    pi_sol = find_ellipsoid_intersection(mu1, Sigma1, mu2, Sigma2, tolerance=tolerance)
    # print(Sigma1[0][0])
    # print(Sigma2[0][0])
    # print(f"(Solution) pi = {pi_sol.n()}")
    # Calculate final ellipsoid
    W1 = Sigma1.inverse()
    W2 = Sigma2.inverse()
    X = pi_sol*W1 + (1 - pi_sol)*W2
    X_inv = round_matrix_to_rational(X.inverse())

    # Compute new mean. Use it to compute alpha scaling factor
    new_mu = (X_inv*(pi_sol*W1*mu1.T + (1 - pi_sol)*W2*mu2.T)).T
    
    alpha = 1 - pi_sol*scal(mu1 * W1 * mu1.T) - (1 - pi_sol)*scal(mu2 * W2 * mu2.T) + scal(new_mu * X * new_mu.T)

    # print(f"alpha = {alpha.n()}")
    # print(f"X*X_inv MSE = {(((X*X_inv) - identity_matrix(X.ncols())).norm('frob')**2)/(X.ncols() * X.ncols())}")
    if alpha < 0:
        raise ValueError("Scaling factor alpha < 0!")
    
    new_Sigma = alpha * X_inv

    return round_matrix_to_rational(new_mu), round_matrix_to_rational(new_Sigma)

def ellipsoid_hyperboloid_intersection(mu1, Sigma1, mu2, Sigma2, tolerance=1.48e-08):
    """
    Ellipsoid/Hyperboloid intersection method from https://folk.ntnu.no/skoge/prost/proceedings/ifac2002/data/content/01090/1090.pdf
    Page 4. This function only works between two full rank quadratic forms of equal dimension, and assumes that Sigma1 is
    positive semi-definite.
    """
    
    # determines the maximum value of the parametrization for which the result will still be an ellipsoid
    lambda_min = min(e for e, _, _ in Sigma2.eigenvectors_left(Sigma1))
    tau_max = min(1, 1 / (1 - lambda_min))
    
    def ellipsoid(tau):
        # calculates the parametrized intersection ellipsoid given the parameter tau
        
        def c(a, b):
            # convex combination of two values based on the parameter
            t = float(tau.real)
            return (1 - t) * a + t * b
        
        # computes the new mean and variance
        Sigma_tau = c(Sigma1, Sigma2)
        mu = Sigma_tau^-1 * c(Sigma1 * mu1, Sigma2 * mu2)
        
        v = scal(c(mu1.T * Sigma1 * mu1, mu2.T * Sigma2 * mu2) - mu.T * Sigma_tau * mu)
        Sigma = (1 - v)^-1 * Sigma_tau
        
        return mu, Sigma
    
    def determinant(tau):
        # calculates the determinant of the parametrized ellipsoid
        _, Sigma = ellipsoid(tau)
        return -ln(Sigma.det())
    
    # determines the value of tau which minimizes the determinant
    res = minimize_scalar(determinant, bounds=(0, tau_max), method="bounded", options={'xatol': tolerance})
    tau_min = res.x
    
    # calculates and returns the minimal parametrized ellipsoid
    new_mu, new_Sigma  = ellipsoid(tau_min)
    return round_matrix_to_rational(new_mu), round_matrix_to_rational(new_Sigma)

def ellipsoid_quadratic_froms_intersection(*mu_Sigma, lb=1e-8, tolerance=1.48e-08):
    # determines the maximum value of the parametrization for which the quadratic form will be PSD
    # lambda_min = min(e for e, _, _ in Sigma2.eigenvectors_left(Sigma1))
    # tau_max = min(1, 1 / (1 - lambda_min)-.001)
    tau_max = 1
    LARGE_PENALTY_VALUE = 1e10
    # Initialize empty lists to store mu's and sigma's
    mus = []
    Sigmas = []
    # Loop through the arguments two at a time
    for i in range(0, len(mu_Sigma), 2):
        mu, Sigma = mu_Sigma[i], mu_Sigma[i+1]
        
        # Append mu and sigma to their respective lists
        mus.append(matrix(mu.reshape(-1, 1)))
        Sigmas.append(matrix(Sigma))
    n = len(mu_Sigma)//2
    def ellipsoid(tau):
        # calculates the parametrized intersection ellipsoid given the parameter tau
        # define a function that computes the convex combination of n matrices from Sigma
        def c_n(Sigma):
            sum_Sigma = matrix(np.zeros_like(Sigma[0]).tolist())
            for i in range(n):
                sum_Sigma += float(tau[i].real) * Sigma[i]
            return sum_Sigma

        # computes the new mean and variance
        Sigma_tau = c_n(Sigmas)
        sum_tau_Sigma_mu = matrix(np.zeros_like(mus[0]).tolist())
        for i in range(n):
            sum_tau_Sigma_mu += float(tau[i].real) * Sigmas[i] * mus[i]
        Sigma_tau_inv = Sigma_tau^-1
        mu = Sigma_tau_inv * sum_tau_Sigma_mu   # Make sure the shapes are correct
        v = 0
        for i in range(n):
            v += float(tau[i].real) * mus[i].T  * Sigmas[i] * mus[i]
        # v -= sum_tau_Sigma_mu.T * Sigma_tau * sum_tau_Sigma_mu
        v -= sum_tau_Sigma_mu.T * Sigma_tau_inv * sum_tau_Sigma_mu
        # print("v: ", v)
        # v = scal(v)
        if (1-v) < lb:
            v = 1-lb
        Sigma = (1 - scal(v))^-1 * Sigma_tau
        return mu, Sigma
    def determinant(tau):
        # calculates the determinant of the parametrized ellipsoid
        _, Sigma = ellipsoid(tau)
        if not is_pos_def(Sigma):
        # If Sigma is not positive semidefinite, return a large penalty value
            return LARGE_PENALTY_VALUE
        return -ln(Sigma.det())
    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)
    
    x0 = np.zeros(n)
    x0[0] = 1
    bounds = [(0, 1) for _ in range(n)]
    linear_constraint = LinearConstraint(np.ones(n), 1, 1)
    # determines the value of tau which minimizes the determinant
    res = minimize(determinant, x0, bounds=bounds, constraints=linear_constraint, method="trust-constr", options={'gtol': 1e-12, 'xtol': tolerance, 'verbose': 3})
    tau_min = res.x
    # print if the convex combination is positive semi-definite
    print("Positive definite: %s" %is_pos_def(ellipsoid(tau_min)[1]))
    print("Tau min: ", tau_min)
    # calculates and returns the minimal parametrized ellipsoid
    new_mu, new_Sigma  = ellipsoid(tau_min)
    return round_matrix_to_rational(new_mu), round_matrix_to_rational(new_Sigma)

def hyperplane_intersection(mu, Sigma, c, gamma, normalize=False):
    """
    Ellipsoid/Hyperplane intersection method from https://people.eecs.berkeley.edu/~akurzhan/ellipsoids/ET_TechReport_v1.pdf
    Page 13. Replace q, Q with mu, Sigma for readability.
    """
    # Find orthogonal matrix S by eq (2.18 - 2.19)
    w = matrix(QQ, [1] + (c.ncols() - 1)*[0])
    # mu = matrix(RDF, mu)
    # Sigma = matrix(RDF, Sigma)
    # c = matrix(RDF, c)

    # First scale c and gamma to ensure c is a unit vector
    if normalize:
        gamma /= c.norm()
        c /= c.norm()

    # Create S via householder reflections
    v = c - w
    R = identity_matrix(c.ncols()) - 2*v.T*v/scal(v*v.T)
    R[:, -1] *= -1 # Correct Determinant
    S = R.T

    # Transform ellipsoid given S and gamma
    mu_prime = mu*S.T - gamma*w
    Sigma_prime = S*Sigma*S.T

    M = S*Sigma.inverse()*S.T

    # Set up slices for final calculations
    U = Sigma_prime[1:, 1:]
    m11 = M[0, 0]
    mu1 = mu_prime[0, 0]
    m_bar = M[0, 1:]

    # Use Sherman-Morrison Formula
    # https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    # to compute \bar{M}^{-1}
    M_bar_inv = U - (U * m_bar.T*m_bar * U)/scal(m11 + m_bar*U*m_bar.T)
    zero1 = zero_matrix(QQ, 1, 1)
    zero2 = zero_matrix(QQ, 1, M_bar_inv.ncols())

    # Apply corrections
    w_prime = (mu_prime + mu1 * concatenate(-1, (m_bar*M_bar_inv.T)))[0, 1:]
    W_prime = (1 - (mu1*mu1)*(m11 - scal(m_bar*M_bar_inv*m_bar.T))) * M_bar_inv # block4(zero1, zero2, zero2.T, M_bar_inv)

    return w_prime, W_prime, S
    # # Transform back to original coordinate space
    # new_mu = w_prime*S + gamma*c
    # new_Sigma = round_matrix_to_rational(S.T*W_prime*S)
    # adjustment = identity_matrix(new_Sigma.ncols()) * min(new_Sigma.eigenvalues())
    # new_Sigma -= adjustment

    # return new_mu, new_Sigma

def rank_one_update(dim, D, v, sigma):
    t = np.zeros(dim+1, )
    d = D.diagonal()
    d_tilde = np.zeros(dim,)
    zeta = np.zeros(dim,)
    l_tilde = np.zeros([dim, dim])

    t[dim] = 1 - sigma
    for j in reversed(range(dim)):
        t[j] = t[j+1] + sigma * (v[0, j]**2)/d[j]
        d_tilde[j] = d[j] * t[j+1] / t[j]
        zeta[j] = - sigma * v[0, j] / (d[j] * t[j+1])

    for i in range(dim):
        for j in range(dim):
            if j < i:
                l_tilde[i, j] = v[0, i] * zeta[j]
            
            elif j == i:
                l_tilde[i, j] = 1

            else:
                l_tilde[i, j] = 0

    return matrix(l_tilde), diagonal_matrix(d_tilde)
