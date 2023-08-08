from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
import numpy as np

load("../framework/load_strategies.sage")
load("../framework/DBDD_generic.sage")
load("../framework/proba_utils.sage")

def scale_ellipsoid_prior(scaling_factor="rank"):
    """
    Decorator that checks for ellipsoid scaling and rescales the ellipsoid
    if needed. @scale_ellipsoid_prior("rank") will scale the ellipsoid
    to (x-\mu) \Sigma^-1 (x-\mu)^T <= n+m.
    """
    def decorator(fn):
        def decorated(self, *args, **kwargs):
            # Check for rank scaling:
            if scaling_factor == "rank":
                scaling = self.expected_length

            else:
                scaling = scaling_factor

            self.scale_ellipsoid(scaling)
            return fn(self, *args, **kwargs)

        return decorated
    return decorator

class EBDD(DBDD_generic):
    """
    This class defines all the elements defining a EBDD instance with all
    the basis computations
    """

    def __init__(self, B, S, mu, embedded_instance, u=None, verbosity=1, homogeneous=False, float_type="ld", D=None, Bvol=None, ellip_scale=1, calibrate_volume=True, circulant=False):
        """constructor that builds a EBDD instance from a lattice, mean, sigma
        and a target
        ;min_dim: Number of coordinates to find to consider the problem solved
        :B: Basis of the lattice
        :S: The Covariance matrix (Sigma) of the uSVP solution
        :mu: The expectation of the uSVP solution
        :u: The unique vector to be found (optinal, for verification purposes)
        :fp_type: Floating point type to use in FPLLL ("d, ld, dd, qd")
        :calibrate_volume: Whether or not to scale by the ellipsoid norm when calculating the volume
        """
        self.calibrate_volume = calibrate_volume
        self.offset = None
        self.integrated_perfect_hints = {}
        self.verbosity = verbosity
        self.B = B  # The lattice Basis
        self.D = D  # The dual Basis
        assert B or D # B or D must be active
        assert check_basis_consistency(B, D, Bvol)
        self.S = S
        self.PP = 0 * S  # Span of the projections so far (orthonormal)
        self.mu = mu
        self.embedded_instance=embedded_instance
        self.homogeneous = homogeneous
        if homogeneous and scal(mu * mu.T) > 0:
            raise InvalidArgument("Homogeneous instances must have mu=0")
        self.u = u
        self.u_original = u
        self.expected_length = self.dim() 
        self.projections = 0
        self.partially_isotropized = False
        self.ellip_scale = ellip_scale
        self.save = {"save": None}
        self.float_type = float_type
        self.estimate_attack(silent=True)
        self.circulant = circulant

    def scale_ellipsoid(self, scaling_factor):
        # Check if ellipsoid doesn't need to be scaled:
        if scaling_factor == self.ellip_scale:
            pass
            
        else:
            self.S *= (self.ellip_scale / scaling_factor)
            self.ellip_scale = scaling_factor

    def convert_hint_e_to_c(self, v, l):
        """
        Converts a hint from the (e || s) coordinate space to the
        (c || s) coordinate space.
        """
        m = self.embedded_instance.m
        q = self.embedded_instance.q
        A = self.embedded_instance.A
        b = self.embedded_instance.b


        ve = v[0, 0:m]
        vs = v[0, m:]

        return concatenate([q*ve, vs - ve*A]), l - scal(b*ve.T)

    def convert_hint_c_to_e(self, v, l):
        """
        Converts a hint from the (c || s) coordinate space to the
        (e || s) coordinate space.
        """
        m = self.embedded_instance.m
        q = self.embedded_instance.q
        A = self.embedded_instance.A
        b = self.embedded_instance.b

        vc = v[0, 0:m]
        vs = v[0, m:]

        return concatenate([vc/q, vs + (vc/q)*A]), l + scal(b*vc.T)/q

    def dim(self):
        if self.B is not None:
            return self.B.nrows()
        else:
            return self.D.nrows()

    def S_diag(self):
        return [self.S[i, i] for i in range(self.S.nrows())]

    @scale_ellipsoid_prior(scaling_factor="rank")
    def volumes(self):
        if self.B is not None:
            Bvol = logdet(self.B * self.B.T) / 2
        else:
            Bvol = -logdet(self.D * self.D.T) / 2
            self.B = dual_basis(self.D)

        # Homogenization doesn't practically affect the experimental hardness
        # don't homogenize during estimation if there's a possibility of
        # numerical error.
        if self.mu.norm() < 1e10:
            S = self.homogenize_S()
            B = block4(self.B, matrix(self.B.nrows(), 1),
                        matrix(1, self.B.ncols()), matrix([1]))
        
        else:
            S = self.S
            B = self.B

        # Calculate ellipsoid norm if secret is known for calibrated estimate
        if self.calibrate_volume and self.u is not None:
            ellip_norm = self.ellip_norm() / self.dim()

        else:
            ellip_norm = 1

        Svol = degen_logdet(S, B, eigh=True) + self.dim()*ln(ellip_norm)
        dvol = Bvol - Svol / 2.
        return (Bvol, Svol, dvol)

    def leak(self, v):
        value = scal(self.u * v.T)
        return value

    def homogenize_S(self, embed_coeff=1):
        # print(np.linalg.eigvalsh(self.S))
        dim_ = self.dim()
        self.scale_ellipsoid(dim_)
        S = block4(self.S + self.mu.T*self.mu, self.mu.T, self.mu, matrix([embed_coeff]))
        # print(np.linalg.eigvalsh(S))
        return S

    def ellip_norm(self, u=self.u):
        if u is None:
            raise InvalidArgument("Solution vector must exist to calculate norm")

        try:
            _, Linv = square_root_inverse_degen(self.S, self.B)
            inv = Linv*Linv.T
            #inv = self.S.inverse()

        except AssertionError:
            inv = self.S.inverse()

        u = u if self.offset is None else u - self.offset
        # if self.integrated_hints:
        #     for index, (S, c, gamma) in enumerate(self.integrated_hints):
        #         u = (u*S.T)[0, 1:]

        norm = scal((u - self.mu) * inv * (u - self.mu).T)
        return RR(norm)

    def test_primitive_dual(self, V, action):
        if self.B is None:
            self.B = dual_basis(self.D)

        W = V * self.B.T
        den = lcm([x.denominator() for x in W[0]])
        num = gcd([x for x in W[0] * den])
        assert den == 1

        if num == 1:
            return True
        if action == "warn":
            logging("non-primitive (factor %d)." %
                    num, style="WARNING", newline=False)
            return True
        elif action == "reject":
            raise RejectedHint("non-primitive (factor %d)." % num)

        raise InvalidHint("non-primitive (factor %d)." % num)

    def mean_update(self, offset):
        self.mu += offset*self.embedded_instance.get_primal_basis()    

    @scale_ellipsoid_prior(scaling_factor=1)
    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"],
                              invalidates=["primal"])
    def apply_perfect_hints(self):
        # Use ellipsoid Hyperplane intersection
        if not self.integrated_perfect_hints:
            raise InvalidHint("No perfect hints integrated!")
 
        V_e = self.integrated_perfect_hints['V_e']
        V = self.integrated_perfect_hints['V']
        gamma = self.integrated_perfect_hints['gamma']
        # Scale V and gamma to be integer
        denom = lcm([x.denominator() for x in V_e.augment(gamma.T).list()])
        V_e = matrix(ZZ, V_e*denom)
        gamma = matrix(ZZ, gamma*denom)
        

        # Solve diophantine system to produce offset
        self.offset = solve_diophantine_system_LWE(V_e, gamma, self.embedded_instance)
        # Convert offset on (e || s) to offset on (c || s)
        b = self.embedded_instance.b
        q = self.embedded_instance.q
        n = self.embedded_instance.n
        self.offset *= self.embedded_instance.get_dual_basis().T
        self.offset -= concatenate([b/q, zero_matrix(1, n)])

        # Intersect lattice with combined perfect hint system
        self.D = lattice_orthogonal_section(self.D, V)

        self.mu -= self.offset
        self.S = round_matrix_to_rational(self.S)
        self.mu = round_matrix_to_rational(self.mu)

        # Re-enable volume calibration
        self.calibrate_volume = True
       
    @scale_ellipsoid_prior(scaling_factor=1)
    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"],
                              invalidates=["primal"])
    def integrate_perfect_hint(self, v, l):
        # Convert hint into (e || s) coordinate space.
        v_e, l_e = self.convert_hint_c_to_e(v, l)
        
        # Check if hints are already integrated. Combine hint vectors
        # into matrix and concat target values. Store (e || s) representation
        if self.integrated_perfect_hints:
            V_e = self.integrated_perfect_hints['V_e']
            V = self.integrated_perfect_hints['V']
            gamma = self.integrated_perfect_hints['gamma']

            self.integrated_perfect_hints['V_e'] = V_e.stack(v_e)
            self.integrated_perfect_hints['V'] = V.stack(v)
            self.integrated_perfect_hints['gamma'] = concatenate(gamma, [l_e])
        
        else:
            self.integrated_perfect_hints['V_e'] = v_e
            self.integrated_perfect_hints['V'] = v
            self.integrated_perfect_hints['gamma'] = vec([l_e])

        dim_ = self.expected_length

        num = scal(v * self.mu.T) - l
        denom = scal(v * self.S * v.T)
        norm = sqrt(scal(v * self.S * v.T))
        
        alpha = num / norm

        #print(f"Alpha:{alpha}")
        if alpha < -1 or alpha > 1:
            raise InvalidHint("Redundant hint! Cut outside ellipsoid!")

        if alpha*(-alpha) > 1 /dim_:
            return

        b = (1 / norm) * v * self.S.T

        # coeff2 = (dim_) / (dim_ - 1) * (1 - num*num/denom)
        coeff2 = (1 - num*num/denom)

        # self.D = lattice_orthogonal_section(self.D, v_c)
        self.expected_length -= 1
        self.mu -= num/denom * v * self.S.T
        self.S -= (self.S * v.T * v * self.S)/denom
        self.S *= coeff2
        
        # Disable volume calibration while rank(ellip) ~= rank(lattice)
        self.calibrate_volume = False

    @scale_ellipsoid_prior(scaling_factor="rank")
    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"], invalidates=["primal"])
    def integrate_modular_hint(self, v, l, k, smooth=True):
       raise NotImplementedError("Modular hints not yet supported") 

    @scale_ellipsoid_prior(scaling_factor="rank")
    @not_after_projections
    @hint_integration_wrapper(force=True)
    def integrate_approx_hint(self, v, l, variance, aposteriori=False):
        if variance < 0:
            raise InvalidHint("variance must be non-negative !")
        if variance == 0:
            raise InvalidHint("variance=0 : must use perfect hint !")
        
        if not aposteriori:
            VS = v * self.S
            mu2 = scal(self.mu * v.T)
            sigma2 = scal(VS * v.T) + variance
            self.mu += (l - mu2)/sigma2 * VS
            self.S -= (VS.T * VS) / sigma2
        else:
            V = concatenate(v, 0)
            VS = V * self.S
            if not scal(VS * VS.T):
                raise RejectedHint("0-Eigenvector of Σ forbidden,")

            den = scal(VS * V.T)
            self.mu += ((l - scal(self.mu * V.T)) / den) * VS
            self.S += (((variance - den) / den**2) * VS.T ) * VS

    @scale_ellipsoid_prior(scaling_factor=1)
    @not_after_projections
    @hint_integration_wrapper(force=True)
    def integrate_ineq_hint(self, v, bound):
        """
         <v, secret> <= bound
        See Eq (3.1.11), Eq. (3.1.12) in Lovasz's the Ellipsoid Method.
        """
        dim_ = self.dim()
        norm = sqrt(scal(v * self.S * v.T))
        # print(f"num: {scal(v*self.mu.T) - bound}, denom: {norm}")
        alpha = (scal(v * self.mu.T) - bound) / norm

        #print(f"Alpha:{alpha}")
        if alpha < -1 or alpha > 1:
            raise InvalidHint("Redundant hint! Cut outside ellipsoid!")

        if -1 <= alpha and alpha <= -1 /dim_:
            return
        
        b = (1 / norm) * v * self.S.T

        coeff = (1 + dim_ * alpha) / (dim_ + 1)
        coeff2 = (dim_ * dim_) / (dim_ * dim_ - 1) * (1 - alpha * alpha)

        self.mu -= coeff * b
        self.S -= (2 * coeff) / (1 + alpha) * b.T * b
        self.S *= coeff2


    @scale_ellipsoid_prior(scaling_factor=1)
    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["primal"], invalidates=["dual"])
    def integrate_central_symmetric_ineq_hint(self, v, bound): #integrate_central_ineq_hint
        """
        <v, center> - bound <= <v, secrets> <= <v, center> + bound
        See Eq (3.1.19), Eq (3.1.20), Eq. (3.1.7) in Lovasz's the Ellipsoid Method.
        """
        dim_ = self.dim()
        norm = sqrt(scal(v * self.S * v.T))
        alpha = -bound / norm

        if alpha < -1 / sqrt(dim_):
            raise InvalidHint("Redundant hint! Cut outside ellipsoid!")
        if alpha >= 0:
            raise InvalidHint("alpha cannot exceed 0!")
        
        b = (1 / norm) * v * self.S
        a2 = alpha * alpha
        coeff = (dim_ / (dim_ - 1)) * (1 - a2)

        self.S -= ((1 - (dim_ * a2)) / (1 - a2)) * b.T * b
        self.S *= coeff

        
    @scale_ellipsoid_prior(scaling_factor="rank")
    @not_after_projections
    @hint_integration_wrapper()
    def integrate_approx_hint_fulldim(self, center,
                                      covariance, aposteriori=False):
        # Using http://www.cs.columbia.edu/~liulp/pdf/linear_normal_dist.pdf
        # with A = Id
        if self.homogeneous:
            raise NotImplementedError()

        if not aposteriori:
            d = self.S.nrows() - 1
            if self.S.rank() != d or covariance.rank() != d:
                raise InvalidHint("Covariances not full dimensional")

            zero = vec(d * [0])
            F = (self.S + block4(covariance, zero.T, zero, vec([1]))).inverse()
            F[-1, -1] = 0
            C = concatenate(center, 1)

            self.mu += ((C - self.mu) * F) * self.S
            self.S -= self.S * F * self.S
        else:
            raise NotImplementedError()

    @scale_ellipsoid_prior(scaling_factor="rank")
    @hint_integration_wrapper(force=False,
                              requires=["primal"],
                              invalidates=["dual"])
    def integrate_short_vector_hint(self, v):
        # Check if projections happened yet. If not, perform partial isotropization
        if not self.partially_isotropized:
            lwe_basis = self.embedded_instance.get_primal_basis()
            self.mu = self.mu * lwe_basis
            self.u = self.u * lwe_basis
            self.S = lwe_basis.T * self.S * lwe_basis
            self.B = self.B * lwe_basis
            if self.offset:
                self.offset = self.offset * lwe_basis

            self.partially_isotropized = True

        V = v - v * self.PP

        if scal((V * self.S) * V.T) == 0:
            raise InvalidHint("Projects to 0")

        self.projections += 1
        PV = identity_matrix(V.ncols()) - projection_matrix(V)
        try:
            self.B = lattice_project_against(self.B, V)
        except ValueError:
            raise InvalidHint("Not in Λ")

        self.mu = self.mu * PV
        self.u = self.u * PV
        if self.offset:
            self.offset = self.offset * PV

        self.S = PV.T * self.S * PV

        self.PP += V.T * (V / scal(V * V.T))
    
    @scale_ellipsoid_prior(scaling_factor=1)
    @not_after_projections
    @hint_integration_wrapper(force=False)
    def integrate_combined_hint(self, mu, Sigma):
        try:
            self.mu, self.S = ellipsoid_intersection(self.mu, self.S, mu, Sigma)
    
        except ValueError:
            raise RejectedHint("Intersection doesn't exist!")


    @scale_ellipsoid_prior(scaling_factor="rank")
    def estimate_attack(self, probabilistic=False, tours=1, silent=False,
        ignore_lift_proba=False, lift_union_bound=False, number_targets=1, scale=False):
        """ Assesses the complexity of the lattice attack on the instance.
        Return value in Bikz
        """

        (Bvol, Svol, dvol) = self.volumes()
        dim_ = self.dim()

        if scale:
            coeff = self.expected_length / sqrt(dim_)
            dvol -= dim_ * ln(coeff) / 2

        beta, delta = compute_beta_delta(
            dim_, dvol, probabilistic=probabilistic, tours=tours, verbose=not silent,
            ignore_lift_proba=ignore_lift_proba, number_targets=number_targets, lift_union_bound=lift_union_bound)

        self.dvol = dvol
        self.delta = delta
        self.beta = beta

        if self.verbosity and not silent:
            self.logging("      Attack Estimation     ", style="HEADER")
            self.logging("ln(dvol)=%4.7f \t ln(Bvol)=%4.7f \t ln(Svol)=%4.7f \t"
                         % (dvol, Bvol, Svol) +
                         "δ(β)=%.6f" % compute_delta(beta),
                         style="DATA", priority=2)
            if delta is not None:
                self.logging("dim=%3d \t δ=%.6f \t β=%3.2f " %
                             (dim_, delta, beta), style="VALUE")
            else:
                self.logging("dim=%3d \t \t \t β=%3.2f " %
                             (dim_, beta), style="VALUE")

            self.logging("")

        return (beta, delta)

    @scale_ellipsoid_prior(scaling_factor="rank")
    def attack(self, beta_max=None, beta_pre=None, randomize=False, tours=1):
        """
        Run the lattice reduction to solve the DBDD instance.
        Return the (blocksize, solution) of a succesful attack,
        or (None, None) on failure
        """
        self.logging("      Running the Attack     ", style="HEADER")

        if self.B is None:
            self.B = dual_basis(self.D)


        # Apply adequate distortion
        denom = lcm([x.denominator() for x in self.B.list()])
        B = block4(self.B, matrix(self.B.nrows(), 1),
                    matrix(1, self.B.ncols()), matrix([1]))
        d = B.nrows()
        # S = self.S + self.mu.T * self.mu
        S = self.homogenize_S()

        _, Linv = square_root_inverse_degen(self.S, self.B)
        
        # Augment Linv to be homogeneous
        Linv = block4(Linv, zero_matrix(Linv.nrows(), 1),
                    -round_matrix_to_rational(self.mu)*Linv, matrix([1]))
        
        L = Linv.inverse()

        M = B * Linv
        # Make the matrix Integral
        denom = lcm([x.denominator() for x in M.list()])
        M = matrix(ZZ, M * denom)

        # Build the BKZ object
        G = GSO.Mat(IntegerMatrix.from_matrix(M), float_type=self.float_type)
        bkz = BKZReduction(G)
        if randomize:
            bkz.lll_obj()
            bkz.randomize_block(0, d, density=d / 4)
            bkz.lll_obj()

        # Find least common divisor of secret in case it isn't integer
        u_den = lcm([x.denominator() for x in self.u.list()]) if self.u is not None else 1

        if beta_pre is not None:
            self.logging("\rRunning BKZ-%d (until convergence)" %
                         beta_pre, newline=False)
            bkz.lll_obj()
            par = BKZ.Param(block_size=beta_pre, strategies=strategies)
            bkz(par)
            bkz.lll_obj()
        else:
            beta_pre = 2
        # Run BKZ tours with progressively increasing blocksizes
        for beta in range(beta_pre, B.nrows() + 1):
            self.logging("\rRunning BKZ-%d" % beta, newline=False)
            if beta_max is not None:
                if beta > beta_max:
                    self.logging("Failure ... (reached beta_max)",
                                 style="SUCCESS")
                    self.logging("")
                    return None, None

            if beta == 2:
                bkz.lll_obj()
            else:
                par = BKZ.Param(block_size=beta,
                                strategies=strategies, max_loops=tours)
                bkz(par)
                bkz.lll_obj()

            # Tries all 3 first vectors because of 2 NTRU parasite vectors
            for j in range(3):
                # Recover the tentative solution,
                # undo distorition, scaling, and test it
                v = vec(bkz.A[j])
                v = u_den * v * L / denom

                # solution for ellipsoidal embedding is dimension (n + m)
                solution = matrix(ZZ, v.apply_map(round))[0, :-1] / u_den

                if self.offset is not None:

                    solution_plus = solution + self.offset
                    solution_minus = solution - self.offset
                    if not self.check_solution(solution_plus) and not self.check_solution(solution_minus):
                        continue

                if self.offset is None and not self.check_solution(solution):
                    continue

                self.logging("Success !", style="SUCCESS")
                self.logging("")
                return beta, solution

        self.logging("Failure ...", style="FAILURE")
        self.logging("")
        return None, None

    def integrate_q_vectors(self, q, min_dim=0, report_every=1, indices=None):
        self.logging("      Integrating q-vectors     ", style="HEADER")
        Sd = self.S_diag()
        n = len(Sd)
        I = []
        J = []
        M = q * identity_matrix(n)
        it = 0
        verbosity = self.verbosity
        if indices is None:
            indices = range(n)
        while self.dim() > min_dim:
            # print("integrate_q_vectors", self.dim(), min_dim)
            if (it % report_every == 0) and report_every > 1:
                self.logging("[...%d]" % report_every, newline=False)
            Sd = self.S_diag()
            L = [(Sd[i], i) for i in indices if i not in I]
            if len(L) == 0:
                break
            _, i = max(L)
            I += [i]
            try:
                didit = self.integrate_short_vector_hint(
                    vec(M[i]), catch_invalid_hint=False)
                if not didit:
                    break
                J += [i]
            except InvalidHint as err:
                self.logging(str(err) + ", Invalid.",
                             style="REJECT", priority=1, newline=True)
            it += 1
            self.verbosity = verbosity if (it % report_every == 0) else 0
        self.verbosity = verbosity
        return [vec(M[i]) for i in J]
