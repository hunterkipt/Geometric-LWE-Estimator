from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction

load("../framework/load_strategies.sage")
load("../framework/DBDD_generic.sage")
load("../framework/proba_utils.sage")

DEBUG = True

class DBDD(DBDD_generic):
    """
    This class defines all the elements defining a DBDD instance with all
    the basis computations
    """

    def __init__(self, B, S, mu, embedded_instance, u=None, verbosity=1, homogeneous=False, float_type="ld", D=None, Bvol=None, circulant=False):
        """constructor that builds a DBDD instance from a lattice, mean, sigma
        and a target
        ;min_dim: Number of coordinates to find to consider the problem solved
        :B: Basis of the lattice
        :S: The Covariance matrix (Sigma) of the uSVP solution
        :mu: The expectation of the uSVP solution
        :u: The unique vector to be found (optinal, for verification purposes)
        :fp_type: Floating point type to use in FPLLL ("d, ld, dd, qd")
        """
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
        self.expected_length = RR(sqrt(self.S.trace()) + 1)
        self.projections = 0
        self.save = {"save": None}
        self.float_type = float_type
        self.estimate_attack(silent=True)
        self.circulant = circulant

    def dim(self):
        if self.B is not None:
            return self.B.nrows()
        else:
            return self.D.nrows()

    def S_diag(self):
        return [self.S[i, i] for i in range(self.S.nrows())]

    def volumes(self):
        if self.B is not None:
            Bvol = logdet(self.B * self.B.T) / 2
            B = self.B
        else:
            Bvol = -logdet(self.D * self.D.T) / 2
            B = dual_basis(self.D)

        S = self.S + self.mu.T * self.mu
        Svol = degen_logdet(S)
        dvol = Bvol - Svol / 2.
        return (Bvol, Svol, dvol)

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


    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"],
                              invalidates=["primal"])
    def integrate_perfect_hint(self, v, l):
        V = self.homogeneize(v, l)
        # assert V.ncols() == self.dim()
        VS = V * self.S
        den = scal(VS * V.T)
        if den == 0:
            raise RejectedHint("Redundant hint")

        self.D = lattice_orthogonal_section(self.D, V)

        num = self.mu * V.T
        self.mu -= (num / den) * VS
        num = VS.T * VS
        # print(num)
        self.S -= num / den

    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"],
                              invalidates=["primal"])
    def integrate_perfect_hint_ellip(self, v, l):
        V = self.homogeneize(v, l)     

        dim_ = self.dim()
        self.S *= dim_
        norm = sqrt(scal(V * self.S * V.T))
        
        alpha = (scal(V * self.mu.T)) / norm

        #print(f"Alpha:{alpha}")
        if alpha < -1 or alpha > 1:
            raise InvalidHint("Redundant hint! Cut outside ellipsoid!")

        if alpha*(-alpha) > 1 /dim_:
            return
        
        b = (1 / norm) * V * self.S.T

        coeff2 = (dim_ * dim_) / (dim_ * dim_ - 1) * (1 - alpha * alpha)

        self.D = lattice_orthogonal_section(self.D, V)
        self.expected_length -= 1
        self.mu -= alpha * b
        self.S -= b.T * b
        self.S *= coeff2
        self.S *= 1/dim_

    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"], invalidates=["primal"])
    def integrate_modular_hint(self, v, l, k, smooth=True):
        V = self.homogeneize(v, l)
        VS = V * self.S
        den = scal(VS * V.T)

        if den == 0:
            raise RejectedHint("Redundant hint")

        if not smooth:
            raise NotImplementedError()

        self.D = lattice_modular_intersection(self.D, V, k)

    @not_after_projections
    @hint_integration_wrapper(force=True)
    def integrate_approx_hint(self, v, l, variance, aposteriori=False):
        if variance < 0:
            raise InvalidHint("variance must be non-negative !")
        if variance == 0:
            raise InvalidHint("variance=0 : must use perfect hint !")
        # Only to check homogeneity if necessary
        self.homogeneize(v, l)

        if not aposteriori:
            V = self.homogeneize(v, l)
            VS = V * self.S
            d = scal(VS * V.T)
            center = scal(self.mu * V.T)
            coeff = (- center / (variance + d))
            self.mu += coeff * VS
            self.S -= (1 / (variance + d) * VS.T) * VS
        else:
            V = concatenate(v, 0)
            VS = V * self.S
            if not scal(VS * VS.T):
                raise RejectedHint("0-Eigenvector of Σ forbidden,")

            den = scal(VS * V.T)
            self.mu += ((l - scal(self.mu * V.T)) / den) * VS
            self.S += (((variance - den) / den**2) * VS.T ) * VS

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

    @hint_integration_wrapper(force=False,
                              requires=["primal"],
                              invalidates=["dual"])
    def integrate_short_vector_hint(self, v):
        V = self.homogeneize(v, 0)
        V -= V * self.PP

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
        self.S = PV.T * self.S * PV
        self.PP += V.T * (V / scal(V * V.T))


    def shape_LWE_approx_hint(self, A, b, m, q, D_s, D_e, error_exp=1):
        print("Kannan shaping for LWE hints starts.")
        _, s_s = average_variance(D_s)
        _, s_e = average_variance(D_e)
        s_c = n*s_s/12 
        #s_c_vec = ((A * A.T)*s_s/(q*q) + s_e/(q*q)).diagonal() 
        id_matrix = identity_matrix(m)
        for i in range(m):
            print(f"cut: {i}")
            vec_v = concatenate(id_matrix[i]/q, A[i]/q)
            # list_q_inverse = zeros(m)
            # list_q_inverse[i] -= 1/q
            # vec_q_inverse = vec(list_q_inverse)
            # vec_v = concatenate(vec_q_inverse, A[i]/q)
            self.integrate_approx_hint(vec_v,
                                        b[0][i]/q,
                                        s_c*error_exp, #s_c_vec[i]*error_exp
                                        aposteriori=False, estimate=False)
        print("Kannan shaping for LWE hints completes.")

    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["primal"], invalidates=["dual"])
    def integrate_central_symmetric_ineq_hint(self, v, l, bound):
        """
        V = [v, -l]
        <V, center> - bound <= <V, secrets> <= <V, center> + bound
        See Eq (3.1.19), Eq (3.1.20), Eq. (3.1.7) in Lovasz's the Ellipsoid Method.
        """
        V = self.homogeneize(v, l)

        """
        Rescaling the Covariance matrix to convert <= d to <=1:
        (Not efficient if consecutively using ellipsoid method.)
        """
        dim_ = self.dim()
        
        self.S *= dim_
        norm = sqrt(scal(V * self.S * V.T))
        alpha = -bound / norm

        if alpha < -1 / dim_:
            raise RejectedHint("Redundant hint! Cut outside ellipsoid!")
        if alpha >= 0:
            raise InvalidHint("alpha cannot exceed 0!")
        
        b = (1 / norm) * V * self.S
        a2 = alpha * alpha
        coeff = (dim_ / (dim_ - 1)) * (1 - a2)
        # self.S -= ((1 - (dim_ * a2)) / (1 - a2)) * b.T * b
        # self.S *= coeff



        self.S -= ((1 - (dim_ * a2)) / (1 - a2)) * b.T * b
        self.S *= coeff

        """
        Recaling the Covariance matrix to convert <= d to <=1:
        (Not efficient if consecutively using ellipsoid method.)
        """
        self.S *= 1/dim_

    @not_after_projections
    @hint_integration_wrapper(force=False, requires=["primal"], invalidates=["dual"])
    def integrate_ineq_hint(self, v, bound):
        """
         <v, secret> <= bound
        See Eq (3.1.11), Eq. (3.1.12) in Lovasz's the Ellipsoid Method.
        """
        dim_ = self.dim()

        S = self.S * dim_
        v = concatenate(v, vec([0]))
        norm = sqrt(scal(v * S * v.T))
        # print(f"num: {scal(v*self.mu.T) - bound}, denom: {norm}")
        alpha = (scal(v * self.mu.T) - bound) / norm

        #print(f"Alpha:{alpha}")
        if alpha < -1 or alpha > 1:
            raise InvalidHint("Redundant hint! Cut outside ellipsoid!")

        if -1 <= alpha and alpha <= -1 /dim_:
            return
        
        b = (1 / norm) * v * S.T

        coeff = (1 + dim_ * alpha) / (dim_ + 1)
        coeff2 = (dim_ * dim_) / (dim_ * dim_ - 1) * (1 - alpha * alpha)

        center = coeff * b
        S -= (2 * coeff) / (1 + alpha) * b.T * b
        S *= coeff2

        self.mu -= center
        # self.mu[0, -1] = 1
        self.S = S * 1/dim_

    @not_after_projections
    @hint_integration_wrapper(force=False)
    def integrate_combined_hint(self, mu, Sigma):
        dim_ = self.dim()
        S = self.S * dim_
        try:
            mu_prime, S_prime = ellipsoid_intersection(self.mu[0,:-1], S[:-1,:-1], mu, Sigma)
       
        except ValueError:
            raise RejectedHint("Intersection Doesn't Exist!")

        self.mu = concatenate(mu_prime, [1])
        self.S = (1/dim_) * block4(S_prime, zero_matrix(S_prime.nrows(), 1), 
                                zero_matrix(1, S_prime.nrows()), zero_matrix(1,1))

    #delete later
    def shape_LWE_approx_perfect_hint_with_c(self, A, b, c, m, q, D_s, D_e, error_exp=1):
        print("Kannan shaping for LWE hints_real c perfect starts.")
        _, s_s = average_variance(D_s)
        _, s_e = average_variance(D_e)
        # s_c = n*s_s/12 
        #s_c_vec = ((A * A.T)*s_s/(q*q) + s_e/(q*q)).diagonal() 
        id_matrix = identity_matrix(m)
        for i in range(m):
            print(f"cut: {i}")
            vec_v = concatenate(id_matrix[i]/q, A[i]/q)
            # list_q_inverse = zeros(m)
            # list_q_inverse[i] -= 1/q
            # vec_q_inverse = vec(list_q_inverse)
            # vec_v = concatenate(vec_q_inverse, A[i]/q)
            if c[0][i] == 0:
                self.integrate_perfect_hint(vec_v,
                                        b[0][i]/q),
                                        
            else:
                self.integrate_approx_hint(vec_v,
                                        b[0][i]/q,
                                        abs(c[0][i])*error_exp, #s_c_vec[i]*error_exp
                                        aposteriori=False, estimate=False)
        print("Kannan shaping for LWE hints_real c perfect completes.")
    #delete later
    def shape_LWE_approx_hint_with_c(self, A, b, c, m, q, D_s, D_e, error_exp=1, narrow_factor = 1):
        print("Kannan shaping for LWE approx hints_real c starts.")
        _, s_s = average_variance(D_s)
        _, s_e = average_variance(D_e)
        # s_c = n*s_s/12 
        #s_c_vec = ((A * A.T)*s_s/(q*q) + s_e/(q*q)).diagonal() 
        id_matrix = identity_matrix(m)
        for i in range(m):
            print(f"cut: {i}")
            vec_v = concatenate(id_matrix[i]/q, A[i]/q)
            # list_q_inverse = zeros(m)
            # list_q_inverse[i] -= 1/q
            # vec_q_inverse = vec(list_q_inverse)
            # vec_v = concatenate(vec_q_inverse, A[i]/q)
            if c[0][i] == 0:
                self.integrate_approx_hint(vec_v,
                                        b[0][i]/q,
                                        narrow_factor*error_exp, #s_c_vec[i]*error_exp
                                        aposteriori=False, estimate=False)
            else:
                self.integrate_approx_hint(vec_v,
                                        b[0][i]/q,
                                        abs(c[0][i])*error_exp, #s_c_vec[i]*error_exp
                                        aposteriori=False, estimate=False)
        print("Kannan shaping for LWE approx hints_real c completes.")
    
    #delete later
    def shape_LWE_ellipsoid_hint_with_c(self, A, b, c, m, q, D_s, D_e, error_exp=1, narrow_factor = 1):
        print("Kannan shaping for LWE ellipsoid hints_real c starts.")
        _, s_s = average_variance(D_s)
        _, s_e = average_variance(D_e)
        # s_c = n*s_s/12 
        #s_c_vec = ((A * A.T)*s_s/(q*q) + s_e/(q*q)).diagonal() 
        id_matrix = identity_matrix(m)
        for i in range(m):
            print(f"cut: {i}")
            vec_v = concatenate(id_matrix[i]/q, A[i]/q)
            # list_q_inverse = zeros(m)
            # list_q_inverse[i] -= 1/q
            # vec_q_inverse = vec(list_q_inverse)
            # vec_v = concatenate(vec_q_inverse, A[i]/q)
            if c[0][i] == 0:
                self.integrate_central_symmetric_ineq_hint(vec_v,
                                        b[0][i]/q,
                                        narrow_factor*error_exp, #s_c_vec[i]*error_exp
                                        estimate=False)
            else:
                self.integrate_central_symmetric_ineq_hint(vec_v,
                                        b[0][i]/q,
                                        abs(c[0][i])*error_exp, #s_c_vec[i]*error_exp
                                        estimate=False)
        print("Kannan shaping for LWE ellipsoid hints_real c completes.")

    def ellip_norm(self, u=None):
        if u is None:
            u = self.u

            if u is None:
                raise ValueError("Solution vector must exist to calculate norm")
        # DBDD_S = round_matrix_to_rational(self.S)
        try:
            inv = (self.S + self.mu.T*self.mu).inverse()

        except AssertionError:
            inv = DBDD_S.inverse()

        except ZeroDivisionError:
            inv = degen_inverse(self.S + self.mu.T*self.mu)
        norm = scal(u * inv * u.T)
        return RR(norm)

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
        B = self.B
        d = B.nrows()
        S = self.S + self.mu.T * self.mu
        L, Linv = square_root_inverse_degen(S, self.B)
        M = B * Linv

        # Make the matrix Integral
        denom = lcm([x.denominator() for x in M.list()])
        M = matrix(ZZ, M * denom)

        # Build the BKZ object
        G = GSO.Mat(IntegerMatrix.from_matrix(M), float_type=self.float_type)
        bkz = BKZReduction(G)
        if randomize:
            bkz.lll_obj()
            bkz.randomize_block(0, d, density=floor(d / 4))
            bkz.lll_obj()

        u_den = lcm([x.denominator() for x in self.u.list()])

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

        if DEBUG:
            print("Secret key:")
            print(self.u)
        basis = {}
        basis_vecs = []
        secret_vec = self.u
        success = False
        for beta in range(beta_pre, B.nrows() + 1):
            self.logging("\rRunning BKZ-%d" % beta, newline=False)
            if beta_max is not None:
                if beta > beta_max:
                    self.logging("Failure ... (reached beta_max)",
                                 style="SUCCESS")
                    self.logging("")
                    # return None, None
                    # return basis | { "outcome" : "FAILURE" }, secret_vec, basis_vecs
                    return -1, secret_vec, basis_vecs

            if beta == 2:
                bkz.lll_obj()
            else:
                par = BKZ.Param(block_size=beta,
                                strategies=strategies, max_loops=tours)
                bkz(par)
                bkz.lll_obj()

            if DEBUG:
                print("Secret key:")
                print(self.u)
                print("Secret key norm: ", float((self.u).norm()))
            # Stores full basis for either successful or final beta value
            basis["BKZ"] = beta
            secret_vec = self.u # for sage matrix export
            basis_vecs = []
            # Tries all 3 first vectors because of 2 NTRU parasite vectors
            for j in range(bkz.A.nrows):
                # Recover the tentative solution,
                # undo distorition, scaling, and test it
                v = vec(bkz.A[j])
                v = u_den * v * L / denom
                solution = matrix(ZZ, v.apply_map(round)) / u_den
                if DEBUG:
                    print(f"Solution {j}:")
                    print(solution)
                    print("Solution norm: ", float(solution.norm()))
                basis_vecs.append(solution) # for sage matrix export
                #with open("outvecs.txt", "a") as f:
                #    f.write(str(list(solution)))
                #    f.write("\n")
                #for val in list(solution):
                #    for i in val:
                #        if i == 0:
                #            print(".", end="")
                #        else:
                #            print(abs(i), end="")
                #    print()

                valid = self.check_solution(solution)
                success |= valid
                if not valid:
                    continue

                if not DEBUG:
                    continue
                self.logging("Success !", style="SUCCESS")
                self.logging("")
                # return beta, solution


            if not success:
                continue
            self.logging("Success !", style="SUCCESS")
            self.logging("")
            return basis["BKZ"], secret_vec, basis_vecs

        self.logging("Failure ...", style="FAILURE")
        self.logging("")
        # return None, None
        return -1, secret_vec, basis_vecs
