from fpylll import *
from fpylll.algorithms.bkz2 import BKZReduction
import numpy as np

load("../framework/load_strategies.sage")
load("../framework/EBDD.sage")
load("../framework/proba_utils.sage")

class EBDD_dec_fail(EBDD):
    """
    This class defines all the elements defining a EBDD instance with all
    the basis computations
    """

    def __init__(self, B, S, mu, embedded_instance, u=None, verbosity=1, homogeneous=False, float_type="ld", D=None, Bvol=None, ellip_scale=1, calibrate_volume=False, circulant=False):
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
        super().__init__(B, S, mu, embedded_instance, u, verbosity, homogeneous, float_type, D, Bvol, ellip_scale, calibrate_volume, circulant)

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

        # Optimization here. Assume S is non-degenerate.
        Svol = degen_logdet(S, B, eigh=True, assume_full_rank=True) + self.dim()*ln(ellip_norm)
        dvol = Bvol - Svol / 2.
        return (Bvol, Svol, dvol)

    def ellip_norm(self):
        if self.u is None:
            raise InvalidArgument("Solution vector must exist to calculate norm")

        try:
            _, Linv = square_root_inverse_degen(self.S, self.B, assume_full_rank=True)
            inv = Linv*Linv.T
            #inv = self.S.inverse()

        except AssertionError:
            inv = self.S.inverse()

        u = self.u if self.offset is None else self.u - self.offset
        # if self.integrated_hints:
        #     for index, (S, c, gamma) in enumerate(self.integrated_hints):
        #         u = (u*S.T)[0, 1:]

        norm = scal((u - self.mu) * inv * (u - self.mu).T)
        return RR(norm)

    @scale_ellipsoid_prior(scaling_factor=1)
    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"],
                              invalidates=["primal"])
    def apply_perfect_hints(self):
        raise NotImplementedError("Optimized Embedding for Inequality/Combined Hints only. Please use the EBDD class for this functionality.")
       
    @scale_ellipsoid_prior(scaling_factor=1)
    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"],
                              invalidates=["primal"])
    def integrate_perfect_hint(self, v, l): 
        raise NotImplementedError("Optimized Embedding for Inequality/Combined Hints only. Please use the EBDD class for this functionality.")

    @scale_ellipsoid_prior(scaling_factor="rank")
    @not_after_projections
    @hint_integration_wrapper(force=True, requires=["dual"], invalidates=["primal"])
    def integrate_modular_hint(self, v, l, k, smooth=True):
       raise NotImplementedError("Modular hints not yet supported") 