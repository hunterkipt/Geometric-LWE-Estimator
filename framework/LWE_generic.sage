
from re import M

load("../framework/proba_utils.sage")
load("../framework/utils.sage")
load("../framework/DBDD_predict_diag.sage")
load("../framework/DBDD_predict.sage")
load("../framework/DBDD_optimized.sage")
load("../framework/DBDD.sage")
load("../framework/EBDD_dec_fail.sage")
load("../framework/EBDD.sage")
load("../framework/ntru.sage")


class LWE_generic:
    """
    This class defines the interface for all LWE-like instances supported by the framework.
    The class is designed to embed LWE, NTRU, etc... concrete instances into the various
    EBDD/DBDD instances
    """
    def __init__():
        raise NotImplementedError("Generic class not meant to be used directly.")

    def embed_into_DBDD(self, dbdd_class=DBDD):
        """
        Factory method for creating a DBDD instance from the underlying cryptographic instance.
        :n: (integer) size of the secret s
        :q: (integer) modulus
        :m: (integer) size of the error e
        :D_e: distribution of the error e (dictionnary form)
        :D_s: distribution of the secret s (dictionnary form)
        """

        # HIDING
        # if self.verbosity:
            # logging("     Build DBDD from LWE     ", style="HEADER")
            # logging("n=%3d \t m=%3d \t q=%d" % (self.n, self.m, self.q), style="VALUE")


        # define the mean and sigma of the instance
        if self.mean_e or self.mean_s:
            mu = vec(self.mean_e + self.mean_s + [1])
            mu = matrix(QQ, mu)

        else:
            mu_e, s_e = average_variance(self.D_e)
            mu_s, s_s = average_variance(self.D_s)
            mu = vec(self.m * [mu_e] + self.n * [mu_s] + [1])

        if self.Sigma_e or self.Sigma_s:
            S = diagonal_matrix(QQ, self.Sigma_e + self.Sigma_s + [0])

        else:
            S = diagonal_matrix(self.m * [s_e] + self.n * [s_s] + [0])

        # Define lattice bases from LWE parameters
        B = build_LWE_lattice(-self.A, self.q) # primal
        D = build_LWE_lattice(self.A/self.q, 1/self.q) # dual

        # A_cen = A.apply_map(recenter) #Make sure here A is centered.
        b_cen = self.b.apply_map(recenter)

        tar = concatenate([b_cen, [0] * self.n])
        B = kannan_embedding(B, tar)
        D = kannan_embedding(D, concatenate([-b_cen/self.q, [0] * self.n])).T

        # print(vec(D.T[640]) * B)

        if self.s is None or self.e_vec is None:
            u = None

        else:
            u = concatenate([self.e_vec, self.s, [1]])

        return dbdd_class(B, S, mu, self, u, verbosity=self.verbosity, D=D, Bvol=self.m*log(self.q))

    def embed_into_DBDD_optimized(self):
        """
        Factory method for creating an optimized DBDD instance from the underlying cryptographic instance.
        """
        return self.embed_into_DBDD(dbdd_class=DBDD_optimized)

    def embed_into_DBDD_predict(self):
        """
        Factory method for creating a (prediction only) DBDD instance from the underlying cryptographic instance.
        """
        return self.embed_into_DBDD(dbdd_class=DBDD_predict)

    def embed_into_DBDD_predict_diag(self):
        """
        Factory method for creating a (prediction only) DBDD instance from the underlying cryptographic instance.
        Note: The resulting DBDD instance only supports diagonal ellipsoid covariance for efficiency.
        """
        return self.embed_into_DBDD(dbdd_class=DBDD_predict_diag)

    def embed_into_EBDD(self, ebdd_class=EBDD):
        """
        Factory method for creating an ellipsoidal DBDD instance from the underlying cryptographic instance.
        :n: (integer) size of the secret s
        :q: (integer) modulus
        :m: (integer) size of the error e
        :D_e: distribution of the error e (dictionary form)
        :D_s: distribution of the secret s (dictionary form)
        """
        if self.verbosity:
            logging("     Build EBDD from LWE     ", style="HEADER")
            logging("n=%3d \t m=%3d \t q=%d" % (self.n, self.m, self.q), style="VALUE")
        
        # Define mean and variance of instance
        s_e = None
        s_s = None

        sigma_se = None
        mean_s = None
        mean_e = None

        if self.D_e is not None:
            _, s_e = average_variance(self.D_e)

        if self.D_s is not None:
            _, s_s = average_variance(self.D_s)

        if s_e == None:
            s_e = 1

        if s_s == None:
            s_s = 1

        if self.Sigma_s is not None and self.Sigma_e is not None:
            sigma_se = diagonal_matrix(QQ, self.Sigma_e + self.Sigma_s)
        elif self.Sigma_s is not None:
            sigma_se = diagonal_matrix(QQ, self.m*[s_e] + self.Sigma_s)
        elif self.Sigma_e is not None:
            sigma_se = diagonal_matrix(QQ, self.Sigma_e + self.n * [s_s])

        if self.mean_s is not None:
            mean_s = vec(self.mean_s)

        if self.mean_e is not None:
            mean_e = vec(self.mean_e)

        # Ensure A matrix is centered for ellipsoidal embedding

        q = self.q

        self.A = self.A.apply_map(recenter)

        # Compute modulus residual vector, c
        self.b = self.b.apply_map(recenter)

        if self.s is None or self.e_vec is None:
            self.c = None
            u = None

        else:
            self.c = (self.s * self.A.T + self.e_vec - self.b)/self.q
            u = concatenate([self.c, self.s])


        # Compute Kannan ellipsoid embedding
        mu, S = kannan_ellipsoid(self.A, self.b, self.q, s_s=s_s, s_e=s_e, homogeneous=False, Sigma_s_e=sigma_se, mean_s=mean_s, mean_e=mean_e)
        
        B = identity_matrix(self.n + self.m)
        return ebdd_class(B, S, mu, self, u, verbosity=self.verbosity, ellip_scale=1)

    def embed_into_EBDD_dec_fail(self):
        """
        Factory method for creating an EBDD instance that is optimized for decryption failures.
        Note: The resulting EBDD instance assumes the ellipsoid shape matrix is full-rank, so perfect hints aren't supported.
        """
        return self.embed_into_EBDD(ebdd_class=EBDD_dec_fail)
