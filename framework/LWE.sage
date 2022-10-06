
from random import randint

load("../framework/LWE_generic.sage")

class LWE(LWE_generic):
    """
    This class is designed to hold a traditional LWE instance (i.e. sA^T + e = b)
    """

    def __init__(self, n, q, m, D_e, D_s, verbosity=1, A=None, b=None, s=None, e_vec=None):
        """
        Constructor that builds an LWE instance
        :n: (integer) size of the secret s
        :q: (integer) modulus
        :m: (integer) size of the error e
        :D_e: distribution of the error e (dictionary form)
        :D_s: distribution of the secret s (dictionary form)
        """

        # Draw random samples if A, s, e not provided. If b is provided, don't sample s and e.
        if A is None:
            A = matrix([[randint(0, q) for _ in range(n)] for _ in range(m)])

        if b is None:
            if s is None:
                s = vec([draw_from_distribution(D_s) for _ in range(n)])

            if e_vec is None:
                e_vec = vec([draw_from_distribution(D_e) for _ in range(m)])
            
            self.b = (s * A.T + e_vec) % q

        else:
            self.b = b
            
        self.n=n
        self.q=q
        self.m=m
        self.D_e=D_e
        self.D_s=D_s
        self.verbosity=verbosity
        self.A=A
        self.s=s
        self.e_vec=e_vec

    def get_primal_basis(self):
        """
        Generates the (primal) LWE lattice basis for {(x, y) | x + yA^t = b mod q}
        """
        # Define lattice bases from LWE parameters
        return build_LWE_lattice(-self.A, self.q) # primal

    def get_dual_basis(self):
        """
        Generates the (dual) LWE lattice basis for {(x, y) | x + yA^t = b mod q}
        """
        return build_LWE_lattice(self.A/self.q, 1/self.q).T