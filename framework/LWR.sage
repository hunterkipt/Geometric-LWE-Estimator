
from random import randint

load("../framework/LWE_generic.sage")

class LWR(LWE_generic):
    """
    This class is designed to hold a traditional LWE instance (i.e. sA^T + e = b)
    """

    def __init__(self, n, p, q, m, D_s, verbosity=1, A=None, s=None):
        """
        Constructor that builds an LWE instance
        :n: (integer) size of the secret s
        :q: (integer) modulus
        :m: (integer) size of the error e
        :D_e: distribution of the error e (dictionary form)
        :D_s: distribution of the secret s (dictionary form)
        """

        # Draw random samples if A, s, e not provided.
        if A is None:
            A = matrix([[randint(0, q) for _ in range(n)] for _ in range(m)])

        if s is None:
            s = vec([draw_from_distribution(D_s) for _ in range(n)])

        # Create LWR specific parameters
        self.D_e = build_mod_switching_error_law(q, p)
        self.b = (q / p) * ((p / q) * s * A.T).apply_map(lambda x: x.round(mode='down'))
        self.b = self.b.apply_map(lambda x: x.round()) 
        self.e_vec = (self.b - s * A.T) 
        self.b = self.b % q 

        assert (s*A.T + self.e_vec) % q == self.b
        
        self.n=n
        self.q=q
        self.m=m
        self.D_s=D_s
        self.verbosity=verbosity
        self.A=A
        self.s=s

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