from optparse import Values


load("../framework/instance_gen.sage")
load("../framework/geometry_twist.sage")
load("../framework/geometry.sage")
load("../framework/EBDD_non_homo.sage")
load("../framework/utils.sage")
load("../framework/proba_utils.sage")
load("../framework/LWE.sage")

#import math
#import numpy as np

RealNumber = RealField(200)
verbosity = False
integrate_ellipsoid = False

def calculate_l_k(beta, c_vector, sigma_1_squared, approximation):
    """
    Calculate the l and k values which minimize the determinant ratio for the given parameters
    :beta: the beta value to use
    :c_vector: the vector of c coordinates to use
    :sigma_1_squared: the LWE variance
    """
    valid = True
    p = len(c_vector)
    c_av = mean(c_vector)

    if approximation:
        part1 = 2*c_av*beta**2 - sigma_1_squared + 2*sigma_1_squared*beta - sigma_1_squared*beta**2
        surd = 4*c_av**2*beta**3 + sigma_1_squared**2 * (beta**4 - 4*beta**3 + 6*beta**2 - 4*beta + 1)
        denom = 2*c_av*beta**2 - 2*c_av*beta
        l = (part1 - math.sqrt(surd))/denom
    else:
        coeff3 = -beta*c_av + beta**2*c_av
        coeff2 = beta**2*sigma_1_squared - 2*beta*sigma_1_squared + sigma_1_squared + beta*c_av - 3*beta**2*c_av
        coeff1 = 2*beta*sigma_1_squared - 2*beta**2*sigma_1_squared + 3*beta**2*c_av
        coeff0 = beta**2*sigma_1_squared - beta**2*c_av

        x = PolynomialRing(RationalField(), 'x').gen()
        f = coeff3*x**3 + coeff2*x**2 + coeff1*x + coeff0
        roots = f.roots()
        
        x1 = roots[0][0]
        l = -1
        if x1 > 0 and x1 < 1:
            l = x1
        if len(roots) > 1 and roots[1][0] > 0 and roots[1][0] < 1:
            x2 = roots[1][0]
            if l == -1:
                l = x2
            else:
                k_prev, valid_prev = calculate_k(l, beta, c_av, sigma_1_squared)
                if valid_prev:
                    r_prev = calculate_r(l, k_prev, beta, p, False)
                k_x2, valid_x2 = calculate_k(x2, beta, c_av, sigma_1_squared)
                if valid_x2:
                    r_x2 = calculate_r(x2, k_x2, beta, p, False)
                if r_x2 < r_prev:
                    l = x2
        if len(roots) == 3 and roots[2][0] > 0 and roots[2][0] < 1:
            x3 = roots[2][0]
            if l == -1:
                l = x3
            else:
                k_prev, valid_prev = calculate_k(l, beta, c_av, sigma_1_squared)
                if valid_prev:
                    r_prev = calculate_r(l, k_prev, beta, p, False)
                k_x3, valid_x3 = calculate_k(x3, beta, c_av, sigma_1_squared)
                if valid_x3:
                    r_x3 = calculate_r(x3, k_x3, beta, p, False)
                if r_x3 < r_prev:
                    l = x3

        if l == -1:
            return 0, 0, False
    
    k, valid = calculate_k(l, beta, c_av, sigma_1_squared)

    return l, k, valid

def calculate_k(l, beta, c_av, sigma_1_squared):
    """
    Calculate the minimum k values for the given parameters
    :l: the minimum l value
    :beta: the beta value to use
    :c_av: the average of the vector of c coordinates
    :sigma_1_squared: the LWE variance
    """
    
    k = math.exp((-l * (1-l) * c_av)/(sigma_1_squared * (1 - l * (1 - 1/beta))))

    # True k value
    #k = (1-(l*(1 - l)*c_av*p)/(d * sigma_1_squared * (1 - l*(1 - 1/beta))))**(d/p)

    if k < 0:
        return 0, False
    if k > 1:
        return 0, False

    return k, True

def calculate_r(l, k, beta, p, approximation):
    """
    Calculate the minimum r value for the given values.
    The r value is the ratio of the volumes of the intersected and posterior ellipsoids
    :l: the l value for minimizing r
    :k: the k value for minimizing k
    :beta: the beta value
    :p: the number of coordinates in the current intersection
    """
    if approximation:
        r = (k * 1/(1 - (1 - 1/beta)*l))**p
    else:
        r = (k * 1/(1 - l))**p
    return r

def r_post_int(beta, c_vector, sigma_1_squared, approximation):
    """
    An upper bound for R_post/int. This is the function we are trying to optimize.
    This function computes l, k and the ratio
    :beta: the beta value to use
    :c_vector: the vector of c values for the coordinates of this intersection
    :sigma_1_squared: the variance of the prior distribution
    """

    l, k, valid = calculate_l_k(beta, c_vector, sigma_1_squared, approximation)

    p = len(c_vector)

    if not valid:
        return 0, False

    r = calculate_r(l, k, beta, p, approximation)

    return r, True

def calculate_c_vector(cs, betas, i, dim):
    """
    Calculates the vector of c values to use for the current beta value
    Outputs an array of c values for coordinates with large enough beta
    The array is sorted in descending order
    :cs: the full vector of mean coordinates for the posterior ellipsoid 
    :betas: the full vector of beta Values
    :i: the index of the beta value to use
    :dim: the dimension of the posterior ellipsoid
    """
    indices = []
    for j in range(dim):
        if betas[i] <= betas[j]:
            indices.append(j)
    c_vector = [cs[j]*cs[j] for j in indices]
    modified_indices = [x for _,x in sorted(zip(c_vector, indices), reverse=True)]
    c_vector.sort(reverse=True)
    return c_vector, modified_indices


def find_optimal_indices(beta_vector, c_vector, sigma_1_squared_value, approximation=False):
    """
    :beta_vector: the ratio of variances between prior and posterior ellipsoids for each coordinate
    :c_vector: the mean of the posterior elliposoid
    :sigma_1_squared_value: the variance of the prior ellipsoid
    """

    betas = beta_vector
    cs = c_vector
    sigma_1_squared = sigma_1_squared_value
    dim= len(betas)

    if verbosity:
        logging(f"n = {dim} sigma_1^2 = {sigma_1_squared}\n")

    r_max = 1
    max_i = 0
    max_p = 0

    r_min = 1
    min_i = 0
    min_p = 0

    if verbosity:
        logging("Calculating r_post/int values")
    for i in range(dim):
        if betas[i] <= 1:
            continue
        c_vector, _ = calculate_c_vector(cs, betas, i, dim)
        for p in range(1, len(c_vector) + 1):
            r, r_usable = r_post_int(betas[i], c_vector[:p], sigma_1_squared, approximation)
            if r > r_max and r_usable: 
                r_max = r
                max_i = i
                max_p = p
            if (r < r_min or r_min == -1) and r_usable: 
                r_min = r
                min_i = i
                min_p = p

    if verbosity:
        logging(f"Maximium: r = {r_max}, p = {max_p}, beta = {betas[max_i]}", )

    min_c_vector, indices_min = calculate_c_vector(cs, betas, min_i, dim)
    if verbosity:
        min_l = calculate_l_k(betas[min_i], min_c_vector[:min_p], sigma_1_squared)
        min_c_av = mean(min_c_vector[:min_p])
        logging(f"Minimum: r = {r_min}, c = {min_c_av}, p = {min_p}, beta = {betas[min_i]}, l, k = {min_l}", )
    
    return indices_min[:min_p]

def main():
    cs_betas = load("alpha_beta_values.sobj")

    find_optimal_indices(cs_betas[1], cs_betas[2], cs_betas[4])


# if __name__ == "__main__":
#     main()
