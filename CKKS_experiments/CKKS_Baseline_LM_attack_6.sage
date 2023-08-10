reset()

load("../framework/LWE.sage")
import numpy as np
# Params Format
# target: Bit security level
# n, m: Dimensions of secret and error, respectively
# logq: The logarithm of the q parameter in base-2
# delta: The logarithm of the delta CKKS parameter in base-2
params = [
    {"target": 128,
     "n": 1024,
     "m": 1024,
     "logq": 25,
     "delta": 13
     },
    {"target": 192,
     "n": 1024,
     "m": 1024,
     "logq": 17,
     "delta": 9
     },
    {"target": 256,
     "n": 1024,
     "m": 1024,
     "logq": 13,
     "delta": 7
     },
    {"target": 128,
     "n": 2048,
     "m": 2048,
     "logq": 51,
     "delta": 26
     },
    {"target": 192,
     "n": 2048,
     "m": 2048,
     "logq": 35,
     "delta": 18
     },
    {"target": 256,
     "n": 2048,
     "m": 2048,
     "logq": 27,
     "delta": 14
     },
    {"target": 128,
     "n": 4096,
     "m": 4096,
     "logq": 101,
     "delta": 40
     },
    {"target": 192,
     "n": 4096,
     "m": 4096,
     "logq": 70,
     "delta": 36
     },
    {"target": 256,
     "n": 4096,
     "m": 4096,
     "logq": 54,
     "delta": 28
     },
    {"target": 128,
     "n": 8192,
     "m": 8192,
     "logq": 202,
     "delta": 40
     },
    {"target": 192,
     "n": 8192,
     "m": 8192,
     "logq": 141,
     "delta": 40
     },
    {"target": 256,
     "n": 8192,
     "m": 8192,
     "logq": 109,
     "delta": 40
     },
    {"target": 128,
     "n": 16384,
     "m": 16384,
     "logq": 411,
     "delta": 40
     },
    {"target": 192,
     "n": 16384,
     "m": 16384,
     "logq": 284,
     "delta": 40
     },
    {"target": 256,
     "n": 16384,
     "m": 16384,
     "logq": 220,
     "delta": 40
     },
    {"target": 128,
     "n": 32768,
     "m": 32768,
     "logq": 827,
     "delta": 40
     },
    {"target": 192,
     "n": 32768,
     "m": 32768,
     "logq": 571,
     "delta": 40
     },
    {"target": 256,
     "n": 32768,
     "m": 32768,
     "logq": 443,
     "delta": 40
     }
]

# Hc function in CKKS error estimation section
def Hc(alpha, N):
        return sqrt(- (log(1 - (1 - alpha)**(2/N))/log(2)))

# Calculate the denominator of the expectation of the volume resulting from integrating t hints.

# <RUI>: This comes from the estimates you wrote in extension_to_t_hints.pdf
def det_denom(s_s, s_e, s_eps, s_hs, s_he, n, t):
        outer_coeff = 4*ln(s_s) + 4*ln(s_e) + (4*t - 8)*ln(s_eps)

        inner_coeff_1 = (7/4)*t*(t-1)*(n**4)*(s_hs**4)*(s_he**4) + t*(n**2)*(s_eps**4)*((s_hs**4) / (s_e**4) + (s_he**4)/(s_s**4))
        inner_coeff_2 = (t*(t-1)*(n**2)*(s_hs**2)*(s_he**2) + t*n*(s_eps**2)*((s_hs**2 / s_e**2) + (s_he**2 / s_s**2)) + s_eps**4 / (s_s**2 * s_e**2))

        return (n/2)*(outer_coeff + ln(inner_coeff_1 + inner_coeff_2**2))

def expected_ellipsoid_norm(Sigma_norm, mu, sigma_noise, Q_1, c_1, Q_2, c_2, dim, gamma_1, gamma_2):

    mean_1 = float(0)
    mean_2 = float(0)
    
    mycount_1 = 0
    while mycount_1 < 2500:
        #Sample normal distribution of dimension dim
        x = np.random.normal(0, 1, (dim, 1))
        #Multiply by Sigma_norm, add mu
        y = Sigma_norm@x + mu
        #apply first quadratic form
        out_1 = (y-c_1).T@Q_1@(y-c_1)
        #subtract from gamma_1 to get alpha_1
        alpha_1 = out_1[0,0] - gamma_1[0,0]
        rand1 = random.random()
        if rand1 <= exp(-1*alpha_1**2/(2*sigma_noise**2)):
            mean_1 += out_1[0,0]
            mycount_1 += 1
            if mycount_1 % 500 == 0:
                print("mycount_1: ", mycount_1)

    mycount_2 = 0
    while mycount_2 < 2500:
        #Sample normal distribution of dimension dim
        x = np.random.normal(0, 1, (dim, 1))
        #Multiply by Sigma_norm, add mu
        y = Sigma_norm@x + mu
        #apply second quadratic form
        out_2 = (y-c_2).T@Q_2@(y-c_2)
        #subtract from gamma_2 to get alpha_2
        alpha_2 = out_2[0,0] - gamma_2[0,0]
        rand2 = random.random()
        if rand2 <= exp(-1*alpha_2**2/(2*sigma_noise**2)):
            mean_2 += out_2[0,0]
            mycount_2 += 1
            if mycount_2 % 500 == 0:
                print("mycount_2: ", mycount_2)

    mean_1 = mean_1/2500
    mean_2 = mean_2/2500
    return mean_1, mean_2


for param in params:
    print(f'Parameter set: n = {param["n"]}, target {param["target"]}-bit security')
    print(f'log(q) = {param["logq"]}, log(Delta) = {param["delta"]}')
    print("==========================================")

    # Calculate Volume of starting lattice and ellipsoid, use m*log(q) for bvol
    Bvol = param["m"] * (param["logq"]*ln(2))

    # Svol is the volume of the ellipsoid defined by the secret distribution
    # var(secret) = 2/3, variance(error)=3.2^2
    Svol_orig = RR(param["n"]*log(2/3) + param["m"]*log(3.2*3.2))
    dvol_orig = Bvol - Svol_orig / 2

    # Calculate BKZ Beta, delta for starting lattice
    beta_orig, delta_orig = compute_beta_delta(
            (param["m"]+param["n"]+1), dvol_orig, probabilistic=False, tours=1, verbose=0,
            ignore_lift_proba=False, number_targets=1, lift_union_bound=False)

    print(f"BKZ Beta Estimate (Initial): {beta_orig: .2f} bikz ~ {beta_orig*0.265: .2f} bits")

    # Calculate estimate after t micciancio style decryption hints
    # adv_queries = 1000 # adv_queries = t TODO: change this to 1 instead of 1000
    adv_queries = 1
    stat_security = 30
    #msg = 0
    msg = 2*float(3.2**2 + 2/3) * (param["n"])/float(2)
    std_fresh = sqrt((4/3)*param["n"] + 1)*3.2

    
    # Standard deviation of ciphertext error after 1 multiplication. Equivalent to \rho^2_{mult} from Ana's writeup.  
    #std_1_mult = sqrt((7*param["n"]**3)*(3.2**4)*((2/3)**2)*2**(-2*param["delta"]) + 2**(-2*param["delta"])*param["n"]*(std_fresh**4 + (1/12)*3.2*3.2) + (param["n"]/18) + (1/12))  
    std_1_mult = sqrt((7*param["n"]**3)*(3.2**4)*((2/3)**2) + param["n"]*(std_fresh**4 + (1/12)*3.2*3.2) + (param["n"]/18) + (1/12) + 2 * msg**2*std_fresh**2)  
    std_1_mult_2 = sqrt(param["n"]*(std_fresh**4 + (1/12)*3.2*3.2) + (param["n"]/18) + (1/12) + 2 * msg**2*std_fresh**2)  

    # Measurement of fresh ciphertext error variance in bits
    bits_fresh = (1/2)*log(param["n"]*(std_fresh**2 + (1/12)))/log(2) + log(Hc(0.0001, param["n"]))/log(2)

    # Calculate ciphertext noise estimates
    # Statistically secure fresh ciphertext error variance
    sigma_eps = sqrt(12*adv_queries)*2**(stat_security / 2)*std_fresh
    sigma_eps_bits = sqrt(12*adv_queries)*2**(stat_security / 2)*sqrt(bits_fresh)
    sigma_eps_1_mult = sqrt(12*adv_queries)*2**(stat_security / 2)*std_1_mult
    
    # <RUI>: This comes from the writeup on the initial structure of the expected determinant (Dana's original writeup)
    num = 2*param["n"]*adv_queries*log(sigma_eps) + 2*param["n"]*(log(sqrt(2/3)) + log(3.2))
    denom = det_denom(sqrt(2/3), 3.2, sigma_eps, 3.2, sqrt(2/3), param["n"], adv_queries)

    num_bits = 2*param["n"]*adv_queries*log(sigma_eps_bits) + 2*param["n"]*(log(sqrt(2/3)) + log(3.2))
    denom_bits = det_denom(sqrt(2/3), 3.2, sigma_eps_bits, 3.2, sqrt(2/3), param["n"], adv_queries)

    num_fresh = 2*param["n"]*adv_queries*log(std_fresh) + 2*param["n"]*(log(sqrt(2/3)) + log(3.2))
    denom_fresh = det_denom(sqrt(2/3), 3.2, std_fresh, 3.2, sqrt(2/3), param["n"], adv_queries)

    
    # q/delta makes volume of inital lattice decrease``
    Bvol_1_mult = param["m"] * (param["logq"]*ln(2) - param["delta"]*ln(2))
    
#     num_1_mult = 2*param["n"]*adv_queries*log(std_1_mult) + 2*param["n"]*(log(sqrt(2/3)*2**(-param["delta"])) + log(3.2*2**(-param["delta"])))
    
    num_1_mult = 2*param["n"]*adv_queries*log(std_1_mult) + 2*param["n"]*(log(sqrt(2/3)) + log(3.2))
    #denom_1_mult = det_denom(sqrt(2/3), 3.2,
                        #std_1_mult, msg*sqrt(4*param["n"]*(2/3)*3.2**2)*2**(-param["delta"]), 
                        #msg*sqrt(4*param["n"]*3.2**4)*2**(-param["delta"]), param["n"], adv_queries)

    #denom_1_mult = det_denom(sqrt(2/3), 3.2,
                        #std_1_mult, msg*sqrt(2/3 + 3.2**2) + sqrt(4*param["n"]*(2/3)*3.2**2), 
                        #msg*sqrt(2/3 + 3.2**2) + sqrt(4*param["n"]*3.2**4), param["n"], adv_queries)

    denom_1_mult = det_denom(sqrt(2/3), 3.2,
                        std_1_mult, msg*sqrt(2 * 3.2**2) + sqrt(param["n"]*(2/3)*3.2**2), 
                        msg*sqrt(2 * 2/3) + sqrt(param["n"]*3.2**4), param["n"], adv_queries)

    # Calculate (expected) volume for ellipsoid after t hints
    Svol_t_hints = RR(num - denom)
    Svol_t_hints_bits = RR(num_bits - denom_bits)
    Svol_t_hints_fresh = RR(num_fresh - denom_fresh)
    Svol_t_hints_1_mult = RR(num_1_mult - denom_1_mult)


    dvol_t_hints = Bvol - Svol_t_hints / 2
    dvol_t_hints_bits = Bvol - Svol_t_hints_bits / 2
    dvol_t_hints_fresh = Bvol - Svol_t_hints_fresh / 2
    dvol_t_hints_1_mult = Bvol - Svol_t_hints_1_mult / 2



    # <RUI>: Insert your code here.
    # TODO: Calculate variance/standard deviation of variables
    expr_num = 200
    good = 200
    #avg_re_n1 = np.zeros((param["n"], 1))
    #avg_im_n1 = np.zeros((param["n"], 1))
    #avg_re_n2 = np.zeros((param["n"], 1))
    #avg_im_n2 = np.zeros((param["n"], 1))
    
    sum_exp_1 = 0
    sum_orig = 0
    sum_ns1 = np.zeros((1, 1))
    avg_im_n1 = np.zeros((1, 1))
    sum_ns2 = np.zeros((1, 1))
    avg_im_n2 = np.zeros((1, 1))
    count_no_int_0 = float(0)
    sum_no_int_0 = float(0)
    count_no_int_0_2 = float(0)
    sum_no_int_0_2 = float(0)
    count_no_int_1 = float(0)
    sum_no_int_1 = float(0)
    count_no_int_2 = float(0)
    sum_no_int_2 = float(0)
    count_no_int_3 = float(0)
    sum_no_int_3 = float(0)
    count_no_int_4 =float(0)
    sum_no_int_4 = float(0)
    sum_norm_real = float(0)
    sum_norm_im = float(0)
    sum_norm_int = float(0)
    count_norm_int = float(0)
    #S_re = np.random.normal(0, sqrt(2/3*param["n"]//2), (param["n"]//2, 1))
    #S_re = np.concatenate((S_re, S_re[::-1]))
    #S_im = np.random.normal(0, sqrt(2/3*param["n"]//2), (param["n"]//2, 1))
    #S_im = np.concatenate((S_im, S_im[::-1]))
    #E_re = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
    #E_re = np.concatenate((E_re, E_re[::-1]))
    #E_im = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
    #E_im = np.concatenate((E_im, E_im[::-1]))
    max_vol = -500.0
    mycount = 0
    for i in range(expr_num):
        mycount += 1
        S_re = np.random.normal(0, sqrt(2/3*param["n"]//2), (param["n"]//2, 1))
        S_re = np.concatenate((S_re, S_re[::-1]))
        S_im = np.random.normal(0, sqrt(2/3*param["n"]//2), (param["n"]//2, 1))
        S_im = np.concatenate((S_im, S_im[::-1]))
        E_re = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
        E_re = np.concatenate((E_re, E_re[::-1]))
        E_im = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
        E_im = np.concatenate((E_im, E_im[::-1]))
        V_re = np.random.normal(0, sqrt(2/3*param["n"]//2), (param["n"]//2, 1))
        V_re = np.concatenate((V_re, V_re[::-1]))
        V_im = np.random.normal(0, sqrt(2/3*param["n"]//2), (param["n"]//2, 1))
        V_im = np.concatenate((V_im, V_im[::-1]))
        V_re_prime = np.random.normal(0, sqrt(2/3*param["n"]//2), (param["n"]//2, 1))
        V_re_prime = np.concatenate((V_re_prime, V_re_prime[::-1]))
        V_im_prime = np.random.normal(0, sqrt(2/3*param["n"]//2), (param["n"]//2, 1))
        V_im_prime = np.concatenate((V_im_prime, V_im_prime[::-1]))
        #E_0_re = np.random.normal(0, sqrt(3.2**2*param["n"]/2 + msg**2*param["n"]/2), (param["n"]//2, 1))
        E_0_re = np.random.normal(msg, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
        E_0_re = np.concatenate((E_0_re, E_0_re[::-1]))
        E_0_im = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
        #E_0_im = np.random.normal(0, sqrt(3.2**2*param["n"]/2 + msg**2*param["n"]/2), (param["n"]//2, 1))
        E_0_im = np.concatenate((E_0_im, E_0_im[::-1]))
        E_1_re = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1)) 
        E_1_re = np.concatenate((E_1_re, E_1_re[::-1]))
        E_1_im = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
        E_1_im = np.concatenate((E_1_im, E_1_im[::-1]))
        E_0_re_prime = np.random.normal(msg, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
        #E_0_re_prime = np.random.normal(0, sqrt(3.2**2*param["n"]/2 + msg**2*param["n"]/2), (param["n"]//2, 1))
        E_0_re_prime = np.concatenate((E_0_re_prime, E_0_re_prime[::-1]))
        E_0_im_prime = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
        #E_0_im_prime = np.random.normal(0, sqrt(3.2**2*param["n"]/2 + msg**2*param["n"]/2), (param["n"]//2, 1))
        E_0_im_prime = np.concatenate((E_0_im_prime, E_0_im_prime[::-1]))
        E_1_re_prime = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1)) 
        E_1_re_prime = np.concatenate((E_1_re_prime, E_1_re_prime[::-1]))
        E_1_im_prime = np.random.normal(0, 3.2*sqrt(param["n"]//2), (param["n"]//2, 1))
        E_1_im_prime = np.concatenate((E_1_im_prime, E_1_im_prime[::-1]))
        # Plug the variables into the quadratic form equation
        #Q_re = np.zeros((param["n"], 4, 4))
        #Q_im = np.zeros((param["n"], 4, 4))
        Q_re = np.zeros((1, 4, 4))
        Q_im = np.zeros((1, 4, 4))
        #for j in range(param["n"]//2):
        for j in range(1):
            Q_re[j,0,0] = E_1_re[j]*E_1_re_prime[j] - E_1_im[j]*E_1_im_prime[j]
            Q_im[j,0,0] = -(E_1_re[j]*E_1_im_prime[j] + E_1_im[j]*E_1_re_prime[j])
            Q_re[j,1,1] = E_1_im[j]*E_1_im_prime[j] - E_1_re[j]*E_1_re_prime[j]
            Q_im[j,1,1] = E_1_im[j]*E_1_re_prime[j] + E_1_re[j]*E_1_im_prime[j]
            Q_re[j,0,1] = -(E_1_im[j]*E_1_re_prime[j] + E_1_re[j]*E_1_im_prime[j])
            Q_re[j,1,0] = Q_re[j,0,1]
            Q_im[j,0,1] = E_1_im[j]*E_1_im_prime[j] - E_1_re[j]*E_1_re_prime[j]
            Q_im[j,1,0] = Q_im[j,0,1]
            Q_re[j,2,2] = V_re[j]*V_re_prime[j] - V_im[j]*V_im_prime[j]
            Q_im[j,2,2] = -(V_re[j]*V_im_prime[j] + V_im[j]*V_re_prime[j])
            Q_re[j,3,3] = V_im[j]*V_im_prime[j] - V_re[j]*V_re_prime[j]
            Q_im[j,3,3] = V_im[j]*V_re_prime[j] + V_re[j]*V_im_prime[j]
            Q_re[j,2,3] = -(V_im[j]*V_re_prime[j] + V_re[j]*V_im_prime[j])
            Q_re[j,3,2] = Q_re[j,2,3]
            Q_im[j,2,3] = V_im[j]*V_im_prime[j] - V_re[j]*V_re_prime[j]
            Q_im[j,3,2] = Q_im[j,2,3]
            Q_re[j,0,2] = (V_re[j]*E_1_re_prime[j] - V_im[j]*E_1_im_prime[j] + V_re_prime[j]*E_1_re[j] - V_im_prime[j]*E_1_im[j])/2  
            Q_re[j,2,0] = Q_re[j,0,2]
            Q_im[j,0,2] = -(V_re[j]*E_1_im_prime[j] + V_im[j]*E_1_re_prime[j] + V_re_prime[j]*E_1_im[j] + V_im_prime[j]*E_1_re[j])/2  
            Q_im[j,2,0] = Q_im[j,0,2]
            Q_re[j,1,3] = -(V_re[j]*E_1_re_prime[j] - V_im[j]*E_1_im_prime[j] + V_re_prime[j]*E_1_re[j] - V_im_prime[j]*E_1_im[j])/2  
            Q_re[j,3,1] = Q_re[j,1,3]
            Q_im[j,1,3] = (V_re[j]*E_1_im_prime[j] + V_im[j]*E_1_re_prime[j] + V_re_prime[j]*E_1_im[j] + V_im_prime[j]*E_1_re[j])/2  
            Q_im[j,3,1] = Q_im[j,1,3]
            Q_re[j,0,3] = -(V_re[j]*E_1_im_prime[j] + V_im[j]*E_1_re_prime[j] + V_re_prime[j]*E_1_im[j] + V_im_prime[j]*E_1_re[j])/2  
            Q_re[j,3,0] = Q_re[j,0,3]
            Q_im[j,0,3] = -(V_re[j]*E_1_re_prime[j] - V_im[j]*E_1_im_prime[j] + V_re_prime[j]*E_1_re[j] - V_im_prime[j]*E_1_im[j])/2  
            Q_im[j,3,0] = Q_im[j,0,3]
            Q_re[j,1,2] = -(V_re[j]*E_1_re_prime[j] + V_im[j]*E_1_im_prime[j] + V_re_prime[j]*E_1_re[j] + V_im_prime[j]*E_1_im[j])/2  
            Q_re[j,2,1] = Q_re[j,1,2]
            Q_im[j,1,2] = -(V_re[j]*E_1_im_prime[j] - V_im[j]*E_1_re_prime[j] + V_re_prime[j]*E_1_im[j] - V_im_prime[j]*E_1_re[j])/2  
            Q_im[j,2,1] = Q_im[j,1,2]
        #for j in range(param["n"]//2+1, param["n"]):
        for j in range(1):
            Q_re[j,0,0] = E_1_re[j]*E_1_re_prime[j] - E_1_im[j]*E_1_im_prime[j]
            Q_im[j,0,0] = E_1_im[j]*E_1_re_prime[j] + E_1_re[j]*E_1_im_prime[j]
            Q_re[j,1,1] = E_1_im[j]*E_1_im_prime[j] - E_1_re[j]*E_1_re_prime[j]
            Q_im[j,1,1] = -(E_1_re[j]*E_1_im_prime[j] + E_1_im[j]*E_1_re_prime[j])
            Q_re[j,0,1] = -(E_1_im[j]*E_1_re_prime[j] + E_1_re[j]*E_1_im_prime[j])
            Q_re[j,1,0] = Q_re[j,0,1]
            Q_im[j,0,1] = E_1_re[j]*E_1_re_prime[j] - E_1_im[j]*E_1_im_prime[j]
            Q_im[j,1,0] = Q_im[j,0,1]
            Q_re[j,2,2] = V_re[j]*V_re_prime[j] - V_im[j]*V_im_prime[j]
            Q_im[j,2,2] = V_im[j]*V_re_prime[j] + V_re[j]*V_im_prime[j]
            Q_re[j,3,3] = V_im[j]*V_im_prime[j] - V_re[j]*V_re_prime[j]
            Q_im[j,3,3] = -(V_re[j]*V_im_prime[j] + V_im[j]*V_re_prime[j])
            Q_re[j,2,3] = -(V_im[j]*V_re_prime[j] + V_re[j]*V_im_prime[j])
            Q_re[j,3,2] = Q_re[j,2,3]
            Q_im[j,2,3] = V_re[j]*V_re_prime[j] - V_im[j]*V_im_prime[j]
            Q_im[j,3,2] = Q_im[j,2,3]
            Q_re[j,0,2] = (V_re[j]*E_1_re_prime[j] - V_im[j]*E_1_im_prime[j] + V_re_prime[j]*E_1_re[j] - V_im_prime[j]*E_1_im[j])/2  
            Q_re[j,2,0] = Q_re[j,0,2]
            Q_im[j,0,2] = (V_re[j]*E_1_im_prime[j] + V_im[j]*E_1_re_prime[j] + V_re_prime[j]*E_1_im[j] + V_im_prime[j]*E_1_re[j])/2  
            Q_im[j,2,0] = Q_im[j,0,2]
            Q_re[j,1,3] = -(V_re[j]*E_1_re_prime[j] - V_im[j]*E_1_im_prime[j] + V_re_prime[j]*E_1_re[j] - V_im_prime[j]*E_1_im[j])/2  
            Q_re[j,3,1] = Q_re[j,1,3]
            Q_im[j,1,3] = -(V_re[j]*E_1_im_prime[j] + V_im[j]*E_1_re_prime[j] + V_re_prime[j]*E_1_im[j] + V_im_prime[j]*E_1_re[j])/2  
            Q_im[j,3,1] = Q_im[j,1,3]
            Q_re[j,0,3] = -(V_re[j]*E_1_im_prime[j] + V_im[j]*E_1_re_prime[j] + V_re_prime[j]*E_1_im[j] + V_im_prime[j]*E_1_re[j])/2  
            Q_re[j,3,0] = Q_re[j,0,3]
            Q_im[j,0,3] = (V_re[j]*E_1_re_prime[j] - V_im[j]*E_1_im_prime[j] + V_re_prime[j]*E_1_re[j] - V_im_prime[j]*E_1_im[j])/2  
            Q_im[j,3,0] = Q_im[j,0,3]
            Q_re[j,1,2] = (V_re[j]*E_1_im_prime[j] + V_im[j]*E_1_re_prime[j] + V_re_prime[j]*E_1_im[j] + V_im_prime[j]*E_1_re[j])/2  
            Q_re[j,2,1] = Q_re[j,1,2]
            Q_im[j,1,2] = -(V_re[j]*E_1_re_prime[j] - V_im[j]*E_1_im_prime[j] + V_re_prime[j]*E_1_re[j] - V_im_prime[j]*E_1_im[j])/2  
            Q_im[j,2,1] = Q_im[j,1,2]

        # Plug the variables into the x vector
        #x = np.zeros((param["n"],4,1))
        x = np.zeros((1,4,1))
        #for j in range(param["n"]):
        for j in range(1):
            x[j,0,0] = S_re[j]
            x[j,1,0] = S_im[j]
            x[j,2,0] = E_re[j]
            x[j,3,0] = E_im[j]

        #L_re = np.zeros((param["n"], 4, 1))
        #L_im = np.zeros((param["n"], 4, 1))
        L_re = np.zeros((1, 4, 1))
        L_im = np.zeros((1, 4, 1))
        #for j in range(param["n"]//2):
        for j in range(1):
            L_re[j,0,0] = E_0_re[j]*E_1_re_prime[j] - E_0_im[j]*E_1_im_prime[j] + E_0_re_prime[j]*E_1_re[j] - E_0_im_prime[j]*E_1_im[j] 
            L_re[j,1,0] = -(E_0_re[j]*E_1_im_prime[j] + E_0_im[j]*E_1_re_prime[j] + E_0_re_prime[j]*E_1_im[j] + E_0_im_prime[j]*E_1_re[j]) 
            L_re[j,2,0] = V_re[j]*E_0_re_prime[j] - V_im[j]*E_0_im_prime[j] + V_re_prime[j]*E_0_re[j] - V_im_prime[j]*E_0_im[j] 
            L_re[j,3,0] = -(V_re[j]*E_0_im_prime[j] + V_im[j]*E_0_re_prime[j] + V_re_prime[j]*E_0_im[j] + V_im_prime[j]*E_0_re[j]) 

            L_im[j,0,0] = -(E_0_re[j]*E_1_re_prime[j] + E_0_im[j]*E_1_im_prime[j] + E_0_re_prime[j]*E_1_re[j] + E_0_im_prime[j]*E_1_im[j]) 
            L_im[j,1,0] = -(E_0_re[j]*E_1_im_prime[j] - E_0_im[j]*E_1_re_prime[j] + E_0_re_prime[j]*E_1_im[j] - E_0_im_prime[j]*E_1_re[j]) 
            L_im[j,2,0] = -(V_re[j]*E_0_im_prime[j] + V_im[j]*E_0_re_prime[j] + V_re_prime[j]*E_0_im[j] + V_im_prime[j]*E_0_re[j])  
            L_im[j,3,0] = V_re[j]*E_0_re_prime[j] - V_im[j]*E_0_im_prime[j] + V_re_prime[j]*E_0_re[j] - V_im_prime[j]*E_0_im[j] 
                
        #for j in range(param["n"]//2, param["n"]):
        for j in range(1):
            L_re[j,0,0] = E_0_re[j]*E_1_re_prime[j] - E_0_im[j]*E_1_im_prime[j] + E_0_re_prime[j]*E_1_re[j] - E_0_im_prime[j]*E_1_im[j] 
            L_re[j,1,0] = -(E_0_re[j]*E_1_im_prime[j] + E_0_im[j]*E_1_re_prime[j] + E_0_re_prime[j]*E_1_im[j] + E_0_im_prime[j]*E_1_re[j])  
            L_re[j,2,0] = V_re[j]*E_0_re_prime[j] - V_im[j]*E_0_im_prime[j] + V_re_prime[j]*E_0_re[j] - V_im_prime[j]*E_0_im[j] 
            L_re[j,3,0] = -(V_re[j]*E_0_im_prime[j] + V_im[j]*E_0_re_prime[j] + V_re_prime[j]*E_0_im[j] + V_im_prime[j]*E_0_re[j]) 

            L_im[j,0,0] = E_0_re[j]*E_1_re_prime[j] + E_0_im[j]*E_1_re_prime[j] + E_0_re_prime[j]*E_1_im[j] + E_0_im_prime[j]*E_1_re[j] 
            L_im[j,1,0] = -(E_0_re[j]*E_1_re_prime[j] - E_0_im[j]*E_1_im_prime[j] + E_0_re_prime[j]*E_1_re[j] - E_0_im_prime[j]*E_1_im[j])  
            L_im[j,2,0] = V_re[j]*E_0_im_prime[j] + V_im[j]*E_0_re_prime[j] + V_re_prime[j]*E_0_im[j] + V_im_prime[j]*E_0_re[j] 
            L_im[j,3,0] = -(V_re[j]*E_0_re_prime[j] - V_im[j]*E_0_im_prime[j] + V_re_prime[j]*E_0_re[j] - V_im_prime[j]*E_0_im[j]) 
        # calculate mu and c for (x-mu_re)^T*Q_re*(x-mu_re) + c_re + noise = gamma_re
        #mu_re = np.zeros((param["n"], 4, 1))
        #c_re = np.zeros((param["n"], 1))
        #mu_im = np.zeros((param["n"], 4, 1))
        #c_im = np.zeros((param["n"], 1))
        #Sigma_re = np.zeros((param["n"], 4, 4))
        #Sigma_im = np.zeros((param["n"], 4, 4))
        #new_Sigma_re_ns1 = np.zeros((param["n"], 4, 4))
        #new_Sigma_re_ns2 = np.zeros((param["n"], 4, 4))
        #new_Sigma_im_ns1 = np.zeros((param["n"], 4, 4))
        #new_Sigma_im_ns2 = np.zeros((param["n"], 4, 4))
        #Vol_re_ns1 = np.zeros((param["n"], 1))
        #Vol_re_ns2 = np.zeros((param["n"], 1))
        #Vol_im_ns1 = np.zeros((param["n"], 1))
        #Vol_im_ns2 = np.zeros((param["n"], 1))

        mu_re = np.zeros((1, 4, 1))
        c_re = np.zeros((1, 1))
        mu_im = np.zeros((1, 4, 1))
        c_im = np.zeros((1, 1))
        Sigma_re_ns1 = np.zeros((1, 4, 4))
        Sigma_re_ns2 = np.zeros((1, 4, 4))
        Sigma_im_ns1 = np.zeros((1, 4, 4))
        Sigma_im_ns2 = np.zeros((1, 4, 4))
        new_Sigma_ns1 = np.zeros((1, 4, 4))
        new_Sigma_ns1_2 = np.zeros((1, 4, 4))
        new_Sigma_ns2 = np.zeros((1, 4, 4))
        new_Sigma_ns2_2 = np.zeros((1, 4, 4))
        new_mu_ns1 = np.zeros((1, 4, 1))
        new_mu_ns1_2 = np.zeros((1, 4, 1))
        new_mu_ns2 = np.zeros((1, 4, 1))
        new_mu_ns2_2 = np.zeros((1, 4, 1))
        Vol_re_ns1 = np.zeros((1, 1))
        Vol_re_ns2 = np.zeros((1, 1))
        Vol_im_ns1 = np.zeros((1, 1))
        Vol_im_ns2 = np.zeros((1, 1))
        
        #for i in range(param["n"]):
        for i in range(1):
            #1. use sigma_eps_mult(ns1) to add noise to the quadratic form
            mu_re[i] =  -0.5 * np.linalg.pinv(Q_re[i]) @ L_re[i]
            c_re[i] = -mu_re[i].T@Q_re[i]@mu_re[i] 
            # use sigma_eps(ns1) to add noise to the real part of the quadratic form
            gamma_re_ns1 = (x[i]-mu_re[i]).T@Q_re[i]@(x[i]-mu_re[i]) + c_re[i] + np.random.normal(0, sqrt(param["n"]//2)*sigma_eps_1_mult, 1)   
            Sigma_elps = np.diag([4*param["n"]//2*2/3, 4*param["n"]//2*2/3, 4*param["n"]//2*3.2**2, 4*param["n"]//2*3.2**2])  
            Sigma_sqrt = np.diag([sqrt(param["n"]//2*2/3), sqrt(param["n"]//2*2/3), sqrt(param["n"]//2*3.2**2), sqrt(param["n"]//2*3.2**2)])
            # Get the inverse of matrix Sigma_elps
            Sigma_elps_inv = np.linalg.pinv(Sigma_elps)
            ellnorm = (x[i]).T@Sigma_elps_inv@(x[i])
            Vol_orig = -ln(matrix(Sigma_elps).det()) -4*ln(ellnorm)
            # imaginary part
            mu_im[i] = -0.5 * np.linalg.pinv(Q_im[i]) @ L_im[i]
            c_im[i] = -mu_im[i].T@Q_im[i]@mu_im[i]
            # use sigma_eps to add noise to the imaginary part
            gamma_im_ns1 = (x[i]-mu_im[i]).T@Q_im[i]@(x[i]-mu_im[i]) + c_im[i] + np.random.normal(0, sqrt(param["n"]/2)*sigma_eps_1_mult, 1)
            mean_1, mean_2 = expected_ellipsoid_norm(Sigma_sqrt, matrix(np.zeros((4,1))), sqrt(param["n"]/2)*sigma_eps_1_mult, Q_re[i], mu_re[i], Q_im[i], mu_im[i], 4, gamma_re_ns1 - c_re[i], gamma_im_ns1 - c_im[i])
            Sigma_re_ns1[i] = Q_re[i]/mean_1
            Sigma_im_ns1[i] = Q_im[i]/mean_2
            noint_flag = empty_ellipsoid_hyperboloid_intersection(mu1 = matrix(np.zeros((4,1))), Sigma1 = matrix(Sigma_elps_inv), mu2 = matrix(mu_re[i]), Sigma2 = matrix(Sigma_re_ns1[i]), lb = 10^(-40))                
            
            if noint_flag == 0:
                new_mu_ns1[i], new_Sigma_ns1[i] = ellipsoid_hyperboloid_intersection(mu1 = matrix(np.zeros((4,1))), Sigma1 = matrix(Sigma_elps_inv), mu2 = matrix(mu_re[i]), Sigma2 = matrix(Sigma_re_ns1[i]), lb = 10^(-40))
            else:
                new_mu_ns1[i] = matrix(np.zeros((4,1)))
                new_Sigma_ns1[i] = matrix(Sigma_elps_inv)
            noint_flag = empty_ellipsoid_hyperboloid_intersection(mu1 = matrix(np.zeros((4,1))), Sigma1 = matrix(Sigma_elps_inv), mu2 = matrix(mu_im[i]), Sigma2 = matrix(Sigma_im_ns1[i]), lb = 10^(-40))                
            if noint_flag == 0:
                new_mu_ns1_2[i], new_Sigma_ns1_2[i] = ellipsoid_hyperboloid_intersection(mu1 = matrix(np.zeros((4,1))), Sigma1 = matrix(Sigma_elps_inv), mu2 = matrix(mu_im[i]), Sigma2 = matrix(Sigma_im_ns1[i]), lb = 10^(-40))
            else:
                new_mu_ns1_2[i] = matrix(np.zeros((4,1)))
                new_Sigma_ns2_2[i] = matrix(Sigma_elps_inv)

            noint_flag = empty_ellipsoid_hyperboloid_intersection(mu1 = matrix(new_mu_ns1[i]), Sigma1 = matrix(new_Sigma_ns1[i]), mu2 = matrix(new_mu_ns1_2[i]), Sigma2 = matrix(new_Sigma_ns1_2[i]), lb = 10^(-40))          
            if noint_flag == 0:
                new_mu_ns1[i], new_Sigma_ns1[i] = ellipsoid_hyperboloid_intersection(mu1 = matrix(new_mu_ns1[i]), Sigma1 = matrix(new_Sigma_ns1[i]), mu2 = matrix(new_mu_ns1_2[i]), Sigma2 = matrix(new_Sigma_ns1_2[i]), lb = 10^(-40))
                Vol_1 = ln(matrix(new_Sigma_ns1[i]).det())
            else:
                if ln(matrix(new_Sigma_ns2[i]).det()) > ln(matrix(new_Sigma_ns2_2[i]).det()):
                    Vol_1 = ln(matrix(new_Sigma_ns1[i]).det())
                else:
                    Vol_1 = ln(matrix(new_Sigma_ns1_2[i]).det())


            
            # 2. use std_1_mult(ns2) to add noise to the quadratic form
            gamma_re_ns2  = (x[i]-mu_re[i]).T@Q_re[i]@(x[i]-mu_re[i]) + c_re[i] + np.random.normal(0, sqrt(param["n"]/2)*std_1_mult_2, 1)
            gamma_im_ns2 = (x[i]-mu_im[i]).T@Q_im[i]@(x[i]-mu_im[i]) + c_im[i] + np.random.normal(0, sqrt(param["n"]/2)*std_1_mult_2, 1)
            # Intersect with real and imaginary
            mean_1, mean_2 = expected_ellipsoid_norm(Sigma_sqrt, matrix(np.zeros((4,1))), sqrt(param["n"]/2)*std_1_mult_2, Q_re[i], mu_re[i], Q_im[i], mu_im[i], 4, gamma_re_ns2 - c_re[i], gamma_im_ns2 - c_im[i])
            Sigma_re_ns2[i] = Q_re[i]/mean_1
            Sigma_im_ns2[i] = Q_im[i]/mean_2
            print("Norm based on real: ", (x[i] - mu_re[i]).T@Sigma_re_ns2[i]@(x[i] - mu_re[i]))
            print("Norm based on im: ", (x[i] - mu_im[i]).T@Sigma_im_ns2[i]@(x[i] - mu_im[i]))
            noint_flag = empty_ellipsoid_hyperboloid_intersection(mu1 = matrix(np.zeros((4,1))), Sigma1 = matrix(Sigma_elps_inv), mu2 = matrix(mu_re[i]), Sigma2 = matrix(Sigma_re_ns2[i]), lb = 10^(-40))                
            ellnorm1 = 0
            if noint_flag == 1:
                print("!!!!!!!!!!!!!!")
            if noint_flag == 0:
                new_mu_ns2[i], new_Sigma_ns2[i] = ellipsoid_hyperboloid_intersection(mu1 = matrix(np.zeros((4,1))), Sigma1 = matrix(Sigma_elps_inv), mu2 = matrix(mu_re[i]), Sigma2 = matrix(Sigma_re_ns2[i]), lb = 10^(-40))
            else:
                new_mu_ns2[i] = matrix(np.zeros((4,1)))
                new_Sigma_ns2[i] = matrix(Sigma_elps_inv)
            noint_flag = empty_ellipsoid_hyperboloid_intersection(mu1 = matrix(np.zeros((4,1))), Sigma1 = matrix(Sigma_elps_inv), mu2 = matrix(mu_im[i]), Sigma2 = matrix(Sigma_im_ns2[i]), lb = 10^(-40))                
            if noint_flag == 1:
                print("!!!!!!!!!!!!!!")
            if noint_flag == 0:
                new_mu_ns2_2[i], new_Sigma_ns2_2[i] = ellipsoid_hyperboloid_intersection(mu1 = matrix(np.zeros((4,1))), Sigma1 = matrix(Sigma_elps_inv), mu2 = matrix(mu_im[i]), Sigma2 = matrix(Sigma_im_ns2[i]), lb = 10^(-40))
            else:
                new_mu_ns2_2[i] = matrix(np.zeros((4,1)))
                new_Sigma_ns2_2[i] = matrix(Sigma_elps_inv)

            noint_flag = empty_ellipsoid_hyperboloid_intersection(mu1 = matrix(new_mu_ns2[i]), Sigma1 = matrix(new_Sigma_ns2[i]), mu2 = matrix(new_mu_ns2_2[i]), Sigma2 = matrix(new_Sigma_ns2_2[i]), lb = 10^(-40))
            if noint_flag == 1:
                print("!!!!!!!!!!!!!!")            
            if noint_flag == 0:
                new_mu_ns2[i], new_Sigma_ns2[i] = ellipsoid_hyperboloid_intersection(mu1 = matrix(new_mu_ns2[i]), Sigma1 = matrix(new_Sigma_ns2[i]), mu2 = matrix(new_mu_ns2_2[i]), Sigma2 = matrix(new_Sigma_ns2_2[i]), lb = 10^(-40))
                Vol_2 = ln(matrix(new_Sigma_ns2[i]).det())
                ellnorm1 = (x[i]-new_mu_ns2[i]).T@new_Sigma_ns2[i]@(x[i]-new_mu_ns2[i])
            else:
                if ln(matrix(new_Sigma_ns2[i]).det()) > ln(matrix(new_Sigma_ns2_2[i]).det()):
                    ellnorm1 = (x[i]-new_mu_ns2[i]).T@new_Sigma_ns2[i]@(x[i]-new_mu_ns2[i])
                    Vol_2 = ln(matrix(new_Sigma_ns2[i]).det())
                else:
                    ellnorm1 = (x[i]-new_mu_ns2_2[i]).T@new_Sigma_ns2_2[i]@(x[i]-new_mu_ns2_2[i])
                    Vol_2 = ln(matrix(new_Sigma_ns2_2[i]).det())
            print("Ellipsoid norm Int: ", ellnorm1)
            print("Original Norm: ", ellnorm)
            print("Vol Int: ", Vol_2)
            if Vol_2 <  -ln(matrix(Sigma_elps).det()):
                Vol_2 = -ln(matrix(Sigma_elps).det())
                ellnorm1 = 0
                print("Revert 1 to original")
            if ellnorm1 > 6:
                print("XXXXXXXXXXXXXXXXXXXXXX")
                good = good -1
            else:
                sum_ns2[i] += exp(Vol_2)
            sum_ns1[i] += exp(Vol_1)
    Svol_avg_ns1 = ln(sum_ns1[0]/expr_num)
    Svol_avg_ns2 = ln(sum_ns2[0]/good)
    print("Ave Vol_ns1: ", Svol_avg_ns1)
    print("Ave Vol_ns2: ", Svol_avg_ns2)
    print("Original Vol: ", -ln(matrix(Sigma_elps).det()))
    dvol_ns1 = ( (Svol_avg_ns1 + 4*ln(4)) * param["n"]/2 + 2*param["n"]*ln(param["n"]/2) ) / 2 + Bvol
    dvol_ns2 = ( (Svol_avg_ns2 + 4*ln(4)) * param["n"]/2 + 2*param["n"]*ln(param["n"]/2) ) / 2 + Bvol
    
           

    # TODO: Use numpy.random.normal(mean, standard_deviation, (4,4)) to create a 4x4 random gaussian matrix for the quadratic form. Call matrix(<variable>) to convert it into a sage matrix.
    # TODO: Ensure the scaling of the quadratic form equation makes the righthand side equal to 1 (from Dana's messages)
    # TODO: Derive the 4x4 ellipsoid matrix for the canonical embedding of the secret/error block.
    # TODO: Use ellipsoid_hyperboloid_intersection function (with the ellipsoid corresponding to the secret/error block as the first parameters) to calculate the intersected ellipsoid.
    # TODO: Find the log determinant of the intersected ellipsoid according to Dana's instructions.
    # TODO: Calculate dvol as above (Bvol - Svol/2) DANA: Should this be Bvol + Svol/2?
    # TODO: Put dvol into compute_beta_delta function as below
    # TODO: Print out results
    # TODO: repeat above for different noise flooding levels. Do once for statistically secure noise flooding, and once where the noise flooding variance = \rho^2_mult.

    beta_ns1, delta_ns1 = compute_beta_delta(
            (param["m"]+param["n"]+1), dvol_ns1, probabilistic=False, tours=1, verbose=0,
            ignore_lift_proba=False, number_targets=1, lift_union_bound=False)
    beta_ns2, delta_ns2 = compute_beta_delta(
            (param["m"]+param["n"]+1), dvol_ns2, probabilistic=False, tours=1, verbose=0,
            ignore_lift_proba=False, number_targets=1, lift_union_bound=False)


    # Calculate BKZ Beta, delta for lattice after t hints
    beta_t_hints, delta_t_hints = compute_beta_delta(
            (param["m"]+param["n"]+1), dvol_t_hints, probabilistic=False, tours=1, verbose=0,
            ignore_lift_proba=False, number_targets=1, lift_union_bound=False)

    beta_t_hints_bits, delta_t_hints_bits = compute_beta_delta(
            (param["m"]+param["n"]+1), dvol_t_hints_bits, probabilistic=False, tours=1, verbose=0,
            ignore_lift_proba=False, number_targets=1, lift_union_bound=False)

    beta_t_hints_fresh, delta_t_hints_fresh = compute_beta_delta(
            (param["m"]+param["n"]+1), dvol_t_hints_fresh, probabilistic=False, tours=1, verbose=0,
            ignore_lift_proba=False, number_targets=1, lift_union_bound=False)

    beta_t_hints_1_mult, delta_t_hints_1_mult = compute_beta_delta(
            (param["m"]+param["n"]+1), dvol_t_hints_1_mult, probabilistic=False, tours=1, verbose=0,
            ignore_lift_proba=False, number_targets=1, lift_union_bound=False)
    print(f"BKZ Beta Estimate ({adv_queries} Hint, 1 Multiplication, statistical noise flooding(std=sigma_eps), msg = N*(s_s^2+s_e^2)): {beta_ns1: .2f} bikz ~ {beta_ns1*0.265: .2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, 1 Multiplication, noise flooding in Ana's paper, msg = N*(s_s^2+s_e^2)): {beta_ns2: .2f} bikz ~ {beta_ns2*0.265: .2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, statistical noise flooding): {beta_t_hints: .2f} bikz ~ {beta_t_hints*0.265: .2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, statistical noise flooding measured in bits): {beta_t_hints_bits: .2f} bikz ~ {beta_t_hints_bits*0.265 :.2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, noise flooding variance = rho_fresh): {beta_t_hints_fresh: .3f} bikz ~ {beta_t_hints_fresh*0.265: .2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, 1 Multiplication: sigma_eps = ct_noise + squared terms), msg = N*(s_s^2+s_e^2): {beta_t_hints_1_mult: .3f} bikz ~ {beta_t_hints_1_mult*0.265: .2f} bits")
    print("")

