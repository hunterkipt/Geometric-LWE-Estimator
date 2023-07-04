reset()

load("../framework/LWE.sage")

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
def det_denom(s_s, s_e, s_eps, s_hs, s_he, n, t):
        outer_coeff = 4*ln(s_s) + 4*ln(s_e) + (4*t - 8)*ln(s_eps)

        inner_coeff_1 = (7/4)*t*(t-1)*(n**4)*(s_hs**4)*(s_he**4) + t*(n**2)*(s_eps**4)*((s_hs**4) / (s_e**4) + (s_he**4)/(s_s**4))
        inner_coeff_2 = (t*(t-1)*(n**2)*(s_hs**2)*(s_he**2) + t*n*(s_eps**2)*((s_hs**2 / s_e**2) + (s_he**2 / s_s**2)) + s_eps**4 / (s_s**2 * s_e**2))

        return (n/2)*(outer_coeff + ln(inner_coeff_1 + inner_coeff_2**2))

for param in params:
    print(f'Parameter set: n = {param["n"]}, target {param["target"]}-bit security')
    print(f'log(q) = {param["logq"]}, log(Delta) = {param["delta"]}')
    print("==========================================")

    # Calculate Volume of starting lattice and ellipsoid, use m*log(q) for bvol
    Bvol = param["m"] * (param["logq"]*ln(2))
    Svol_orig = RR(param["n"]*log(2/3) + param["m"]*log(3.2*3.2))

    dvol_orig = Bvol - Svol_orig / 2

    # Calculate BKZ Beta, delta for starting lattice
    beta_orig, delta_orig = compute_beta_delta(
            (param["m"]+param["n"]+1), dvol_orig, probabilistic=False, tours=1, verbose=0,
            ignore_lift_proba=False, number_targets=1, lift_union_bound=False)

    print(f"BKZ Beta Estimate (Initial): {beta_orig: .2f} bikz ~ {beta_orig*0.265: .2f} bits")

    # Calculate estimate after t micciancio style decryption hints
    adv_queries = 1000 # adv_queries = t
    stat_security = 30
    std_fresh = sqrt((4/3)*param["n"] + 1)*3.2

    std_1_mult = sqrt((7*param["n"]**3)*(3.2**4)*((2/3)**2)*2**(-2*param["delta"]) + 2**(-2*param["delta"])*param["n"]*(std_fresh**4 + (1/12)*3.2*3.2) + (param["n"]/18) + (1/12))
    
#     std_1_mult = sqrt((7*param["n"]**3)*(3.2**4)*((2/3)**2) + param["n"]*(std_fresh**4 + (1/12)*3.2*3.2) + (param["n"]/18) + (1/12))
    bits_fresh = (1/2)*log(param["n"]*(std_fresh**2 + (1/12)))/log(2) + log(Hc(0.0001, param["n"]))/log(2)

    # Calculate ciphertext noise estimate
    sigma_eps = sqrt(12*adv_queries)*2**(stat_security / 2)*std_fresh
    sigma_eps_bits = sqrt(12*adv_queries)*2**(stat_security / 2)*sqrt(bits_fresh)

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
    denom_1_mult = det_denom(sqrt(2/3), 3.2,
                        std_1_mult, sqrt(4*param["n"]*(2/3)*3.2**2)*2**(-param["delta"]), 
                        sqrt(4*param["n"]*3.2**4)*2**(-param["delta"]), param["n"], adv_queries)

#     denom_1_mult = det_denom(sqrt(2/3), 3.2,
                        # std_1_mult, sqrt(4*param["n"]*(2/3)*3.2**2), 
                        # sqrt(4*param["n"]*3.2**4), param["n"], adv_queries)

    # Calculate (expected) volume for ellipsoid after t hints
    Svol_t_hints = RR(num - denom)
    Svol_t_hints_bits = RR(num_bits - denom_bits)
    Svol_t_hints_fresh = RR(num_fresh - denom_fresh)
    Svol_t_hints_1_mult = RR(num_1_mult - denom_1_mult)


    dvol_t_hints = Bvol - Svol_t_hints / 2
    dvol_t_hints_bits = Bvol - Svol_t_hints_bits / 2
    dvol_t_hints_fresh = Bvol - Svol_t_hints_fresh / 2
    dvol_t_hints_1_mult = Bvol - Svol_t_hints_1_mult / 2

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
    
    print(f"BKZ Beta Estimate ({adv_queries} Hint, statistical noise flooding): {beta_t_hints: .2f} bikz ~ {beta_t_hints*0.265: .2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, statistical noise flooding measured in bits): {beta_t_hints_bits: .2f} bikz ~ {beta_t_hints_bits*0.265 :.2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, noise flooding variance = rho_fresh): {beta_t_hints_fresh: .3f} bikz ~ {beta_t_hints_fresh*0.265: .2f} bits")
    print(f"BKZ Beta Estimate ({adv_queries} Hint, 1 Multiplication: sigma_eps = ct_noise + squared terms): {beta_t_hints_1_mult: .3f} bikz ~ {beta_t_hints_1_mult*0.265: .2f} bits")
    print("")

