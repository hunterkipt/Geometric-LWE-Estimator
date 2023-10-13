reset()
load("../framework/LWE.sage")
import numpy as np

q = 3329
for k in range(20):
    name = 'results_exp_2_[(0,)]_0.75_' + str(k) + '.npz'
    data = np.load(name, allow_pickle=True)
    mydict = data['ntt_coeff_dist']
    secret = data['skpv']
    #print("Secret: ", secret)
    mydict2 = mydict.ravel()[0]
    logdet = 32*ln(q) - 64*ln(3.0/2.0)
    count = 0
    for j in range(64):
        mydict3 = mydict2[j][0]
        mysum = 0
        mymean = 0
        myvariance = 0
        for i in range(3329):
            mysum += mydict3[i-1664]
            mymean += (i-1664)*mydict3[i-1664]
            myvariance += (i-1664)**2*mydict3[i-1664]
        myvariance -= mymean**2
        if myvariance < .000115:
            count +=1
        else:
            logdet -= ln(myvariance)/2
        #print("Sum: ", mysum)
        #print("Mean: ", mymean)
        #print("Variance: ", myvariance)
        #print("Secret coordinate", secret[0,j])

    print("noise variance 0.75 and k = ", k)
    print("Logdet: ", float(logdet))
    print("Count: ", count)
    beta, delta = compute_beta_delta(192-count, logdet, probabilistic=False, tours=1, verbose=0, ignore_lift_proba=False, number_targets=1, lift_union_bound=False)
    print("Beta: ", beta)


#for k in range(20):
#    name = 'results_exp_2_[(0,)]_1.0_' + str(k) + '.npz'
#    data = np.load(name, allow_pickle=True)
#    mydict = data['ntt_coeff_dist']
#    mydict2 = mydict.ravel()[0]
#    logdet = 32*ln(q) - 64*ln(3.0/2.0)
#    count = 0
#    for j in range(32):
#        mydict3 = mydict2[2*j][0]
#        mysum = 0
#        mymean = 0
#        myvariance = 0
#        for i in range(3329):
#            mysum += mydict3[i-1664]
#            mymean += (i-1664)*mydict3[i-1664]
#            myvariance += (i-1664)**2*mydict3[i-1664]
#        myvariance -= mymean**2
#        if myvariance < .0011:
#            count +=1
#        else:
#            logdet -= ln(myvariance)/2
    #print("Sum: ", mysum)
#        print("Mean: ", mymean)
#        print("Variance: ", myvariance)

#    print("noise variance 1.0 and k = ", k)
#    print("Logdet: ", float(logdet))
#    print("Count: ", count)
#    beta, delta = compute_beta_delta(161-count, logdet, probabilistic=False, tours=1, verbose=0, ignore_lift_proba=False, number_targets=1, lift_union_bound=False)
#    print("Beta: ", beta)