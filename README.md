# A Sage Toolkit to attack and estimate the hardness of LWE with Side Information

This repository contains the artifacts associated to the article

Improved Estimation of LWE with Side Information via a Geometric Approach

The code is written for Sage 9.3 (Python 3 Syntax). The code is a fork of the `Leaky LWE Estimator` seen [here](https://github.com/lducas/leaky-LWE-Estimator).

## Organization
The library itself is to be found in the `framework` folder. 
`Sec5.2_validation` and `Sec6_applications` contain the code to reproduce our experiments.


## Details of the new API
Compared to the prior implementation, we have updated the way EBDD (and DBDD) instances are created to better reflect the structure of the problem and make the toolkit more easily extensible.

We have revised the front-end of the toolkit with the `LWE` and `LWR` classes. You first create an LWE or LWR instance, and then embed it into an EBDD (or DBDD) instance. We only provide the full-fledged implementation of an EBDD instance through the method `embed_into_EBDD()`. DBDD instances can be created through the methods `embed_into_DBDD()`, `embed_into_DBDD_optimized()`, `embed_into_DBDD_predict()`, and `embed_into_DBDD_predict_diag()`. Let us create a small LWE instance and estimate its security in bikz. The code should be run from the directories `Sec5.2_validation` or `Sec6_applications`.


```sage
sage: load("../framework/LWE.sage")
....: n = 70
....: m = n
....: q = 3301
....: D_s = build_centered_binomial_law(40)
....: D_e = D_s
....: lwe_instance = LWE(n, q, m, D_e, D_s)
....: ebdd = lwe_instance.embed_into_EBDD()
....: beta, delta = ebdd.estimate_attack()
```
```text
      Build EBDD from LWE      
 n= 70   m= 70   q=3301 
       Attack Estimation      
 dim=140         δ=1.012466      β=43.47  
```

Like the prior toolkit, our EBDD implementation contains an attack procedure that runs BKZ with an iteratively increasing block size. It then compares the recovered secret with the actual embedded secret.

```sage 
sage: secret = ebdd.attack()
```
```text
       Running the Attack      
Running BKZ-41   Success ! 
```

Here, the block size stopped at 41 while an average blocksize of 43.47 has been estimated.

## Integrating Hints into EBDD instances

Once the EBDD instances are created, integrating hints is mostly the same as in the prior toolkit. Keep in mind that EBDD instances are using a different coordinate space for the secret than DBDD instances. We provide a method `ebdd.convert_hint()` that converts hints on the `(e || s)` coordinate space to the `(c || s)` coordinate space. Once hints are in the correct coordinate space, the hints are integrated the same (except for perfect hints) as in the prior toolkit. Please see their [documentation](https://github.com/lducas/leaky-LWE-Estimator) for more details.

For our new types of hints (and perfect hints) we will document the procedures here.

### Integrating Inequality Hints

Inequality hints are integrated in the exact same way as previous hints. Here we simulate hints where we underestimate the value of the inner product by 1.

```sage
sage: v0 = vec([randint(0, 1) for i in range(m + n)])                                                                
....: v1 = vec([randint(0, 1) for i in range(m + n)])                                                                
....: v2 = vec([randint(0, 1) for i in range(m + n)])                                                                
....: v3 = vec([randint(0, 1) for i in range(m + n)])                                                                
....: # Integrate inequality hints of the form <vi, s> >= li                                                         
....: l0, l1, l2, l3 = ebdd.leak(v0) - 1, ebdd.leak(v1) - 1, ebdd.leak(v2) - 1, ebdd.leak(v3) - 1                    
....: # Inequality hints require the form <vi, s> <= li, so invert                                                   
....: _ = ebdd.integrate_ineq_hint(-v0, -l0)                                                                         
....: _ = ebdd.integrate_ineq_hint(-v1, -l1)                                                                         
....: _ = ebdd.integrate_ineq_hint(-v2, -l2)                                                                         
....: _ = ebdd.integrate_ineq_hint(-v3, -l3)                                                                         
```
```text
 integrate ineq hint     Unworthy hint, Forced it. 
 integrate ineq hint     Worthy hint !   dim=140, δ=1.01246890, β=43.45 
 integrate ineq hint     Worthy hint !   dim=140, δ=1.01247355, β=43.41 
 integrate ineq hint     Worthy hint !   dim=140, δ=1.01247355, β=43.41 
```

Notice that the expected improvement in Bikz for inequality hints is quite low in general, especially when the hints are sampled independently of the secret. They are most effective at modeling information that is correlated with the secret (such as in decryption failure attacks).

### Integrating Combined Hints

Combined hints represent the knowledge of a second ellipsoid that contains the secret. We will simulate this by creating an offset to the secret and use that as a mean of a second ellipsoid.

```sage
sage: v0 = vec([randint(0, 1) for i in range(m + n)])                                                                
....: # Find length of v0                                                                                            
....: radius = v0.norm()                                                                                             
....: # Create center of ellipsoid offset from the secret
....: center = ebdd.u + v0                                                                                           
....: # Create shape matrix                                                                                          
....: shape = radius**2 * identity_matrix(m + n)                                                                     
....: _ = ebdd.integrate_combined_hint(center, shape)
....: secret = ebdd.attack()     
```
```text
integrate combined hint         Worthy hint !   dim=140, δ=1.02546977, β=2.00 
       Running the Attack      
Running BKZ-2   Success ! 
```

Combined hints can be very effective given you know that the secret lies exactly on the surface of the candidate ellipsoid.

### Integrating Perfect Hints

Perfect hints have changed slightly from the prior toolkit. A new method `apply_perfect_hints()` needs to be called after all perfect hints have been integrated.

```sage
sage: # Simulating perfect hints
....: v0 = vec([randint(0, 1) for i in range(m + n)])
....: v1 = vec([randint(0, 1) for i in range(m + n)])
....: v2 = vec([randint(0, 1) for i in range(m + n)])
....: v3 = vec([randint(0, 1) for i in range(m + n)]) 
....: # Computing l = <vi, s>
....: l0, l1, l2, l3 = ebdd.leak(v0), ebdd.leak(v1), ebdd.leak(v2), ebdd.leak(v3)
```

Let us now integrate the perfect hints into our instance.

```sage
sage: # Integrate perfect hints
....: _ = ebdd.integrate_perfect_hint(v0, l0) 
....: _ = ebdd.integrate_perfect_hint(v1, l1) 
....: _ = ebdd.integrate_perfect_hint(v2, l2) 
....: _ = ebdd.integrate_perfect_hint(v3, l3)
....: _ = ebdd.apply_perfect_hints()
....: secret = ebdd.attack()
```
```text
 integrate perfect hint          Worthy hint !   dim=140, δ=1.01253503, β=42.37 
 integrate perfect hint          Worthy hint !   dim=140, δ=1.01264451, β=40.39 
 integrate perfect hint          Worthy hint !   dim=140, δ=1.01268962, β=39.49 
 integrate perfect hint          Worthy hint !   dim=140, δ=1.01273457, β=38.59 
 apply perfect hints             Worthy hint !   dim=136, δ=1.01330913, β=28.44 
       Running the Attack      
Running BKZ-30   Success ! 
  
```

Notice how the dimension of the lattice does not change until `apply_perfect_hints()` is called.
