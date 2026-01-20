
## Lattice Reduction with Short Vector Integration

Pre-reqs:

- Sage 9.1 or greater

Run experiment with 60 guessable variables and noise 1.2 (st. dev)

```bash
sage load-ntt-data-modified.sage --guesses 60 --noise 1.2
```

or 

Write the following `experiment.json` file

```json
{
    "experiments": [
        "guesses":  [60],
        "num-iterations": 1,
        "noise": {
            "start": 1.2,
            "stop": 1.2,
            "num": 1
        }
    ]
}
```

and run

```bash
sage load-ntt-data-modified.sage experiment.json
```

For more experimental arguments see help

```bash
sage load-ntt-data-modified.sage -h
```


