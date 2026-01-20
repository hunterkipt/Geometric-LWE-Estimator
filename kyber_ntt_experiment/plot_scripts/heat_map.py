
from json import load, dump
import numpy as np
import matplotlib.pyplot as plt

def padded_avg(lst, dim):
    success_betas = sum(list(map(lambda x: x[1], lst)))
    return 0.2 * (success_betas + dim * (5 - len(lst)))

successes = { "start": 0.4, "step": 0.2 }
for i in range(45, 65):
    with open(f"counts_{i:>2}.json", "r") as f:
        success_counts = load(f)
    with open(f"results_{i:>2}.json", "r") as f:
        bkz = load(f)
    betas = []
    idx = 0
    for k, v in success_counts.items():
        betas.append(bkz[idx:(idx+v)])
        idx += v
    successes[i] = {
        "dim": d(i),
        "BKZ": betas
    }

# with open("successes.json", "w") as f:
#     dump(successes, f)

# with open("successes.json", "r") as f:
#     success = load(f)

n = 64 - 45 + 1
heat = np.empty((n, 9))
for g in range(n):
    result = success[str(g + 45)]
    d = result["dim"]
    heat[g] = list(
        map(
            lambda x: padded_avg(x, d), 
            result["BKZ"]
        )
    )

heat = heat.T

fig, ax = plt.subplots()
im = ax.imshow(heat)

guesses = np.arange(45, 65)
noise = list (
    map (
        lambda x: round(10 * x) / 10,
        np.arange (
            success["start"], 
            success["start"] + (9 * success["step"]),
            success["step"]
        )
    )
)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(guesses)), labels=guesses)
ax.set_yticks(np.arange(len(noise)), labels=noise)

ax.set_xlabel("guesses")
ax.set_ylabel("noise (std)")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(noise)):
    for j in range(len(guesses)):
        text = ax.text(j, i, int(heat[i, j]),
                       ha="center", va="center", color="w")

ax.set_title("Average BKZ")
fig.tight_layout()
# plt.show()
plt.savefig("bkz_map.png")

    
