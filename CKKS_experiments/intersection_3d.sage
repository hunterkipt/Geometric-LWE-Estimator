reset()
load("../framework/LWE.sage")

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from mpl_toolkits.mplot3d import Axes3D


def generate_random_mu_Sigma(dim, num_pairs):
    mu_Sigma_pairs = []
    mu = np.random.rand(dim)
    A = np.random.rand(dim, dim)
    Sigma = A @ A.T
    mu_Sigma_pairs += [mu, Sigma]

    seed = np.random.rand()
    print(f"seed: {seed}")

    for _ in range(num_pairs - 1):
        mu = np.random.rand(dim)
        if seed > 0.9:
            A = np.random.rand(dim, dim)
            Sigma = A @ A.T
        else:
            A = np.random.rand(dim, dim)
            D = np.diag(np.random.choice([-1, 1], size=dim))
            Sigma = A @ D @ A.T
        mu_Sigma_pairs += [mu, Sigma]

    return mu_Sigma_pairs


def plot_ellipsoid_3d(ax, mu, Sigma, label="", color=None, num_points=50):
    mu = np.array(mu).flatten()
    Sigma = np.array(Sigma)
    vals, vecs = eig(Sigma)
    if not np.all(vals > 0):
        return

    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    rx, ry, rz = np.sqrt(vals)

    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    u, v = np.meshgrid(u, v)

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    sphere_points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    scaled = np.diag([rx, ry, rz]) @ sphere_points
    rotated = vecs @ scaled
    rotated[0, :] += mu[0]
    rotated[1, :] += mu[1]
    rotated[2, :] += mu[2]

    X = rotated[0, :].reshape(num_points, num_points)
    Y = rotated[1, :].reshape(num_points, num_points)
    Z = rotated[2, :].reshape(num_points, num_points)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, color=color)
    ax.plot([], [], [], color=color, label=label)


def plot_hyperboloid_3d(ax, mu, Sigma, label="", color=None, num_points=50):
    mu = np.array(mu).flatten()
    Sigma = np.array(Sigma)
    vals, vecs = eig(Sigma)
    if not (np.any(vals > 0) and np.any(vals < 0)):
        return

    pos_count = np.sum(vals > 0)
    neg_count = np.sum(vals < 0)
    idx = np.argsort(np.abs(vals))
    vals = vals[idx]
    vecs = vecs[:, idx]

    # One-sheet
    if pos_count == 2 and neg_count == 1:
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(-2, 2, num_points)
        u, v = np.meshgrid(u, v)
        x_ = np.cosh(v) * np.cos(u)
        y_ = np.cosh(v) * np.sin(u)
        z_ = np.sinh(v)
        shape_label = label + " (one-sheet)"
        x_flat = x_.ravel()
        y_flat = y_.ravel()
        z_flat = z_.ravel()
        top_count = None
    else:
        # Two-sheet
        r = np.linspace(0, 2, num_points)
        th = np.linspace(0, 2 * np.pi, num_points)
        R, TH = np.meshgrid(r, th)
        X_top = R * np.cos(TH)
        Y_top = R * np.sin(TH)
        Z_top = np.sqrt(1 + R**2)
        X_bot = R * np.cos(TH)
        Y_bot = R * np.sin(TH)
        Z_bot = -np.sqrt(1 + R**2)
        x_flat = np.concatenate([X_top.ravel(), X_bot.ravel()])
        y_flat = np.concatenate([Y_top.ravel(), Y_bot.ravel()])
        z_flat = np.concatenate([Z_top.ravel(), Z_bot.ravel()])
        shape_label = label + " (two-sheet)"
        top_count = X_top.size

    scale = np.sqrt(np.abs(vals))
    D = np.diag(scale)
    xyz = np.vstack([x_flat, y_flat, z_flat])
    scaled = D @ xyz
    rotated = vecs @ scaled
    rotated[0, :] += mu[0]
    rotated[1, :] += mu[1]
    rotated[2, :] += mu[2]

    if top_count is not None:
        # two-sheet case
        X_t = rotated[0, :top_count].reshape(num_points, num_points)
        Y_t = rotated[1, :top_count].reshape(num_points, num_points)
        Z_t = rotated[2, :top_count].reshape(num_points, num_points)
        X_b = rotated[0, top_count:].reshape(num_points, num_points)
        Y_b = rotated[1, top_count:].reshape(num_points, num_points)
        Z_b = rotated[2, top_count:].reshape(num_points, num_points)
        ax.plot_surface(X_t, Y_t, Z_t, rstride=1, cstride=1, alpha=0.5, color=color)
        ax.plot_surface(X_b, Y_b, Z_b, rstride=1, cstride=1, alpha=0.5, color=color)
    else:
        # one-sheet case
        X = rotated[0, :].reshape(num_points, num_points)
        Y = rotated[1, :].reshape(num_points, num_points)
        Z = rotated[2, :].reshape(num_points, num_points)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, color=color)

    ax.plot([], [], [], color=color, label=shape_label)


###############################################################################
# Example usage
###############################################################################
dim = 3
num_pairs = 6
pairs = generate_random_mu_Sigma(dim, num_pairs)

# Assume this intersection function is defined in your environment
new_mu, new_Sigma = ellipsoid_quadratic_froms_intersection(*pairs, lb=0.1, tolerance=1e-8)
print("new_mu:", new_mu)
print("new_Sigma:", new_Sigma)

mu_and_Sigma = []
for i in range(0, len(pairs), 2):
    if i + 1 < len(pairs):
        mu_and_Sigma.append((pairs[i], pairs[i + 1]))
mu_and_Sigma.append((new_mu.numpy().flatten(), new_Sigma.numpy()))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

colors = plt.cm.tab10(np.linspace(0, 1, len(mu_and_Sigma)))

for i, (mu, Sigma) in enumerate(mu_and_Sigma):
    e_vals, _ = eig(Sigma)
    color = colors[i]
    if np.all(e_vals > 0):
        plot_ellipsoid_3d(ax, mu, Sigma, label=f"Ellipsoid {i+1}", color=color)
    elif np.any(e_vals < 0):
        plot_hyperboloid_3d(ax, mu, Sigma, label=f"Hyperboloid {i+1}", color=color)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("3D Quadratic Forms")
ax.legend()
plt.show()
