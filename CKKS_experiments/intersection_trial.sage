reset()

load("../framework/LWE.sage")
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from numpy.linalg import inv, det, eig
import matplotlib.lines as mlines


# Generate random mu and Sigma pairs
def generate_random_mu_Sigma(dim, num_pairs):
    mu_Sigma_pairs = []
    mu = np.random.rand(dim)
    A = np.random.rand(dim, dim)
    Sigma = np.dot(A, A.transpose())  # Positive semi-definite
    mu_Sigma_pairs.extend([mu, Sigma])
    seed = np.random.rand()
    print(f"seed: {seed}")
    for i in range(num_pairs-1):
        mu = np.random.rand(dim)  # Random center
        if seed > 0.9:
            # Ellipsoid
            A = np.random.rand(dim, dim)
            Sigma = np.dot(A, A.transpose())  # Positive semi-definite
        else:
            # Hyperboloid-like shape
            A = np.random.rand(dim, dim)
            D = np.diag(np.random.choice([-1, 1], dim))  # Diagonal matrix with -1 or 1
            Sigma = np.dot(A, np.dot(D, A.transpose()))  # Mixed eigenvalues

        mu_Sigma_pairs.extend([mu, Sigma])
    return mu_Sigma_pairs

def plot_ellipsoid(mu, Sigma, label):
    # Generate theta values
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Create the unit circle
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Form the points in 2D space
    points = np.array([x, y])
    
    # Calculate the transformation matrix
    L = np.linalg.cholesky(np.linalg.inv(Sigma))
    
    # Transform the points
    transformed_points = L @ points
    
    # Translate the points to the new center (mu)
    transformed_points[0, :] += mu[0]
    transformed_points[1, :] += mu[1]
    
    # Plot the points
    plt.plot(transformed_points[0, :], transformed_points[1, :], label=label)

def plot_hyperboloid(mu, Sigma, label):
    # Ensure Sigma has mixed eigenvalues for a hyperboloid
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    if not (np.any(eigenvalues > 0) and np.any(eigenvalues < 0)):
        raise ValueError("Sigma must have mixed eigenvalues for a hyperboloid.")

    # Generate t values for two branches of the hyperbola
    t = np.linspace(-3, 3, 100)

    # Standard hyperbola equations
    x1 = np.cosh(t)
    y1 = np.sinh(t)

    x2 = -np.cosh(t)
    y2 = np.sinh(t)

    # Scale and rotate points using the eigenvectors and eigenvalues
    transformed_points1 = eigenvectors @ np.diag(np.sqrt(np.abs(eigenvalues))) @ np.array([x1, y1])
    transformed_points2 = eigenvectors @ np.diag(np.sqrt(np.abs(eigenvalues))) @ np.array([x2, y2])

    # Translate to the new center (mu)
    transformed_points1 += mu[:, np.newaxis]
    transformed_points2 += mu[:, np.newaxis]

    # Plot the points
    plt.plot(transformed_points1[0, :], transformed_points1[1, :], label=label + ' branch 1')
    plt.plot(transformed_points2[0, :], transformed_points2[1, :], label=label + ' branch 2')

# Dimension of the ellipsoids
dim = 2

# Number of (mu, Sigma) pairs
num_pairs = 6

# Generate random (mu, Sigma) pairs
mu_Sigma_pairs = generate_random_mu_Sigma(dim, num_pairs)

# # Print the generated pairs for verification
# for i in range(0, len(mu_Sigma_pairs), 2):
#     print(f"mu{i//2}: {mu_Sigma_pairs[i]}")
#     print(f"Sigma{i//2}: {mu_Sigma_pairs[i+1]}")

# You can then pass these to your function like so:
new_mu, new_Sigma = ellipsoid_quadratic_froms_intersection(*mu_Sigma_pairs, lb=0.1, tolerance=1e-8)
print(f"new_mu: {new_mu}")
print(f"new_Sigma: {new_Sigma}")

# Given mu and Sigma pairs
mu_and_Sigma = []
for i in range(0, len(mu_Sigma_pairs), 2):
    if i + 1 < len(mu_Sigma_pairs):
        mu_and_Sigma.append(((mu_Sigma_pairs[i]), (mu_Sigma_pairs[i+1])))
mu_and_Sigma.append(((new_mu.numpy().flatten()), ((new_Sigma).numpy())))

print(mu_and_Sigma)

# Create a figure
plt.figure()

# Plot
for i, (mu, Sigma) in enumerate(mu_and_Sigma):
    eigenvalues, _ = eig(Sigma)
    if np.all(eigenvalues > 0):
        label = f"Ellipsoid {i+1}"
        plot_ellipsoid(mu, Sigma, label)
    elif np.all(eigenvalues < 0):
        pass
    elif np.any(eigenvalues < 0):
        label = f"Hyperboloid {i+1}"
        plot_hyperboloid(mu, Sigma, label)

# Add grid and labels for better visualization
plt.grid(True)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Quadratic Forms")

# Add legend to the plot
plt.legend()

# Show the plot
plt.show()





