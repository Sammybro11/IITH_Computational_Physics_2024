import numpy as np
import matplotlib.pyplot as plt


# Define the function proportional to the target distribution
def f(x):
    # Convert x to a NumPy array to handle both scalars and arrays
    x = np.array(x)

    # Initialize output array with zeros (same shape as x)
    result = np.zeros_like(x)

    # Create a mask for valid x values within (0,1)
    mask = (x > 0) & (x < 1)

    # Apply function only where valid
    result[mask] = -np.sin(x[mask]) * np.log(x[mask])

    # Return result (element-wise safe)
    return result


# Initialize parameters
n_samples = 100000  # total number of samples to generate
samples = []  # list to store accepted samples
x_current = 0.5  # start point in the middle of [0,1]
sigma = 0.1  # proposal step size (standard deviation)

# Run the Metropolis algorithm
for i in range(n_samples):
    # Propose a new candidate from a normal distribution centered at current x
    x_proposed = x_current + np.random.normal(0, sigma)

    # Reflect if the proposed value goes out of [0,1] (to stay within bounds)
    if x_proposed < 0:
        x_proposed = -x_proposed
    elif x_proposed > 1:
        x_proposed = 2 - x_proposed

    # Compute acceptance ratio (unnormalized)
    p_current = f(x_current)
    p_proposed = f(x_proposed)

    # Avoid divide-by-zero errors
    if p_current == 0:
        acceptance_ratio = 1
    else:
        acceptance_ratio = p_proposed / p_current

    # Accept or reject based on Metropolis criterion
    if np.random.rand() < min(1, acceptance_ratio):
        x_current = x_proposed  # accept move

    # Store the current sample
    samples.append(x_current)

# Convert to NumPy array for analysis
samples = np.array(samples)

# Plot the sampled distribution
plt.figure(figsize=(8, 5))
plt.hist(samples, bins=100, density=True, alpha=0.6, color='skyblue', label='Sampled distribution')

# Plot the true (unnormalized) function for comparison
x = np.linspace(0.001, 0.999, 500)
plt.plot(x, f(x) / np.trapz(f(x), x), 'r-', lw=2, label='True (normalized) function')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.legend()
plt.title('Sampling from f(x) = -sin(x) ln(x) using Metropolis algorithm')
plt.show()
