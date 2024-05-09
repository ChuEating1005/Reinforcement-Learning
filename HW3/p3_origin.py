
import numpy as np
import matplotlib.pyplot as plt

# Define epsilon for clipping and the range for r
epsilon = 0.2
r = np.linspace(0.5, 1.5, 300)  # Extending beyond 1 + epsilon for illustration

# Define the clipping function with a negative Advantage function
def clipped_function_negative(r, epsilon, A_t):
    # Assume A_t is negative; modify the function accordingly
    # The function is multiplied by the negative advantage to reflect decrease
    if(A_t > 0): return np.where(r <= 1 + epsilon, r * A_t, (1 + epsilon) * A_t)
    else: return np.where(r >= 1 - epsilon, r * A_t, (1 - epsilon) * A_t)

# Plotting the function
plt.subplot(1, 2, 1)
plt.plot(r, clipped_function_negative(r, epsilon, 1), label='Clipped Function with $A_t > 0$', color='black', linewidth=2)
plt.scatter([1], [1], color='red')  # Highlight the point (1, 1)

# Add dashed lines for 1-epsilon and 1+epsilon
plt.axvline(1 - epsilon, color='gray', linestyle='dashed', label='1 - ε')
plt.axvline(1 + epsilon, color='gray', linestyle='dashed', label='1 + ε')

# Add labels and title
plt.title('$A_t > 0$')
plt.xlabel('Probability Ratio r')
plt.ylabel('Clipped Value')
plt.ylim(0, 1.7)  # Set y-axis limits to show the plateau clearly
plt.grid(False)
plt.legend()


# Plotting the function for negative advantage
plt.subplot(1, 2, 2)
plt.plot(r, clipped_function_negative(r, epsilon, -1), label='Clipped Function with $A_t < 0$', color='black', linewidth=2)
plt.scatter([1], [-1], color='red')  # Highlight the point (1, -1)

# Add dashed lines for 1-epsilon and 1+epsilon
plt.axvline(1 - epsilon, color='gray', linestyle='dashed', label='1 - ε')
plt.axvline(1 + epsilon, color='gray', linestyle='dashed', label='1 + ε')

# Add labels and title
plt.title('$A_t < 0$')
plt.xlabel('Probability Ratio r')
plt.ylabel('Clipped Value')
plt.ylim(1.5 * -1, 0)  # Adjust y-axis limits based on negative advantage
plt.grid(False)
plt.legend()

plt.tight_layout()
plt.show()