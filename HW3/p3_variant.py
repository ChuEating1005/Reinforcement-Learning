import numpy as np
import matplotlib.pyplot as plt

# Parameters
epsilon = 0.2  # Epsilon for clipping
r = np.linspace(0.5, 1.5, 1000)  # Range of probability ratio values

# Define the PPO clip function based on the given probability ratios
def ppo_clip(r, A_t, epsilon=0.2):
    clip_min = (1 - epsilon) * A_t
    clip_max = (1 + epsilon) * A_t
    # Apply clipping based on the sign of A_t
    clip_values = np.where(r < 1 - epsilon, clip_min, np.where(r > 1 + epsilon, clip_max, r * A_t))    
    return clip_values
# Calculate clipped values for both positive and negative A_t
clipped_values_positive = ppo_clip(r, 1, epsilon)
clipped_values_negative = ppo_clip(r, -1, epsilon)

# Plotting the function
plt.subplot(1, 2, 1)
plt.plot(r, clipped_values_positive, label='Clipped Function with $A_t > 0$', color='black', linewidth=2)
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
plt.plot(r, clipped_values_negative, label='Clipped Function with $A_t < 0$', color='black', linewidth=2)
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