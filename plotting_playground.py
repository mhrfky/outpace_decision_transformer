import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
# Define the function to create a surface with global maxima and minima
def function(x, y):
    final = [0, 8]
    init = np.array([0, 0])
    
    def dist_to_final(x,y):
        return np.log((final[0] - x ) ** 2 + (final[1] - y) ** 2)
    def dist_to_init(x,y):
        return np.log((init[0] - x ) ** 2 + (init[1] - y) ** 2)
    return - dist_to_final(x,y) + dist_to_init(x,y)

# Generate the grid
x = np.linspace(-10, 10, 400)
y = np.linspace(-2, 12, 400)
X, Y = np.meshgrid(x, y)
Z = function(X, Y)

# Plot the function
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
plt.colorbar(contour)

# Mark the global minimum and maximum
plt.plot(0, 0, 'ro', markersize=10, label='Global Minimum (0, 0)')
plt.plot(0, 8, 'go', markersize=10, label='Global Maximum (0, 8)')

# Labels and title
plt.title('2D Plot with Global Maxima and Minima')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.xlim(-2, 10)
plt.ylim(-2, 10)
plt.savefig("playground/2d_gaussian_improved_smoothl.png")

plt.show()
