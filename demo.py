import matplotlib.pyplot as plt
import numpy as np

# Define the parametric equations for the heart shape
t = np.linspace(0, 2 * np.pi, 1000)
x = 16 * np.sin(t)**3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

plt.figure(figsize=(6,6))
plt.plot(x, y, color='red')
plt.title('Heart Shape')
plt.axis('off')  # Turn off the axis
plt.show()