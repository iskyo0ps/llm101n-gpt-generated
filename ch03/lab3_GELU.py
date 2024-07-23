import numpy as np
import matplotlib.pyplot as plt

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# Test the GELU function
x = np.linspace(-3, 3, 100)
y = gelu(x)

# Plot the GELU function
plt.plot(x, y)
plt.title('GELU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()