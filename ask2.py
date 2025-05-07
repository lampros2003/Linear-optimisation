import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Visualize 
def plot_solution():
    x1 = np.linspace(0, 15, 1000)
    
    x2_1 = (2.7 - 0.3*x1)/0.1
    
    x2_2 = 12 - x1
    
    x2_3 = (6 - 0.6*x1)/0.4
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot constraints
    plt.plot(x1, x2_1, label='Sensitive tissue: 0.3x₁ + 0.1x₂ ≤ 2.7')
    plt.plot(x1, x2_2, label='Tumor: 0.5x₁ + 0.5x₂ = 6', linewidth=2)
    plt.plot(x1, x2_3, label='Tumor center: 0.6x₁ + 0.4x₂ ≥ 6')
    
    #plot objective function at Z0 = 6 and Z0 = 10
    Z0 = 6
    x2_line = Z0 - 0.6 * x1
    plt.plot(x1, x2_line, label='Z0 = 0.6x₁ + 0.5x₂', color='orange')
    Z0 = 10
    x2_line2 = Z0 - 0.6 * x1
    plt.plot(x1, x2_line2, label='Z0 = 0.6x₁ + 0.5x₂', color='black')
    # plot the z gradient 
    gradient_vector = np.array([0.6, 0.5])
    origin = np.array([0, 0])
    plt.quiver(*origin, *gradient_vector, angles='xy', 
               scale_units='xy', scale=0.1, color='red', label='Gradient Vector')

    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.xlabel('x₁ (Dose of ray 1, kilorads)')
    plt.ylabel('x₂ (Dose of ray 2, kilorads)')
    plt.title('Radiation Therapy Optimization')
    plt.grid(True)
    plt.legend(loc='upper right')
    
  
    plt.show()

plot_solution()