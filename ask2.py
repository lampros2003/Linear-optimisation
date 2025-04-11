import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Decision variables:
# x_1: Dose of radiation ray 1 (kilorads)
# x_2: Dose of radiation ray 2 (kilorads)

# Objective function - Minimize healthy tissue radiation
# min Z = 0.6x_1 + 0.5x_2
c = np.array([0.6, 0.5])

# Inequality constraints
# 1. Sensitive tissue: 0.3x_1 + 0.1x_2 <= 2.7
# 2. Tumor center of mass: 0.6x_1 + 0.4x_2 >= 6 (convert to <= by multiplying by -1)

A_ub = np.array([
    [0.3, 0.1],   # Sensitive tissue
    [-0.6, -0.4]  # Tumor center (reversed for >=)
])
b_ub = np.array([2.7, -6])

# Equality constraint
# Tumor: 0.5x_1 + 0.5x_2 = 6
A_eq = np.array([[0.5, 0.5]])
b_eq = np.array([6])

# Bounds for variables (non-negativity)
bounds = [(0, None), (0, None)]

# Solve the linear program
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='simplex')

# Print the results
print("Status:", result.message)
print("Optimal solution:")
print(f"x_1 = {result.x[0]:.4f} kilorads")
print(f"x_2 = {result.x[1]:.4f} kilorads")
print(f"Minimum total radiation to healthy tissue: {result.fun:.4f}")

# Visualize the solution
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
    # plot the z gradient vector
    gradient_vector = np.array([0.6, 0.5])
    origin = np.array([0, 0])
    plt.quiver(*origin, *gradient_vector, angles='xy', scale_units='xy', scale=0.1, color='red', label='Gradient Vector')

    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.xlabel('x₁ (Dose of ray 1, kilorads)')
    plt.ylabel('x₂ (Dose of ray 2, kilorads)')
    plt.title('Radiation Therapy Optimization')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # Print constraint values at optimal solution
    print("\nConstraint values at optimal solution:")
    print(f"Sensitive tissue: 0.3x₁ + 0.1x₂ = {0.3*result.x[0] + 0.1*result.x[1]:.4f} <= 2.7")
    print(f"Tumor: 0.5x₁ + 0.5x₂ = {0.5*result.x[0] + 0.5*result.x[1]:.4f} = 6")
    print(f"Tumor center: 0.6x₁ + 0.4x₂ = {0.6*result.x[0] + 0.4*result.x[1]:.4f} >= 6")
    
    plt.show()

plot_solution()