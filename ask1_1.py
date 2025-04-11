"""max Z = 3x1 + x2
όταν
(Π1) 6x1 + 3x2 ≥12
(Π2) 4x1 + 8x2 ≥16
(Π3) 6x1 + 5x2 ≤30
(Π4) 6x1 + 7x2 ≤36
x1, x2 ≥0"""

"""
(α) Να παραστήσετε γραφικά την εφικτή περιοχή του προβλήματος καθώς και όλες τις
κορυφές της. Περιγράψτε τη μορφή της εφικτής περιοχής. Με γραφικό τρόπο βρείτε
τη βέλτιστη κορυφή του προβλήματος, εάν υπάρχει.
(β) Αν η αντικειμενική συνάρτηση του παραπάνω προβλήματος είναι min Z = x1 +
c2x2, ποιο είναι το εύρος τιμών που θα μπορούσε να πάρει το c2 έτσι ώστε η βέλτιστη
λύση να βρίσκεται στην τομή των ευθειών που ορίζουν οι περιορισμοί Π1 και Π2;
(γ) Αν στο παραπάνω πρόβλημα μεγιστοποίησης η αντικειμενική συνάρτηση ήταν Z =
c1x1 + c2x2 βρείτε τις σχετικές τιμές των c1 και c2 έτσι ώστε η βέλτιστη λύση να
βρίσκεται στην τομή των ευθειών που ορίζουν οι περιορισμοί Π3 και Π4"""
import numpy as np  
import matplotlib.pyplot as plt 
import sympy as sp
f_objective = np.array([3, 1])  
A = np.array([[6, 3], [4, 8], [6, 5], [6, 7]])  
b = np.array([12, 16, 30, 36]) 
x1_bounds = (0, None)  # x1 >= 0
x2_bounds = (0, None)  # x2 >= 0
def plot_constraints():
    x1 = np.linspace(0, 10, 100)  
    plt.figure(figsize=(10, 8))

    x2_values = []
    for i in range(A.shape[0]):
        if A[i, 1] != 0:  
            x2 = (b[i] - A[i, 0] * x1) / A[i, 1]
            x2_values.append(x2)
            plt.plot(x1, x2, label=f'Constraint {i+1}')
        else:
            plt.axvline(x=b[i]/A[i, 0], label=f'Constraint {i+1}')

    lower_bound = np.maximum(x2_values[0], x2_values[1])
    upper_bound = np.minimum(x2_values[2], x2_values[3])
    
    # Fill the feasible region 
    mask = (lower_bound <= upper_bound)
    
 
    plt.fill_between(x1, lower_bound, upper_bound, 
                    where=mask, 
                    color='green', alpha=0.5)
    #plot gradient vector
    gradient_vector = np.array([3, 1])
    origin = np.array([0, 0])
    plt.quiver(*origin, *gradient_vector, angles='xy', scale_units='xy', scale=1, color='red', label='Gradient Vector')
    
    ## arbitrary line Z0 = 3x1 + x2
    Z0 = 10
    x2_line = Z0 - 3 * x1
    plt.plot(x1, x2_line, label='Z0 = 3x1 + x2', color='orange')
    
    
    plt.xlim(0, 10)

    plt.ylim(0, 10)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Feasible Region and Constraints')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_constraints()
    # The feasible region is the area where all constraints overlap.
    # We can solve the problem graphicaly by creating an arbitrary
    #  line Z0 = 3x1 + x2
    #and moving it parallel to the vector (3, 1) 
    # tHE optimal solution is the las vertex touched by the line.
    #obviosly that is the  
    # [x1, x2] = [2.5,3]
    #C3 and C4  INTERSECTION
    print("the maxixmum is at the intersection of C3 and C4")
    print("x1 = 2.5, x2 = 3")
    
    print(f"Z ={2.5*3+3} ")