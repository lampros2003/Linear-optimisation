import numpy as np
from itertools import combinations
import pandas as pd
from scipy.optimize import linprog


# Define the constraints as hyperplanes
hyperplanes = [
    {'name': 'x1 + x2 = 10', 'coeffs': [1, 1, 0], 'rhs': 10},
    {'name': 'x2 + x3 = 15', 'coeffs': [0, 1, 1], 'rhs': 15},
    {'name': 'x1 + x3 = 12', 'coeffs': [1, 0, 1], 'rhs': 12},
    {'name': '20x1 + 10x2 + 15x3 = 300', 'coeffs': [20, 10, 15], 'rhs': 300},
    {'name': 'x1 = 0', 'coeffs': [1, 0, 0], 'rhs': 0},
    {'name': 'x2 = 0', 'coeffs': [0, 1, 0], 'rhs': 0},
    {'name': 'x3 = 0', 'coeffs': [0, 0, 1], 'rhs': 0}
]

def solve_system(eq1, eq2, eq3):
    """Solve  system of 3 linear equations with 3 variables 3x3 system"""
    A = np.array([eq1['coeffs'], eq2['coeffs'], eq3['coeffs']])
    b = np.array([eq1['rhs'], eq2['rhs'], eq3['rhs']])
    
    try:
        x = np.linalg.solve(A, b)
        return x
    except np.linalg.LinAlgError:
        return None

def is_feasible(point):
    """Check if a point satisfies all constraints"""
    x1, x2, x3 = point
    
    #Check n-negativity
    if x1 < -1e-10 or x2 < -1e-10 or x3 < -1e-10:
        return False
    
    #Check other constraints with a small tolerance
    if x1 + x2 < 10 - 1e-10:
        return False
    if x2 + x3 < 15 - 1e-10:
        return False
    if x1 + x3 < 12 - 1e-10:
        return False
    if 20*x1 + 10*x2 + 15*x3 > 300 + 1e-10:
        return False
        
    return True

def objective_value(point):
    """Calculate the objective function value"""
    x1, x2, x3 = point
    return 8*x1 + 5*x2 + 4*x3

def check_degeneracy(vertex, hyperplanes_list):
    """Check if a vertex is degenerate """
    active_planes = []
    
    for hp in hyperplanes_list:
        x1, x2, x3 = vertex
        
        
        a, b, c = hp['coeffs']
        
        rhs = hp['rhs']
        if abs(a*x1 + b*x2 + c*x3 - rhs) < 1e-10:
            active_planes.append(hp['name'])
    
    return len(active_planes) > 3, active_planes

# Find all  vertices by solving systems of 3x3 equations
vertices = []
vertex_info = []
for eqs in combinations(hyperplanes, 3):
    point = solve_system(eqs[0], eqs[1], eqs[2])
    
    if point is not None:
        is_feas = is_feasible(point)
        is_deg, active_planes = check_degeneracy(point, hyperplanes)
        
        obj_val = objective_value(point) if is_feas else None
        
        vertex_info.append({
            'point': tuple(np.round(point, 10)),
            'equations': [eq['name'] for eq in eqs],
            'feasible': is_feas,
            'degenerate': is_deg,
            'active_planes': active_planes,
            'objective_value': obj_val
        })
        
        if is_feas:
            vertices.append(point)

# Sort and display the results
vertex_df = pd.DataFrame(vertex_info)
feasible_df = vertex_df[vertex_df['feasible']].copy()
feasible_df = feasible_df.sort_values(by='objective_value')

# Part B:
def find_basic_solutions():
    """Find all basic solutions after adding slack variables"""
    # A matrix 
    A = np.array([
        [1, 1, 0, -1, 0, 0, 0],    
        [0, 1, 1, 0, -1, 0, 0],    
        [1, 0, 1, 0, 0, -1, 0],    
        [20, 10, 15, 0, 0, 0, 1]   
    ])
    
    # B matrix
    b = np.array([10, 15, 12, 300])
    
    # Variables
    var_names = ['x1', 'x2', 'x3', 's1', 's2', 's3', 's4']
    
    basic_solutions = []
    
    # Try all combinations of 4 variables to be in the basis
    for basis_indices in combinations(range(7), 4):
        # Extract variables columns φορ hte basis
        B = A[:, basis_indices]
        
        # Try to solve the system
        try:
            # Check if invertible
            if abs(np.linalg.det(B)) > 1e-10:
                #Solve 
                basis_values = np.linalg.solve(B, b)
                
                # Create  solution vector
                solution = np.zeros(7)
                for i, idx in enumerate(basis_indices):
                    solution[idx] = basis_values[i]
                
                #Check if feasible 
                is_feasible = np.all(solution >= -1e-10)
                
                #Check if  degen
                zero_basics = sum(1 for i, idx in enumerate(basis_indices) if abs(basis_values[i]) < 1e-10)
                is_degenerate = zero_basics > 0
                
                # Calculate Z for feasible solutions
                obj_val = None
                if is_feasible:
                    obj_val = 8*solution[0] + 5*solution[1] + 4*solution[2]
                
                # Store the solution info
                basic_solutions.append({
                    'basis_variables': [var_names[idx] for idx in basis_indices],
                    'solution': solution,
                    'feasible': is_feasible,
                    'degenerate': is_degenerate,
                    'objective_value': obj_val,
                    'x1': solution[0],
                    'x2': solution[1],
                    'x3': solution[2],
                    's1': solution[3],
                    's2': solution[4],
                    's3': solution[5],
                    's4': solution[6]
                })
        except np.linalg.LinAlgError:
            # no solution
            continue
    
    return basic_solutions

def add_slack_variables():
    """Convert the problem to standard form with slack variables """
    print("\nPart B: ")
    print("Original constraints:")
    print("x1 + x2 >= 10")
    print("x2 + x3 >= 15")
    print("x1 + x3 >= 12")
    print("20x1 + 10x2 + 15x3 ≤ 300")
    
    print("\nWith slack variables:")
    print("x1 + x2 - s1 = 10")
    print("x2 + x3 - s2 = 15")
    print("x1 + x3 - s3 = 12")
    print("20x1 + 10x2 + 15x3 + s4 = 300")
    print("x1, x2, x3, s1, s2, s3, s4 >= 0")
    
    # Find all basic solutions
    basic_solutions = find_basic_solutions()
    
    # Create a DataFrame for better display
    df = pd.DataFrame(basic_solutions)
    
    # Count solutions
    feasible_count = df['feasible'].sum()
    infeasible_count = len(df) - feasible_count
    
    # Count degenerate solutions
    degenerate_count = df['degenerate'].sum()
    
    print(f"\nFound {len(df)} basic solutions")
    print(f"Of these, {feasible_count} are feasible and {infeasible_count} are infeasible")
    print(f"There are {degenerate_count} degenerate basic solutions")
    
    # Display all basic solutions
    print("\nAll basic solutions:")
    for i, sol in enumerate(basic_solutions):
        basis_str = ', '.join(sol['basis_variables'])
        status = "FEASIBLE" if sol['feasible'] else "INFEASIBLE"
        degeneracy = "DEGENERATE" if sol['degenerate'] else "NON-DEGENERATE"
        
        print(f"\nSolution {i+1}: {status}, {degeneracy}")
        print(f"Basis variables: {basis_str}")
        print(f"x1 = {sol['x1']:.6f}, x2 = {sol['x2']:.6f}, x3 = {sol['x3']:.6f}")
        print(f"s1 = {sol['s1']:.6f}, s2 = {sol['s2']:.6f}, s3 = {sol['s3']:.6f}, s4 = {sol['s4']:.6f}")
        
        if sol['feasible']:
            print(f"Objective value: {sol['objective_value']:.6f}")
    
    return basic_solutions

# Part C: Find optimal solution and establish correspondence
def compare_solutions(vertices_df, basic_solutions):
    """Compare vertices from Part A with basic feasible solutions from Part B"""
    print("\nPart C: Correspondence between vertices and basic feasible solutions")
    
    # Filter for feasible basic solutions
    feasible_basic = [sol for sol in basic_solutions if sol['feasible']]
    
    #Sort by Ζ
    feasible_basic.sort(key=lambda x: x['objective_value'] if x['objective_value'] is not None else float('inf'))
    
    #Find the optimal solution
    optimal = feasible_basic[0]
    print(f"\nOptimal solution:")
    print(f"x1 = {optimal['x1']:.6f}, x2 = {optimal['x2']:.6f}, x3 = {optimal['x3']:.6f}")
    print(f"s1 = {optimal['s1']:.6f}, s2 = {optimal['s2']:.6f}, s3 = {optimal['s3']:.6f}, s4 = {optimal['s4']:.6f}")
    print(f"Objective value: {optimal['objective_value']:.6f}")
    
    #Createvertices το solutions map
    print("\nCorrespondence between vertices and basic feasible solutions:")
    for i, vertex_row in vertices_df[vertices_df['feasible']].iterrows():
        vertex = vertex_row['point']
        #Find matching basic feasible solution
        for j, basic_feasible_solution in enumerate(feasible_basic):
            if (abs(vertex[0] - basic_feasible_solution['x1']) < 1e-10 and 
                abs(vertex[1] - basic_feasible_solution['x2']) < 1e-10 and 
                abs(vertex[2] - basic_feasible_solution['x3']) < 1e-10):
                
                print(f"\nVertex {vertex} corresponds to Basic Feasible Solution {j+1}")
                print(f"Basis variables: {', '.join(basic_feasible_solution['basis_variables'])}")
                print(f"Objective value: {basic_feasible_solution['objective_value']:.6f}")
                
                if j == 0:  
                    print("THIS IS THE OPTIMAL SOLUTION")
                break

# Display -----------------------------


print("Part A:")
print(f"Found {len(vertex_info)} potential vertices")
print(f"Of these, {len(vertices)} are feasible")

print("\nFeasible vertices (sorted by objective value):")
for idx, row in feasible_df.iterrows():
    print(f"Vertex: {row['point']}")
    print(f"  Formed by: {', '.join(row['equations'])}")
    print(f"  Active constraints: {', '.join(row['active_planes'])}")
    print(f"  Degenerate: {row['degenerate']}")
    print(f"  Objective value: {row['objective_value']}")
    print()

basic_solutions = add_slack_variables()
compare_solutions(vertex_df, basic_solutions)

# Verification --------------------------------------------- only for verification, not part of the solution-------------------------
c = [8, 5, 4]
A_ub = [[-1, -1, 0], [0, -1, -1], [-1, 0, -1], [20, 10, 15]]
b_ub = [-10, -15, -12, 300]
bounds = [(0, None), (0, None), (0, None)]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
print("\nVerification with scipy.optimize:")
print(f"Optimal solution: x1 = {result.x[0]}, x2 = {result.x[1]}, x3 = {result.x[2]}")
print(f"Optimal objective value: {result.fun}")
