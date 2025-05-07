import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def create_initial_tableau():
    # Initial tableau with slack variables s1, s2, s3

    return np.array([
        [1, -2, -1, -6, 4, 0, 0, 0, 0],  
        [0, 1, 2, 4, -1, 1, 0, 0, 6],    
        [0, 2, 3, -1, 1, 0, 1, 0, 12],   
        [0, 1, 0, 1, 1, 0, 0, 1, 2]      
    ], dtype=float)
##pretty print our tableaux 
def display_tableau(tableau, basic_vars):
    columns = ['Z', 'x₁', 'x₂', 'x₃', 'x₄', 's₁', 's₂', 's₃', 'RHS']
    df = pd.DataFrame(tableau, columns=columns)
    df.insert(0, 'Basic', basic_vars)
    return df

def find_entering_variables(tableau):
    # Find all <0 coefficients in objective row (potential entering vars)
    objective_coeffs = tableau[0, 1:-1]  
    negative_indices = np.where(objective_coeffs < 0)[0]
    entering_candidates = [idx + 1 for idx in negative_indices]
    
    # Sort by absolute value
    return sorted(entering_candidates, key=lambda x: tableau[0, x])
    
def find_leaving_variables(tableau, entering_col):
    # Calculate ratios for minimum ratio test
    ratios = []
    for i in range(1, len(tableau)):
        if tableau[i, entering_col] <= 0:
            ratios.append((i, float('inf')))
        else:
            ratios.append((i, tableau[i, -1] / tableau[i, entering_col]))
    
    # Filter out ινφ ratios (unbounded)
    valid_ratios = [r for r in ratios if r[1] != float('inf')]
    
    if not valid_ratios:
        return []  # No  leaving variables (unbounded)
    
    min_ratio = min(r[1] for r in valid_ratios)
    
    # Find all rows with minimum ratio (potential leaving variables)
    leaving_candidates = [row for row, ratio in ratios if ratio == min_ratio]
    
    return leaving_candidates

def pivot(tableau, basic_vars, entering_col, leaving_row):
   
    new_tableau = tableau.copy()
    new_basic_vars = basic_vars.copy()
    
    # Get the pivot element
    pivot_element = tableau[leaving_row, entering_col]
    
    #Update basic 
    col_names = ['Z', 'x₁', 'x₂', 'x₃', 'x₄', 's₁', 's₂', 's₃']
    new_basic_vars[leaving_row-1] = col_names[entering_col]
    
    # Normalize pivot 
    new_tableau[leaving_row] = tableau[leaving_row] / pivot_element
    
    # Update  other rows
    for i in range(len(tableau)):
        if i != leaving_row:
            factor = tableau[i, entering_col]
            new_tableau[i] = tableau[i] - factor * new_tableau[leaving_row]
    
    return new_tableau, new_basic_vars

def is_optimal(tableau):
    # Check if all coefficients in objective row are non-negative
    objective_coeffs = tableau[0, 1:-1]  
    return np.all(objective_coeffs >= 0)

def solve_simplex():
    tableau = create_initial_tableau()
    basic_vars = ['Z', 's₁', 's₂', 's₃']
    
    print("Initial Tableau:")
    print(display_tableau(tableau, basic_vars))
    
    iteration = 0
    symbols_lst = ['Z', 'x₁', 'x₂', 'x₃', 'x₄', 's₁', 's₂', 's₃']
    while not is_optimal(tableau):
        iteration += 1
        print(f"\nIteration {iteration}:")
        
        # Find potential entering variables
        entering_candidates = find_entering_variables(tableau)
        print(f"Potential entering variables: {[symbols_lst[col] for col in entering_candidates]}")
        
        # Select the entering variable (most negative )
        entering_col = entering_candidates[0]
        entering_var = symbols_lst[entering_col]
        print(f"Selected entering variable: {entering_var}")
        
        #Find potential leaving vars
        leaving_candidates = find_leaving_variables(tableau, entering_col)
        
        if not leaving_candidates:
            print("Problem is unbounded!")
            return
        
        leaving_vars = [basic_vars[row-1] for row in leaving_candidates]
        print(f"Potential leaving variables: {leaving_vars}")
        
        #Select the leaving variable ,first candidate no special evaluation
        leaving_row = leaving_candidates[0]
        leaving_var = basic_vars[leaving_row-1]
        print(f"Selected leaving variable: {leaving_var}")
        
        # Perform pivot
        tableau, basic_vars = pivot(tableau, basic_vars, entering_col, leaving_row)
        print("New tableau:")
        print(display_tableau(tableau, basic_vars))
    
    # Extract solution
    print("\nOptimal solution found :)!")
    solution = {}
    for i, var in enumerate(basic_vars[1:]):  # Skip Z
        solution[var] = tableau[i+1, -1]
    
    # Set non-basic  to zero
    all_vars = ['x₁', 'x₂', 'x₃', 'x₄', 's₁', 's₂', 's₃']
    for var in all_vars:
        if var not in basic_vars:
            solution[var] = 0
    
    print("\nSolution:")
    for var in ['x₁', 'x₂', 'x₃', 'x₄']:
        print(f"{var} = {solution.get(var, 0)}")
    
    print(f"\nObjective value: Z = {tableau[0, -1]}")

def explore_all_paths():
    # Initialize a directed graph
    G = nx.DiGraph()
    
    # Create initial tableau and basic variables
    initial_tableau = create_initial_tableau()
    initial_basic_vars = ['Z', 's₁', 's₂', 's₃']
    
    # Use a queue for breadth-first exploration 
    # (tableau, basic_vars, iteration)
    queue = deque([(initial_tableau, initial_basic_vars, 0)])
     # Track visited states
    visited = set() 
    # Map state keys to node IDs for better visualization
    
    node_map = {}  
    while queue:
        current_tableau, current_basic_vars, iteration = queue.popleft()
        
        # Create a key for the current state 
        values = [f"{current_basic_vars[i+1]}={current_tableau[i+1,-1]:.2f}" 
                 for i in range(len(current_basic_vars)-1)]
        state_key = tuple(values)
        
        # Skip if already visited
        if state_key in visited:
            continue
            
        # Mark as visited
        visited.add(state_key)
        
        # Generate a node if needed
        if state_key not in node_map:
            node_id = len(node_map)
            node_map[state_key] = node_id
            
            # Create node label
            basic_vars_str = ", ".join(values)
            z_value = current_tableau[0, -1]
            node_label = f"Node {node_id}\n{basic_vars_str}\nZ={z_value:.2f}"
            
            # Add node to graph
            G.add_node(node_id, label=node_label, iteration=iteration, 
                       optimal=is_optimal(current_tableau), z_value=z_value)
        
        # Get current node ID
        current_node_id = node_map[state_key]
        
        # If solution is optimal, mark it and continue
        if is_optimal(current_tableau):
            G.nodes[current_node_id]['optimal'] = True
            continue
        
        # Find all possible entering 
        entering_candidates = find_entering_variables(current_tableau)
        
        # Explore each entering variable option
        for entering_col in entering_candidates:
            entering_var = ['Z', 'x₁', 'x₂', 'x₃', 'x₄', 's₁', 's₂', 's₃'][entering_col]
            
            # Find all possible leaving variables
            leaving_candidates = find_leaving_variables(current_tableau, entering_col)
            
            if not leaving_candidates:
                # Unbounded problem for this path
                continue
            
            # Explore each leaving variable 
            for leaving_row in leaving_candidates:
                leaving_var = current_basic_vars[leaving_row-1]
                
                # Create new tableau and basic variables after pivot
                new_tableau, new_basic_vars = pivot(
                    current_tableau, current_basic_vars.copy(), 
                    entering_col, leaving_row
                )
                
                # Add to queue 
                queue.append((new_tableau, new_basic_vars, iteration + 1))
                
                # Create key for the next state
                new_values = [f"{new_basic_vars[i+1]}={new_tableau[i+1,-1]:.2f}" 
                             for i in range(len(new_basic_vars)-1)]
                new_state_key = tuple(new_values)
                
                # Create node for the next state if needed
                if new_state_key not in node_map:
                    new_node_id = len(node_map)
                    node_map[new_state_key] = new_node_id
                    
                    # Create node label
                    new_basic_vars_str = ", ".join(new_values)
                    new_z_value = new_tableau[0, -1]
                    new_node_label = f"Node {new_node_id}\n{new_basic_vars_str}\nZ={new_z_value:.2f}"
                    
                    # Add node to graph
                    G.add_node(new_node_id, label=new_node_label, iteration=iteration+1, 
                              optimal=is_optimal(new_tableau), z_value=new_z_value)
                
                # Get next node ID
                next_node_id = node_map[new_state_key]
                
                # Add edge between nodes
                G.add_edge(current_node_id, next_node_id, 
                          label=f"+{entering_var}/-{leaving_var}")
    
    # Draw the graph 
    plt.figure(figsize=(18, 14))  # Larger figure size
    
    
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)  
    
    # Adjust node positions to prevent overlap
    iteration_levels = {}
    for node, data in G.nodes(data=True):
        level = data.get('iteration', 0)
        if level not in iteration_levels:
            iteration_levels[level] = []
        iteration_levels[level].append(node)
    
    # Adjust positions based on iteration levels
    for level, nodes in iteration_levels.items():
        if len(nodes) > 1:
            # Spread nodes at same level horizontally
            base_y = 0.2 * level  
            spacing = 1.0 / (len(nodes) + 1)
            for i, node in enumerate(nodes):
                x_pos = -0.5 + (i + 1) * spacing
                pos[node] = np.array([x_pos, base_y])
    
    # Draw nodes with different colors 
    optimal_nodes = [n for n, d in G.nodes(data=True) if d.get('optimal', False)]
    other_nodes = [n for n in G.nodes() if n not in optimal_nodes]
    
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, 
                          node_color='skyblue', node_size=3000, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=optimal_nodes, 
                          node_color='lightgreen', node_size=3500, alpha=0.8)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrows=True, width=1.5, 
                          arrowsize=15, arrowstyle='->', edge_color='gray',
                          connectionstyle='arc3,rad=0.1')  # Curved edges to avoid overlap
    
    # Draw node labels 
    node_labels = {n: d['label'] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, 
                           font_family='sans-serif', font_weight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=4))
    
    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                                label_pos=0.3)  # Adjust label position on edge
    
    # Add  legend
    plt.plot([0], [0], 'o', color='skyblue', label='Intermediate Solutions', ms=15, alpha=0.8)
    plt.plot([0], [0], 'o', color='lightgreen', label='Optimal Solutions', ms=15, alpha=0.8)
    plt.legend(loc='best', fontsize=12)
    
    # Add  title
    plt.title('Simplex Adjacency Graph - All Possible Paths', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Save and show  graph
    plt.savefig('simplex_adjacency_graph.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\nSimplex Adjacency Graph Summary:")
    print(f"Total nodes (basic feasible solutions): {len(G.nodes())}")
    print(f"Total edges (pivot operations): {len(G.edges())}")
    print(f"Optimal solutions found: {len(optimal_nodes)}")
    
    # List all optimal solutions
    print("\nOptimal Solutions:")
    for node in optimal_nodes:
        z_value = G.nodes[node]['z_value']
        print(f"Node {node}: Z = {z_value}")
        print(G.nodes[node]['label'])
        print("---")

# Main script execution
if __name__ == "__main__":
    # First run standard simplex 
    print("Part (a): Step-by-step execution of Simplex algorithm")
    solve_simplex()
    
    # Then explore all possible paths
    print("\n\nPart (b): Exploring all possible Simplex paths")
    explore_all_paths()