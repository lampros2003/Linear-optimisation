from collections import deque
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def parse_problem_from_file(file_path):
    """Parse a linear programming problem from a text file.

    Args:
        file_path (str): Path to the file containing the LP problem in canonical form

    Returns:
        tuple: (objective coeffs, constraint coeffs, RHS values, variable names, slack names)
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]

    # Parse objective function (first line)
    obj_line = lines[0].lower()
    is_max = "max" in obj_line

    # Extract objective function coefficients
    obj_parts = obj_line.split(":", 1)[1].strip()
    obj_terms = _parse_expression(obj_parts)

    # Parse constraints
    constraints = []
    rhs_values = []

    for line in lines[1:]:
        if "subject to" in line.lower():
            continue

        if "<=" in line:
            lhs, rhs = line.split("<=")
            constraints.append(_parse_expression(lhs.strip()))
            rhs_values.append(float(rhs.strip()))
        elif ">=" in line:
            lhs, rhs = line.split(">=")
            constraints.append(
                {var: -coef for var, coef in _parse_expression(lhs.strip()).items()}
            )
            rhs_values.append(-float(rhs.strip()))
        elif "=" in line and "==" not in line:
            lhs, rhs = line.split("=")
            constraints.append(_parse_expression(lhs.strip()))
            rhs_values.append(float(rhs.strip()))

    # Identify all variables
    all_vars = set()
    for terms in [obj_terms] + constraints:
        all_vars.update(terms.keys())

    # Remove 'const' if it's in the variables
    if "const" in all_vars:
        all_vars.remove("const")

    # Modified sorting to handle Unicode subscripts
    def var_sort_key(var_name):
        # Get the first character (typically 'x')
        prefix = var_name[0]

        # Get the index part (may contain subscripts)
        suffix = var_name[1:]

        # Try to extract a number for sorting
        try:
            # For standard digits
            if suffix.isdigit():
                return (prefix, int(suffix))
            # For subscript digits (₁, ₂, etc.)
            elif suffix:
                # Simple handling - just use the Unicode code point for ordering
                return (prefix, ord(suffix[0]))
            else:
                return (prefix, 0)
        except:
            # Fall back to string comparison if anything fails
            return (prefix, suffix)

    # Sort variables using the custom key function
    var_names = sorted(all_vars, key=var_sort_key)

    # Create slack variable names
    slack_names = [f"s{i+1}" for i in range(len(constraints))]

    # Convert to coefficient matrices
    obj_coeffs = [0]  # Z coefficient
    for var in var_names:
        obj_coeffs.append(-obj_terms.get(var, 0) if is_max else obj_terms.get(var, 0))

    # Constraints matrix
    constraint_coeffs = []
    for constraint in constraints:
        row = [0]  # Z coefficient
        for var in var_names:
            row.append(constraint.get(var, 0))
        constraint_coeffs.append(row)

    return obj_coeffs, constraint_coeffs, rhs_values, var_names, slack_names, is_max


def _parse_expression(expr):
    """Parse an expression like '2x₁ + 3x₂ - x₃' into a dictionary of {var: coef}"""
    terms = {}
    expr = expr.replace("-", "+-").replace("++", "+")
    if expr.startswith("+"):
        expr = expr[1:]

    for term in expr.split("+"):
        term = term.strip()
        if not term:
            continue

        # Handle negative terms
        if term.startswith("-"):
            term = term[1:]
            factor = -1
        else:
            factor = 1

        # Find variable and coefficient
        if "x" in term:
            parts = term.split("x", 1)
            coef_str = parts[0].strip()
            var_name = "x" + parts[1].strip()

            if coef_str == "" or coef_str == "-":
                coef = factor * 1
            else:
                coef = factor * float(coef_str)
        else:
            # Constant term
            var_name = "const"
            coef = factor * float(term)

        terms[var_name] = (
            terms.get(var_name, 0) + coef
        )  # Sum coefficients if variable appears multiple times

    return terms


def create_initial_tableau(file_path=None):
    """Create the initial tableau for the given problem.

    Args:
        file_path (str, optional): Path to the problem definition. 
        If None, uses the default problem.

    Returns:
        tuple: (tableau as numpy array, list of basic variable names, all variable names)
    """
    if file_path is None:
        # Use the original hardcoded example
        tableau = np.array(
            [
                [1, -2, -1, -6, 4, 0, 0, 0, 0],
                [0, 1, 2, 4, -1, 1, 0, 0, 6],
                [0, 2, 3, -1, 1, 0, 1, 0, 12],
                [0, 1, 0, 1, 1, 0, 0, 1, 2],
            ],
            dtype=float,
        )
        var_names = ["x₁", "x₂", "x₃", "x₄"]
        slack_names = ["s₁", "s₂", "s₃"]
        is_max = True
    else:
        # Parse problem from file
        obj_coeffs, constraint_coeffs, rhs_values, var_names, slack_names, is_max = (
            parse_problem_from_file(file_path)
        )

        # Create identity matrix for slack variables
        n_constraints = len(constraint_coeffs)
        identity = np.eye(n_constraints)

        # Build tableau
        tableau = np.zeros((n_constraints + 1, len(obj_coeffs) + n_constraints + 1))

        # Set objective row
        tableau[0, : len(obj_coeffs)] = obj_coeffs

        # Set constraint rows
        for i, constraint in enumerate(constraint_coeffs):
            tableau[i + 1, : len(constraint)] = constraint
            # Add slack variables
            tableau[i + 1, len(obj_coeffs) + i] = 1
            # Set RHS
            tableau[i + 1, -1] = rhs_values[i]

    # Create basic variable names
    basic_vars = ["Z"] + slack_names

    # Create full list of column names for display
    col_names = ["Z"] + var_names + slack_names + ["RHS"]

    return tableau, basic_vars, col_names


##pretty print our tableaux
def display_tableau(tableau, basic_vars, col_names):
    """Display the tableau with proper variable names."""
    df = pd.DataFrame(tableau, columns=col_names)
    df.insert(0, "Basic", basic_vars)
    return df


def find_entering_variables(tableau):
    """Find potential entering variables based on the tableau."""
    
    # Find all <0 coefficients in objective row (potential entering vars)
    objective_coeffs = tableau[0, 1:-1]
    negative_indices = np.where(objective_coeffs < 0)[0]
    entering_candidates = [idx + 1 for idx in negative_indices]

    # Sort by absolute value
    return sorted(entering_candidates, key=lambda x: tableau[0, x])


def find_leaving_variables(tableau, entering_col):
    """Find potential leaving variables based on the tableau."""
    
    
    
    
    # Calculate ratios for minimum ratio test
    ratios = []
    for i in range(1, len(tableau)):
        if tableau[i, entering_col] <= 0:
            ratios.append((i, float("inf")))
        else:
            ratios.append((i, tableau[i, -1] / tableau[i, entering_col]))

    # Filter out ινφ ratios (unbounded)
    valid_ratios = [r for r in ratios if r[1] != float("inf")]

    if not valid_ratios:
        return []  # No  leaving variables (unbounded)

    min_ratio = min(r[1] for r in valid_ratios)

    # Find all rows with minimum ratio (potential leaving variables)
    leaving_candidates = [row for row, ratio in ratios if ratio == min_ratio]

    return leaving_candidates


def pivot(tableau, basic_vars, entering_col, leaving_row, col_names):
    """Perform a pivot operation on the tableau."""
    new_tableau = tableau.copy()
    new_basic_vars = basic_vars.copy()

    # Get the pivot element
    pivot_element = tableau[leaving_row, entering_col]

    # Update basic
    new_basic_vars[leaving_row - 1] = col_names[entering_col]

    # Normalize pivot
    new_tableau[leaving_row] = tableau[leaving_row] / pivot_element

    # Update other rows
    for i, row in enumerate(tableau):
        if i != leaving_row:
            factor = row[entering_col]
            new_tableau[i] = row - factor * new_tableau[leaving_row]

    return new_tableau, new_basic_vars


def is_optimal(tableau):
    # Check if all coefficients in objective row are non-negative
    objective_coeffs = tableau[0, 1:-1]
    return np.all(objective_coeffs >= 0)


def solve_simplex(file_path=None):
    """Solve the linear programming problem using the simplex method."""
    tableau, basic_vars, col_names = create_initial_tableau(file_path)

    print("Initial Tableau:")
    print(display_tableau(tableau, basic_vars, col_names))

    iteration = 0
    while not is_optimal(tableau):
        iteration += 1
        print(f"\nIteration {iteration}:")

        # Find potential entering variables
        entering_candidates = find_entering_variables(tableau)
        print(
            f"Potential entering variables: {[col_names[col] for col in entering_candidates]}"
        )

        # Select the entering variable (most negative)
        entering_col = entering_candidates[0]
        entering_var = col_names[entering_col]
        print(f"Selected entering variable: {entering_var}")

        # Find potential leaving vars
        leaving_candidates = find_leaving_variables(tableau, entering_col)

        if not leaving_candidates:
            print("Problem is unbounded!")
            return

        leaving_vars = [basic_vars[row - 1] for row in leaving_candidates]
        print(f"Potential leaving variables: {leaving_vars}")

        # Select the leaving variable
        leaving_row = leaving_candidates[0]
        leaving_var = basic_vars[leaving_row - 1]
        print(f"Selected leaving variable: {leaving_var}")

        # Perform pivot
        tableau, basic_vars = pivot(
            tableau, basic_vars, entering_col, leaving_row, col_names
        )
        print("New tableau:")
        print(display_tableau(tableau, basic_vars, col_names))

    # Extract solution
    print("\nOptimal solution found!")
    solution = {}
    for i, var in enumerate(basic_vars[1:]):  # Skip Z
        solution[var] = tableau[i + 1, -1]

    # Set non-basic variables to zero
    var_names = col_names[1:-1]  # Exclude Z and RHS
    for var in var_names:
        if var not in basic_vars:
            solution[var] = 0

    print("\nSolution:")
    for var in col_names[
        1 : len(col_names) - len(basic_vars)
    ]:  # Show only original variables
        print(f"{var} = {solution.get(var, 0)}")

    print(f"\nObjective value: Z = {tableau[0, -1]}")


def explore_all_paths(file_path=None):
    """Explore all possible pivot paths from initial tableau to optimal solutions."""
    # Initialize a directed graph
    graph = nx.DiGraph()

    # Create initial tableau and basic variables
    tableau, basic_vars, col_names = create_initial_tableau(file_path)

    # Use a queue for breadth-first exploration
    # (tableau, basic_vars, iteration)
    queue = deque([(tableau, basic_vars, 0)])
    # Track visited states
    visited = set()
    # Map state keys to node IDs for better visualization

    node_map = {}
    while queue:
        current_tableau, current_basic_vars, iteration = queue.popleft()

        # Create a key for the current state
        values = [
            f"{current_basic_vars[i+1]}={current_tableau[i+1,-1]:.2f}"
            for i in range(len(current_basic_vars) - 1)
        ]
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
            graph.add_node(
                node_id,
                label=node_label,
                iteration=iteration,
                optimal=is_optimal(current_tableau),
                z_value=z_value,
            )

        # Get current node ID
        current_node_id = node_map[state_key]

        # If solution is optimal, mark it and continue
        if is_optimal(current_tableau):
            graph.nodes[current_node_id]["optimal"] = True
            continue

        # Find all possible entering
        entering_candidates = find_entering_variables(current_tableau)

        # Explore each entering variable option
        for entering_col in entering_candidates:
            # Use col_names to get the entering variable name
            entering_var = col_names[entering_col]

            # Find all possible leaving variables
            leaving_candidates = find_leaving_variables(current_tableau, entering_col)

            if not leaving_candidates:
                # Unbounded problem for this path
                continue

            # Explore each leaving variable
            for leaving_row in leaving_candidates:
                leaving_var = current_basic_vars[leaving_row - 1]

                # Create new tableau and basic variables after pivot
                new_tableau, new_basic_vars = pivot(
                    current_tableau,
                    current_basic_vars.copy(),
                    entering_col,
                    leaving_row,
                    col_names,  # Pass the col_names parameter
                )

                # Add to queue
                queue.append((new_tableau, new_basic_vars, iteration + 1))

                # Create key for the next state
                new_values = [
                    f"{new_basic_vars[i+1]}={new_tableau[i+1,-1]:.2f}"
                    for i in range(len(new_basic_vars) - 1)
                ]
                new_state_key = tuple(new_values)

                # Create node for the next state if needed
                if new_state_key not in node_map:
                    new_node_id = len(node_map)
                    node_map[new_state_key] = new_node_id

                    # Create node label
                    new_basic_vars_str = ", ".join(new_values)
                    new_z_value = new_tableau[0, -1]
                    new_node_label = (
                        f"Node {new_node_id}\n{new_basic_vars_str}\nZ={new_z_value:.2f}"
                    )

                    # Add node to graph
                    graph.add_node(
                        new_node_id,
                        label=new_node_label,
                        iteration=iteration + 1,
                        optimal=is_optimal(new_tableau),
                        z_value=new_z_value,
                    )

                # Get next node ID
                next_node_id = node_map[new_state_key]

                # Add edge between nodes
                graph.add_edge(
                    current_node_id,
                    next_node_id,
                    label=f"+{entering_var}/-{leaving_var}",
                )

    # Draw the graph
    plt.figure(figsize=(18, 14))  # Larger figure size

    pos = nx.spring_layout(graph, k=0.5, iterations=100, seed=42)

    # Adjust node positions to prevent overlap
    iteration_levels = {}
    for node, data in graph.nodes(data=True):
        level = data.get("iteration", 0)
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
    optimal_nodes = [n for n, d in graph.nodes(data=True) if d.get("optimal", False)]
    other_nodes = [n for n in graph.nodes() if n not in optimal_nodes]

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=other_nodes,
        node_color="skyblue",
        node_size=3000,
        alpha=0.8,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=optimal_nodes,
        node_color="lightgreen",
        node_size=3500,
        alpha=0.8,
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        width=1.5,
        arrowsize=15,
        arrowstyle="->",
        edge_color="gray",
        connectionstyle="arc3,rad=0.1",
    )  # Curved edges to avoid overlap

    # Draw node labels
    node_labels = {n: d["label"] for n, d in graph.nodes(data=True)}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=node_labels,
        font_size=9,
        font_family="sans-serif",
        font_weight="bold",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=4),
    )

    # Draw edge labels
    edge_labels = {(u, v): d["label"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3
    )  # Adjust label position on edge

    # Add  legend
    plt.plot(
        [0], [0], "o", color="skyblue", label="Intermediate Solutions", ms=15, alpha=0.8
    )
    plt.plot(
        [0], [0], "o", color="lightgreen", label="Optimal Solutions", ms=15, alpha=0.8
    )
    plt.legend(loc="best", fontsize=12)

    # Add  title
    plt.title("Simplex Adjacency graph - All Possible Paths", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Save and show  graph
    plt.savefig("simplex_adjacency_graph.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary
    print("\nSimplex Adjacency graph Summary:")
    print(f"Total nodes (basic feasible solutions): {len(graph.nodes())}")
    print(f"Total edges (pivot operations): {len(graph.edges())}")
    print(f"Optimal solutions found: {len(optimal_nodes)}")

    # List all optimal solutions
    print("\nOptimal Solutions:")
    for node in optimal_nodes:
        z_value = graph.nodes[node]["z_value"]
        print(f"Node {node}: Z = {z_value}")
        print(graph.nodes[node]["label"])
        print("---")


# Main script execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Solve a linear programming problem using the simplex method."
    )
    parser.add_argument("--file", type=str, help="Path to the problem definition file")
    parser.add_argument(
        "--explore", action="store_true", help="Explore all possible pivot paths"
    )
    args = parser.parse_args()

    # First run standard simplex
    print("Step-by-step execution of Simplex algorithm")
    solve_simplex(args.file)

    # If the explore flag is set, run the exploration of all paths
    if args.explore:
        print("\n\nExploring all possible Simplex paths")
        explore_all_paths(args.file)
