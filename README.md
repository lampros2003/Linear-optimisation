# Linear Optimisation Project

## Overview
This repository contains code and analysis for a linear programming homework project.
The project resulted in a complete CLI app which can parse any solveble linear programming problem in canonical form, and provide it's ideal solution, and an exploration of all possible solutions throught the form of a simplex adjacency graph.

## Features

- Solve linear programming problems using the simplex method
- Support for loading problems from text files in canonical form
- Visualize all possible pivot paths in a graph
- Handle both maximization and minimization problems
- Support for Unicode subscript notation (e.g., x₁, x₂, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Linear-optimisation.git
cd Linear-optimisation
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Basic Usage

Run the simplex algorithm on a problem defined in a text file:

```bash
python ask6_a.py --file problem.txt
```

### Explore All Paths

To solve the problem and visualize all possible pivot paths:

```bash
python ask6_a.py --file problem.txt --explore
```

## Input File Format

Create a text file with your linear programming problem in the following format:

```
[max/min]: [objective function]
subject to:
[constraint 1]
[constraint 2]
...
[constraint n]
```

### Example

```
max: 2x₁ + x₂ + 6x₃ - 4x₄
subject to:
x₁ + 2x₂ + 4x₃ - x₄ <= 6
2x₁ + 3x₂ - x₃ + x₄ <= 12
x₁ + x₃ + x₄ <= 2
```
Notes:
- The first line defines the objective function (max or min)
- Use `<=`, `>=`, or `=` for constraints
- Variables should be in the form `x` followed by a number (can use subscripts like `x₁`)
- Coefficients can be positive or negative
## Dependencies
The project requires the following Python libraries:

- contourpy==1.3.1
- cycler==0.12.1
- fonttools==4.57.0
- kiwisolver==1.4.8
- matplotlib==3.10.1
- mpmath==1.3.0
- networkx==3.4.2
- numpy==2.2.4
- packaging==24.2
- pandas==2.2.3
- pillow==11.1.0
- pyparsing==3.2.3
- python-dateutil==2.9.0.post0
- pytz==2025.2
- scipy==1.15.2
- six==1.17.0
- sympy==1.13.3
- tzdata==2025.2

