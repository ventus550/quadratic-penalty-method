import pulp
import numpy as np
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from upsetplot import UpSet, plot, from_indicators
import matplotlib.pyplot as plt


# Function to create a binary indicator matrix for sets covering elements
def create_indicator_matrix(sets, num_items):
    indicator_matrix = torch.zeros((len(sets), num_items), dtype=torch.float32)
    for i, s in enumerate(sets):
        for elem in s:
            indicator_matrix[i, elem] = 1.0
    return indicator_matrix

# Function to generate a random set cover instance
def generate_set_cover_instance(num_sets, num_items):
    sets = []
    for _ in range(num_sets):
        # Each set covers a random number of unique elements
        set_size = torch.randint(1, num_items + 1, (1,)).item()
        covered_elements = torch.randint(0, num_items, (set_size,))
        sets.append(covered_elements.tolist())
    
    indicator_matrix = create_indicator_matrix(sets, num_items)
    if all(indicator_matrix.sum(axis=0) > 0):
        return indicator_matrix
    return generate_set_cover_instance(num_sets, num_items)

# Function to solve set cover using quadratic penalty
def torch_solver(indicator_matrix, penalty_weight=1.0, max_iterations=1000, tolerance=1e-5, p = 1):    
    # Decision variables: whether to include each set
    x = torch.zeros(len(indicator_matrix), requires_grad=True)
    
    optimizer = torch.optim.Adam([x], lr=0.1)

    for iteration in range(max_iterations):
        optimizer.zero_grad()
        penalty_weight *= 1.1
        
        # Calculate the coverage
        coverage = torch.matmul(x, indicator_matrix)
        
        # Penalty for uncovered elements
        uncovered_penalty = torch.sum(torch.relu(1 - coverage) ** p)

        # Penalty for non-binary solution
        nonbinary_penalty = torch.sum(torch.abs((x - 1) * x))
        
        # Objective: minimize the number of sets used plus the penalty
        objective = torch.sum(x) + penalty_weight * (uncovered_penalty + nonbinary_penalty)
        
        # Backpropagation
        objective.backward()
        optimizer.step()

        # Apply binary constraint to x
        with torch.no_grad():
            x.clamp_(0, 1)  # Ensuring x is in [0, 1]

        # Stopping condition
        if uncovered_penalty.item() < tolerance:
            # print(f"Converged in {iteration} iterations.")
            break
    return x  # Return binary decision variables

def pulp_solver(indicator_matrix):
	# Get the number of subsets (rows) and elements (columns)
	num_subsets, num_items = indicator_matrix.shape

	# Create a Pulp problem instance
	problem = pulp.LpProblem("SetCover", pulp.LpMinimize)

	# Create binary decision variables for each subset
	x = pulp.LpVariable.dicts('x', range(num_subsets), cat='Binary')

	# Objective function: Minimize the total cost of the selected subsets
	problem += pulp.lpSum(x)

	# Constraints: Each element in the universe must be covered by at least one subset
	for j in range(num_items):  # For each element in the universe
		problem += pulp.lpSum([indicator_matrix[i, j] * x[i] for i in range(num_subsets)]) >= 1, f"Cover_{j+1}"

	# Solve the problem
	problem.solve(pulp.PULP_CBC_CMD(msg=0))
	return torch.tensor([x[i].varValue for i in range(num_subsets)])

def upset_plot(data, orientation="horizontal"):
	return UpSet(from_indicators(pd.DataFrame(data.bool()).T), subset_size='count', orientation=orientation)