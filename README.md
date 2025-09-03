# Grid Optimization Solvers

This repository contains a collection of algorithms designed to solve grid-based pairing problems under adjacency and color constraints. The project includes multiple solver implementations and benchmarking tools to evaluate performance across different scenarios.

## Overview

The optimization problem involves selecting valid pairs of adjacent cells in a grid, subject to color compatibility and forbidden zones. Each solver aims to minimize the cost, defined as the sum of absolute differences between paired cell values, plus the value of unpaired cells.

## Solvers

- **Greedy Solver**: Selects pairs with minimal cost in a greedy fashion. Fast but not guaranteed to be optimal.
- **Fulkerson Solver**: Uses the Ford-Fulkerson algorithm on a bipartite graph for optimal matching when all cell values are equal.
- **General Solver**: Applies maximum weight matching using NetworkX for grids with arbitrary values and constraints.

## Features

- Grid parsing from structured input files
- Color-based pairing rules and forbidden cell handling
- Visualization of grid states using Pygame
- Unit tests for grid loading and solver correctness
- Performance comparison across solver strategies

## Technologies

- Python (NumPy, NetworkX, Pygame, Matplotlib)
- Object-oriented design with modular architecture
- BFS and flow-based graph algorithms

## Usage

1. Place grid input files in the `input/` directory.
2. Run `main.py` to visualize and test solver outputs.
3. Use `compare()` to benchmark Greedy vs. Fulkerson on selected grids.

## License

This project is licensed under the MIT License.

