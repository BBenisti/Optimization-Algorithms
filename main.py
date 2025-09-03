from grid import *
from solver import *

"""
#Initial available code: Do not delete
grid = Grid(2, 3)
print(grid)

data_path = "input/"

file_name = data_path + "grid01.in"
grid = Grid.grid_from_file(file_name)
print(grid)

file_name = data_path + "grid01.in"
grid = Grid.grid_from_file(file_name, read_values=True)
print(grid)

solver = SolverEmpty(grid)
solver.run()
print("The final score of SolverEmpty is:", solver.score())



#Compare Greedy and Fulkerson for grid WITH values

def compare(number):
    "number is a character which represents the number of the grid"
    path = 'input/grid' + number + '.in'
    grid = Grid.grid_from_file(path,read_values=True)
    solvF = SolverFulkerson(grid)
    solvG = SolverGreedy(grid)
    print("The result with Greedy's method is:", solvG.sum_diff())
    print("The result with Fulkerson's method is:", solvF.ford_fulkerson())
    return None

path = 'input/'
grid = Grid.grid_from_file(path + 'grid05.in',read_values=True)
print(grid.plot())
"""

if __name__ == "__main__":
    pygame.init()  # Initialize Pygame
    data_path = "input/"
    file_name = data_path + "grid04.in"  # Input file containing the grid data
    grid = Grid.grid_from_file(file_name, read_values=True)  # Create a grid from the input file
    visualizer = GridVisualizer(grid)  # Create a visualizer for the grid
    visualizer.run()  # Start the game loop

