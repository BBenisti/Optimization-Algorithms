# This will work if ran from the root folder (the folder in which there is the subfolder code/)
import sys
import os

# Ajoute dynamiquement le chemin absolu du dossier "code"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))

import unittest 
from grid import Grid
from solver import *

class Test_GridLoading(unittest.TestCase):
    def test_grid0(self):
        grid = Grid.grid_from_file("input/grid00.in",read_values=True)
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        self.assertEqual(grid.color, [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(grid.value, [[5, 8, 4], [11, 1, 3]])

    def test_grid0_novalues(self):
        grid = Grid.grid_from_file("input/grid00.in",read_values=False)
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        self.assertEqual(grid.color, [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(grid.value, [[1, 1, 1], [1, 1, 1]])

    def test_grid1(self):
        grid = Grid.grid_from_file("input/grid01.in",read_values=True)
        self.assertEqual(grid.n, 2)
        self.assertEqual(grid.m, 3)
        self.assertEqual(grid.color, [[0, 4, 3], [2, 1, 0]])
        self.assertEqual(grid.value, [[5, 8, 4], [11, 1, 3]])

class Test_functions(unittest.TestCase):

    def test_grid0(self):
        grid = Grid.grid_from_file("input/grid00.in",read_values=True)
        
        #Tests is_forbidden
        self.assertFalse(grid.is_forbidden(0,0))
        self.assertFalse(grid.is_forbidden(1,1))
        
        #Test all_pairs (contains every possible combinations)
        L = []
        for i1 in range(grid.n):
            for j1 in range(grid.m):
                for i2 in range(grid.n):
                    for j2 in range(grid.m):
                        if i1 == i2 and abs(j1 - j2) == 1 or j1 == j2 and abs(i1 - i2) == 1: #Test on coordonates
                            if ((i2,j2),(i1,j1)) not in L and ((i1,j1),(i2,j2)) not in L:
                                L.append(((i1,j1),(i2,j2)))
        self.assertEqual(grid.all_pairs(), L)

    def test_grid1(self):
        grid = Grid.grid_from_file("input/grid01.in",read_values=True)

        #Tests is_forbidden
        self.assertTrue(grid.is_forbidden(0,1))
        self.assertFalse(grid.is_forbidden(0,0))

        #Test all_pairs
        L = [((0,0),(1,0)),((0,2),(1,2)),((1,0),(1,1)),((1,1),(1,2))]
        self.assertEqual(grid.all_pairs(),L)

    def test_grid2(self):
        grid = Grid.grid_from_file("input/grid02.in",read_values=True)

        #Tests is_forbidden
        self.assertTrue(grid.is_forbidden(0,1))
        self.assertFalse(grid.is_forbidden(1,1))

        #Test all_pairs
        L = [((0,0),(1,0)),((0,2),(1,2)),((1,0),(1,1)),((1,1),(1,2))]
        self.assertEqual(grid.all_pairs(),L)

"""
In this section, we used the code main in order to watch the following
grids and make calculations of the optimal result by hand
"""

        
class Test_Solver(unittest.TestCase):

    def test_greedy(self):
        
        #with grid00
        grid = Grid.grid_from_file("input/grid00.in",read_values=True)
        solver = SolverGreedy(grid)
        self.assertEqual(solver.sum_diff(),14) #Even though it i not the optimal score, it is what greedy is supposed to return

        #with grid01
        grid = Grid.grid_from_file("input/grid01.in",read_values=True)
        solver = SolverGreedy(grid)
        self.assertEqual(solver.sum_diff(),8)

        #with grid02
        grid = Grid.grid_from_file("input/grid02.in",read_values=True)
        solver = SolverGreedy(grid)
        self.assertEqual(solver.sum_diff(),1)

        #with grid11
        grid = Grid.grid_from_file("input/grid11.in",read_values=True)
        solver = SolverGreedy(grid)
        self.assertEqual(solver.sum_diff(),26)

        #with grid13
        grid = Grid.grid_from_file("input/grid13.in",read_values=True)
        solver = SolverGreedy(grid)
        self.assertEqual(solver.sum_diff(),22)

    def test_fulkerson(self):
        """
        as asked, we will test our program only with grids in which values are all equal to 1 (Cf question 6)
        """
        #with grid01
        grid = Grid.grid_from_file("input/grid01.in",read_values=False)
        solver = SolverFulkerson(grid)
        solver.run()
        self.assertEqual(solver.score(),1)
        
        #with grid03
        grid = Grid.grid_from_file("input/grid03.in",read_values=False)
        solver = SolverFulkerson(grid)
        solver.run()
        self.assertEqual(solver.score(),2)

        #with grid04
        grid = Grid.grid_from_file("input/grid04.in",read_values=False)
        solver = SolverFulkerson(grid)
        solver.run()
        self.assertEqual(solver.score(),4)

        #with grid11
        grid = Grid.grid_from_file("input/grid11.in",read_values=False)
        solver = SolverFulkerson(grid)
        solver.run()
        self.assertEqual(solver.score(),26)

        #with grid13
        grid = Grid.grid_from_file("input/grid13.in",read_values=False)
        solver = SolverFulkerson(grid)
        solver.run()
        self.assertEqual(solver.score(),22)


    def test_general(self):
        # Test with grid00
        grid = Grid.grid_from_file("input/grid00.in", read_values=True)
        solver = SolverGeneral(grid)
        matching, score = solver.solve()
        self.assertEqual(score, 12)  

        # Test with grid01
        grid = Grid.grid_from_file("input/grid01.in", read_values=True)
        solver = SolverGeneral(grid)
        matching, score = solver.solve()
        self.assertEqual(score, 8)  

        # Test with grid02
        grid = Grid.grid_from_file("input/grid02.in", read_values=True)
        solver = SolverGeneral(grid)
        matching, score = solver.solve()
        self.assertEqual(score, 1) 
    
        # Test with grid05
        grid = Grid.grid_from_file("input/grid05.in", read_values=True)
        solver = SolverGeneral(grid)
        matching, score = solver.solve()
        self.assertEqual(score, 35)

if __name__ == '__main__':
    unittest.main()
