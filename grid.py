import math
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import pygame
import sys

"""
This is the grid module. It contains the Grid class and its associated methods.
"""

class Grid():
    """
    A class representing the grid. 

    Attributes: 
    -----------
    n: int
        Number of lines in the grid
    m: int
        Number of columns in the grid
    color: list[list[int]]
        The color of each grid cell: value[i][j] is the value in the cell (i, j), i.e., in the i-th line and j-th column. 
        Note: lines are numbered 0..n-1 and columns are numbered 0..m-1.
    value: list[list[int]]
        The value of each grid cell: value[i][j] is the value in the cell (i, j), i.e., in the i-th line and j-th column. 
        Note: lines are numbered 0..n-1 and columns are numbered 0..m-1.
    colors_list: list[char]
        The mapping between the value of self.color[i][j] and the corresponding color
    """
    

    def __init__(self, n, m, color=[], value=[]):
        """
        Initializes the grid.

        Parameters: 
        -----------
        n: int
            Number of lines in the grid
        m: int
            Number of columns in the grid
        color: list[list[int]]
            The grid cells colors. Default is empty (then the grid is created with each cell having color 0, i.e., white).
        value: list[list[int]]
            The grid cells values. Default is empty (then the grid is created with each cell having value 1).
        
        The object created has an attribute colors_list: list[char], which is the mapping between the value of self.color[i][j] and the corresponding color
        """
        self.n = n
        self.m = m
        if not color:
            color = [[0 for j in range(m)] for i in range(n)]            
        self.color = color
        if not value:
            value = [[1 for j in range(m)] for i in range(n)]            
        self.value = value
        self.colors_list = ['w', 'r', 'b', 'g', 'k']

    def __str__(self): 
        """
        Prints the grid as text.
        """
        output = f"The grid is {self.n} x {self.m}. It has the following colors:\n"
        for i in range(self.n): 
            output += f"{[self.colors_list[self.color[i][j]] for j in range(self.m)]}\n"
        output += f"and the following values:\n"
        for i in range(self.n): 
            output += f"{self.value[i]}\n"
        return output

    def __repr__(self): 
        """
        Returns a representation of the grid with number of rows and columns.
        """
        return f"<grid.Grid: n={self.n}, m={self.m}>"

    def plot(self): 
        """
        Plots a visual representation of the grid.
        """
        # On crée un dictionnaire contenant le codage des couleurs
        color_map = {
            0: "white",  # Blanc
            1: "red",    # Rouge
            2: "blue",   # Bleu
            3: "green",  # Vert
            4: "black"   # Noir
        }

        # On crée une figure et un axe
        fig, ax = plt.subplots(figsize=(self.m, self.n)) #Crée un tableau de taille m.n
        ax.set_xlim(0, self.m) #limite horizontale du tableau
        ax.set_ylim(0, self.n) #limite verticale du tableau

        # Dessiner la grille
        for i in range(self.n):
            for j in range(self.m):
                color = color_map[self.color[i][j]]
                rect = plt.Rectangle((j, self.n - i - 1), 1, 1, color=color, edgecolor='black') #Crée un rectangle à bord noir, de taille 1x1
                ax.add_patch(rect)
                ax.text(j + 0.5, self.n - i - 0.5, str(self.value[i][j]), #Affiche la valeur de la case au centre de cette dernière(en blanc si case noire et noir sinon)
                        color="black" if self.color[i][j] != 4 else "white",
                        ha='center', va='center', fontsize=8)

        # Configurer l'affichage
        ax.set_xticks(np.arange(0, self.m + 1, 1))
        ax.set_yticks(np.arange(0, self.n + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='both', color='black', linewidth=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def is_forbidden(self, i, j):
        """
        Returns True if the cell (i, j) is black and False otherwise
        """
        if self.color[i][j] == 4:
            return True
        else:
            return False

    def cost(self, pair):
        """
        Returns the cost of a pair
 
        Parameters: 
        -----------
        pair: tuple[tuple[int]]
            A pair in the format ((i1, j1), (i2, j2))

        Output: 
        -----------
        cost: int
            the cost of the pair defined as the absolute value of the difference between their values
        """
        return abs(self.value[pair[0][0]][pair[0][1]] - self.value[pair[1][0]][pair[1][1]])


    def all_pairs(self):
        """
        Returns a list of all pairs of cells that can be taken together. 

        Outputs a list of tuples of tuples [(c1, c2), (c1', c2'), ...] where each cell c1 etc. is itself a tuple (i, j)
        """
        list_pair = []
        common_color = [0,1,2]
        green_cond = [0,3]
        for i1 in range(self.n):
            for j1 in range(self.m):
                for i2 in range(self.n):
                    for j2 in range(self.m):
                        if i1 == i2 and abs(j1 - j2) == 1 or j1 == j2 and abs(i1 - i2) == 1: #Test on coordonates
                            if self.is_forbidden(i1, j1) == False and self.is_forbidden(i2, j2) == False: #Test on black color condition
                                if self.color[i1][j1] in green_cond and self.color[i2][j2] in green_cond and ((i1,j1),(i2,j2)) not in list_pair and ((i2,j2),(i1,j1)) not in list_pair: #Test on green condition
                                    c1 = (i1,j1)
                                    c2 = (i2,j2)
                                    list_pair.append((c1, c2))
                                if self.color[i1][j1] in common_color and self.color[i2][j2] in common_color and ((i1,j1),(i2,j2)) not in list_pair and ((i2,j2),(i1,j1)) not in list_pair: #Test on red,blue,white condition
                                    c1 = (i1,j1)
                                    c2 = (i2,j2)
                                    list_pair.append((c1, c2))
                                
        return list_pair
    


                
    

    @classmethod
    def grid_from_file(cls, file_name, read_values=False): 
        """
        Creates a grid object from class Grid, initialized with the information from the file file_name.
        
        Parameters: 
        -----------
        file_name: str
            Name of the file to load. The file must be of the format: 
            - first line contains "n m" 
            - next n lines contain m integers that represent the colors of the corresponding cell
            - next n lines [optional] contain m integers that represent the values of the corresponding cell
        read_values: bool
            Indicates whether to read values after having read the colors. Requires that the file has 2n+1 lines

        Output: 
        -------
        grid: Grid
            The grid
        """
        with open(file_name, "r") as file:
            n, m = map(int, file.readline().split())
            color = [[] for i_line in range(n)]
            for i_line in range(n):
                line_color = list(map(int, file.readline().split()))
                if len(line_color) != m: 
                    raise Exception("Format incorrect")
                for j in range(m):
                    if line_color[j] not in range(5):
                        raise Exception("Invalid color")
                color[i_line] = line_color

            if read_values:
                value = [[] for i_line in range(n)]
                for i_line in range(n):
                    line_value = list(map(int, file.readline().split()))
                    if len(line_value) != m: 
                        raise Exception("Format incorrect")
                    value[i_line] = line_value
            else:
                value = []

            grid = Grid(n, m, color, value)
        return grid
    @classmethod
    def grid_from_file(cls, file_name, read_values=True):
        # Load grid data from a file
        with open(file_name, "r") as f:
            lines = f.readlines()
        
        # Read grid dimensions (n = rows, m = columns)
        n, m = map(int, lines[0].split())
        
        # Read color values for each cell
        color = [list(map(int, lines[i + 1].split())) for i in range(n)]
        
        # Read the cell values if specified (default to 1 if not)
        if read_values:
            value = [list(map(int, lines[i + 1 + n].split())) for i in range(n)]
        else:
            value = [[1] * m for _ in range(n)]
        
        return cls(n, m, color, value)
    def __init__(self, n, m, color, value):
        # Initialize grid with its size, colors, and values
        self.n = n
        self.m = m
        self.color = color
        self.value = value
        self.colors_list = [(255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0)]  # RGB values for colors
        self.permanently_selected = set()  # Track permanently selected cells

    def is_forbidden(self, i, j):
        # Check if a cell is forbidden (color value 4)
        return self.color[i][j] == 4

    def cost(self, pair):
        # Calculate the cost of pairing two cells based on their values
        (i1, j1), (i2, j2) = pair
        return abs(self.value[i1][j1] - self.value[i2][j2])
    
    def valid_pair(self, i1, j1, i2, j2):
        # Check if two cells can be paired according to the color and adjacency rules
        color1 = self.color[i1][j1]
        color2 = self.color[i2][j2]

        if self.is_forbidden(i1, j1) or self.is_forbidden(i2, j2):
            return False

        # Pairing rules based on colors
        if color1 == 0:  # White
            if color2 == 4:  # Black
                return False
        elif color1 == 1:  # Red
            if color2 not in [0, 1, 2]:  # White, Red, or Blue
                return False
        elif color1 == 2:  # Blue
            if color2 not in [0, 1, 2]:  # White, Red, or Blue
                return False
        elif color1 == 3:  # Green
            if color2 not in [0, 3]:  # White or Green
                return False
        
        # Check if cells are adjacent (horizontally or vertically)
        if (i1 == i2 and abs(j1 - j2) == 1) or (j1 == j2 and abs(i1 - i2) == 1):
            return True

        return False
    
    def compatible_color(self, paire1, paire2): 

        """
        Returns the compatibility of 2 cases 
        Parameters: 

        pair: tuple[tuple[int]]

            A pair in the format ((i1, j1), (i2, j2))
        -----------
        compatible_color: boolean
            accordingly to the rules, this function testsif 2 cases are allowed to be paired
            as a reminder : - green pairs with white/green
                            - blue pairs with white/blue/red
                            - red pairs with white/blue/red
                            - white pairs with every color except black
        """
        i1 , j1 = paire1 
        i2 , j2 = paire2 
        if self.color[i1][j1]==0 :
            return (self.is_forbidden(i2,j2))==False
        elif self.color[i1][j1]==1 :
            return self.color[i2][j2] in [0,1,2]
        elif self.color[i1][j1]==2 :
            return self.color[i2][j2] in [0,1,2]
        elif self.color[i1][j1]==3 :
            return self.color[i2][j2] in [0,3]
        else :
            return False
    
    def all_pairs(self):
        # Get all valid adjacent pairs in the grid
        pairs = []
        for i in range(self.n):
            for j in range(self.m):
                # Check rightward adjacency
                if j + 1 < self.m and self.valid_pair(i, j, i, j + 1):
                    pairs.append(((i, j), (i, j + 1)))
                # Check downward adjacency
                if i + 1 < self.n and self.valid_pair(i, j, i + 1, j):
                    pairs.append(((i, j), (i + 1, j)))
        return pairs

import pygame
import sys

class GridVisualizer:
    def __init__(self, grid):
        pygame.init()
        self.grid = grid
        self.screen = pygame.display.set_mode((grid.m * 100, grid.n * 100 + 150))
        pygame.display.set_caption("Grid Game")
        self.cell_size = 100
        self.selected_cells = []
        self.current_player = 1
        self.scores = {1: 0, 2: 0}
        self.selected_color = (139, 69, 19)
        self.game_over = False

    def draw_grid(self, final_message=None):
        self.screen.fill((255, 255, 255))
        # Police standard pour les textes
        font = pygame.font.Font(None, 36)

        for i in range(self.grid.n):
            for j in range(self.grid.m):
                color = self.selected_color if (i, j) in self.grid.permanently_selected else self.grid.colors_list[self.grid.color[i][j]]
                pygame.draw.rect(self.screen, color, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (0, 0, 0), (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size), 1)
                text = font.render(str(self.grid.value[i][j]), True, (0, 0, 0))
                self.screen.blit(text, (j * self.cell_size + 30, i * self.cell_size + 30))

        score_text = font.render(f"Player 1: {self.scores[1]} | Player 2: {self.scores[2]}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, self.grid.n * self.cell_size + 10))

        if final_message:
            # Police plus petite pour le message final
            small_font = pygame.font.Font(None, 24)
            penalty_text = small_font.render(final_message, True, (255, 0, 0))
            self.screen.blit(penalty_text, (10, self.grid.n * self.cell_size + 60))

        pygame.display.flip()

    def remaining_cells_sum(self):
        total = 0
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i, j) not in self.grid.permanently_selected and not self.grid.is_forbidden(i, j):
                    total += self.grid.value[i][j]
        return total

    def handle_pair_selection(self):
        pair = tuple(self.selected_cells)
        if self.grid.valid_pair(*pair[0], *pair[1]):
            points = self.grid.cost(pair)
            self.scores[self.current_player] += points
            self.grid.permanently_selected.update(self.selected_cells)
            self.selected_cells = []
            self.current_player = 3 - self.current_player
        else:
            self.selected_cells = []

    def has_valid_pair(self):
        # Parcourt toutes les paires possibles non sélectionnées et non interdites
        for i in range(self.grid.n):
            for j in range(self.grid.m):
                if (i, j) in self.grid.permanently_selected or self.grid.is_forbidden(i, j):
                    continue
                for k in range(i, self.grid.n):
                    for l in range(self.grid.m):
                        if i == k and l <= j:
                            continue
                        if (k, l) in self.grid.permanently_selected or self.grid.is_forbidden(k, l):
                            continue
                        if self.grid.valid_pair(i, j, k, l):
                            return True
        return False

    def run(self):
        clock = pygame.time.Clock()
        final_message = None

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    x, y = event.pos
                    i, j = y // self.cell_size, x // self.cell_size
                    if (i, j) not in self.selected_cells and (i, j) not in self.grid.permanently_selected:
                        self.selected_cells.append((i, j))
                        if len(self.selected_cells) == 2:
                            self.handle_pair_selection()

            if not self.game_over and not self.has_valid_pair():
                penalty = self.remaining_cells_sum()
                bonus_player = self.current_player  # Le joueur qui n'a pas joué en dernier
                self.scores[bonus_player] += penalty
                final_message = f"{penalty} points ajoutés au joueur {bonus_player}"
                self.game_over = True
                self.draw_grid(final_message)
                # Boucle pour garder l'affichage final jusqu'à fermeture par l'utilisateur
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    clock.tick(30)

            self.draw_grid(final_message)
            clock.tick(30)
