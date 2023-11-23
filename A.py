import numpy as np

grid = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 3, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])

# Getting the indices of cells where food is present (value > 0)
food_positions = np.argwhere(grid > 0)

# Getting the level of food at each position
food_levels = [grid[position[0], position[1]] for position in food_positions]

# Converting the positions to tuples
food_positions = [tuple(position) for position in food_positions]

# Displaying the positions and levels of the food
for position, level in zip(food_positions, food_levels):
    print(f"Food at position {position} with level {level}")
