from puzzles.rubiks_cube_puzzle import RubiksCube
from puzzle_solver import PuzzleSolver
from models.rubiks_cube.rubiks_MLP_model import RubiksCubeMLPModel


# Setup Puzzle Solver to learn to solve a Rubik's 3x3x3 Cube using a MLP model as heuristic function
puzzle = RubiksCube()
model = RubiksCubeMLPModel()
solver = PuzzleSolver(puzzle, model)


# Generate training states, up to 15 moves away from the final state and up to 10.000 states for each level
solver.generate_states(max_depth=15, max_level_size=10000)
# Train the model on the generated states dataset
solver.train_model()
# Display some model evaluation if needed
solver.evaluate_model()


# Create a new unsolved Rubik's Cube (up to) 5 movements away from the final state
c = puzzle.new_puzzle(total_movements=5)
print("Puzzle to solve:")
print(puzzle.to_string(c))

# Solve it with the Puzzle Solver A* algorithm. Should be really quick for a state so close to the final state
solution, visited_nodes, elapsed_time = solver.solve(c)

print("Heuristic Model Puzzle Solver:\n")
print(f"Solution found: {solution}")
print(f"Visited Nodes:  {visited_nodes}")
print(f"Elapsed Time:   {elapsed_time:.4f}s")


# To compare, solve it with a BFS algorithm (brute force). Should take some time even though the state is close to the final state
print("\n\nCompare with BFS:\n")

solution, visited_nodes, elapsed_time = solver.bfs_solve(c)

print(f"Solution found: {solution}")
print(f"Visited Nodes:  {visited_nodes}")
print(f"Elapsed Time:   {elapsed_time:.4f}s")