from src.puzzles.rubiks_cube_puzzle import RubiksCube
from puzzle_solver import PuzzleSolver
from src.models.rubiks_cube.rubiks_MLP_model import RubiksCubeMLPModel


rubiks_cube_puzzle = RubiksCube()
rubiks_cube_MLP_model = RubiksCubeMLPModel()
solver = PuzzleSolver(rubiks_cube_puzzle, rubiks_cube_MLP_model)

solver.generate_states(max_depth=15, max_level_size=10000)
solver.train_model()
solver.evaluate_model()

c = rubiks_cube_puzzle.new_puzzle(total_movements=5)
#c = rubiks_cube_puzzle.from_string('GGGGGGGGG WWWWWWWWW OROOOROOO YYYYYYYYY RORORRRRR BBBBBBBBB')
print("Puzzle to solve:")
print(rubiks_cube_puzzle.to_string(c))

solution, visited_nodes, elapsed_time = solver.solve(c)

print("Heuristic Model Puzzle Solver:\n")
print(f"Solution found: {solution}")
print(f"Visited Nodes:  {visited_nodes}")
print(f"Elapsed Time:   {elapsed_time:.4f}s")


print("\n\nCompare with BFS:\n")

solution, visited_nodes, elapsed_time = solver.bfs_solve(c)

print(f"Solution found: {solution}")
print(f"Visited Nodes:  {visited_nodes}")
print(f"Elapsed Time:   {elapsed_time:.4f}s")