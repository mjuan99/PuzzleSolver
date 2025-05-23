from read_write_helper import read_from_file
from src.puzzles.rubiks_cube_puzzle import RubiksCube
from puzzle_solver import PuzzleSolver


def evaluate(solver, p, name):
  solution, visited_nodes, elapsed_time = solver.solve(p)
  print(f"{name:12} - Visited Nodes: {visited_nodes} - Elapsed Time: {elapsed_time:6.4f}s")

puzzle = RubiksCube()
model_paths = ['pkls/rubik_complex_MLP_15_50000.pkl', 'pkls/rubik_CNN.pkl']
model_names = ['MLP', 'CNN']
distances = [5, 7, 9]
iterations = 3

solvers = []
for path in model_paths:
  solvers.append(PuzzleSolver(puzzle, read_from_file(path)))

for distance in distances:
  for iteration in range(iterations):
    p = puzzle.new_puzzle(total_movements=distance)
    print(f"Distance {distance} - Iteration {iteration + 1}")
    for index in range(len(solvers)):
      evaluate(solvers[index], p, model_names[index])
    print()