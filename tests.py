from read_write_helper import write_to_file, read_from_file
from src.puzzles.rubiks_cube_puzzle import RubiksCube
from puzzle_solver import PuzzleSolver
from src.models.rubiks_cube.rubiks_MLP_model import RubiksCubeMLPModel
from src.models.rubiks_cube.rubiks_basic_MLP_model import RubiksCubeBasicMLPModel
from src.models.rubiks_cube.rubiks_XGB_model import RubiksCubeXGBModel
from src.models.rubiks_cube.rubiks_CNN_model import RubiksCubeCNNModel



def generate_states_and_save(max_depth, max_level_size, puzzle, states_file_path):
  solver = PuzzleSolver(puzzle)
  states = solver.generate_states(max_depth=max_depth, max_level_size=max_level_size)
  write_to_file(states_file_path, states)

def train_model_and_save(states, model, model_file_path):
  solver = PuzzleSolver(None, model)
  solver.states = states
  solver.train_model()
  write_to_file(model_file_path, solver.model)
  solver.evaluate_model()

def evaluate(solver, c, name):
  solution, visited_nodes, elapsed_time = solver.solve(c)
  print(f"{name:12} - Visited Nodes: {visited_nodes} - Elapsed Time: {elapsed_time:6.4f}s")


rubiks_cube_puzzle = RubiksCube()
cnn_model = RubiksCubeCNNModel()
solver = PuzzleSolver(rubiks_cube_puzzle, cnn_model)

solver.states = read_from_file('pkls/rubik_states_15_50000.pkl')

solver.train_model()
solver.evaluate_model()

write_to_file('pkls/rubik_CNN.pkl', solver.model)

solver2 = PuzzleSolver(rubiks_cube_puzzle, read_from_file('pkls/rubik_complex_MLP_15_50000.pkl'))
c = rubiks_cube_puzzle.new_puzzle(total_movements=8)

evaluate(solver, c, "CNN")
evaluate(solver2, c, "MLP")