from read_write_helper import write_to_file, read_from_file
from rubiks_cube_puzzle import RubiksCube
from puzzle_solver import PuzzleSolver
from rubiks_MLP_model import RubiksCubeMLPModel


read_states_from_file = False
write_states_to_file = True
states_file_path = 'level_states.pkl'
read_model_from_file = False
write_model_to_file = False
model_file_path = 'model.pkl'
evaluate_model = True


rubiks_cube_puzzle = RubiksCube()

if read_model_from_file:
  rubiks_cube_model = read_from_file(model_file_path)
  solver = PuzzleSolver(rubiks_cube_puzzle, rubiks_cube_model)
else:
  rubiks_cube_model = RubiksCubeMLPModel()
  solver = PuzzleSolver(rubiks_cube_puzzle, rubiks_cube_model)
  if read_states_from_file:
    states = read_from_file(states_file_path)
    solver.states = states
  else:
    solver.generate_states(max_level_size=20000)
    if write_states_to_file:
      write_to_file(states_file_path, solver.states)
  solver.train_model()
  if write_model_to_file:
    write_to_file(model_file_path, solver.model)

if evaluate_model:
  solver.evaluate_model()

c = rubiks_cube_puzzle.new_puzzle(total_movements=7)
print(rubiks_cube_puzzle.to_string(c))

print(solver.solve(c))
