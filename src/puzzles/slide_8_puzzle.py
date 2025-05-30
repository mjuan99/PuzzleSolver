from puzzles.puzzle_interface import Puzzle
import random

class Slide8Puzzle(Puzzle):
  def __init__(self):
    self.movements = ["U", "D", "L", "R"]

  def new_puzzle(self, total_movements=0):
    puzzle = (
      0, 1, 2,
      3, 4, 5,
      6, 7, 8
    )

    if total_movements > 0:
      applied_movements = []
      for _ in range(total_movements):
        movement = random.choice(self.movements)
        while (not self.is_valid_movement(puzzle, movement)) or self.is_redundant(applied_movements, movement):
          movement = random.choice(self.movements)
        puzzle = self.apply_movement(puzzle, movement)
        applied_movements.append(movement)

    return puzzle

  def apply_movement(self, state, movement):
    empty_index = state.index(0)
    x = empty_index % 3
    y = empty_index // 3
    if movement == "U":
      y -= 1
    elif movement == "D":
      y += 1
    elif movement == "R":
      x += 1
    elif movement == "L":
      x -= 1
    swap_index = y * 3 + x

    state_list = list(state)
    state_list[empty_index], state_list[swap_index] = state_list[swap_index], state_list[empty_index]

    return tuple(state_list)

  def get_movements(self):
    random.shuffle(self.movements)
    return self.movements

  def is_valid_movement(self, state, movement):
    empty_index = state.index(0)
    x = empty_index % 3
    y = empty_index // 3
    if movement == "U":
      return y > 0
    elif movement == "D":
      return y < 2
    elif movement == "R":
      return x < 2
    elif movement == "L":
      return x > 0
    else:
      return False

  def is_redundant(self, prev_movements, movement):
    opposite_moves = {"U": "D", "D": "U", "L": "R", "R": "L"}
    return (len(prev_movements) > 0) and (movement == opposite_moves[prev_movements[-1]])

  def to_string(self, state):
    return f'\
    {state[0]:2} {state[1]:2} {state[2]:2}\n\
    {state[3]:2} {state[4]:2} {state[5]:2}\n\
    {state[6]:2} {state[7]:2} {state[8]:2}\n\
    '

  def from_string(self, string):
    return tuple(string.replace(' ', '').upper())