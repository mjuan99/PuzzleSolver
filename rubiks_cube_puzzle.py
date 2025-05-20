from puzzle_interface import Puzzle
import random

class RubiksCube(Puzzle):
  def __init__(self):
    self.movements = [
      "U", "U'", "U2",
      "D", "D'", "D2",
      "F", "F'", "F2",
      "B", "B'", "B2",
      "L", "L'", "L2",
      "R", "R'", "R2"
    ]

  def new_puzzle(self, total_movements=0):
    puzzle = (
      'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y',
      'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R',
      'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
      'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W',
      'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
      'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G',
    )

    if total_movements > 0:
      applied_movements = []
      for _ in range(total_movements):
        movement = random.choice(self.movements)
        while self.is_redundant(applied_movements, movement):
          movement = random.choice(self.movements)
        puzzle = self.apply_movement(puzzle, movement)
        applied_movements.append(movement)

    return puzzle

  def apply_movement(self, state, movement=''):
    t = state
    if movement == 'R':
      return (
          t[20], t[1], t[2], t[3], t[4], t[5], t[18], t[19], t[8],
          t[15], t[16], t[9], t[10], t[11], t[12], t[13], t[14], t[17],
          t[31], t[32], t[33], t[21], t[22], t[23], t[24], t[25], t[26],
          t[27], t[28], t[29], t[30], t[47], t[48], t[49], t[34], t[35],
          t[36], t[37], t[38], t[39], t[40], t[41], t[42], t[43], t[44],
          t[45], t[46], t[6], t[7], t[0], t[50], t[51], t[52], t[53],
      )
    elif movement == 'R2':
      return (
          t[33], t[1], t[2], t[3], t[4], t[5], t[31], t[32], t[8],
          t[13], t[14], t[15], t[16], t[9], t[10], t[11], t[12], t[17],
          t[47], t[48], t[49], t[21], t[22], t[23], t[24], t[25], t[26],
          t[27], t[28], t[29], t[30], t[6], t[7], t[0], t[34], t[35],
          t[36], t[37], t[38], t[39], t[40], t[41], t[42], t[43], t[44],
          t[45], t[46], t[18], t[19], t[20], t[50], t[51], t[52], t[53],
      )
    elif movement == 'R\'':
      return (
          t[49], t[1], t[2], t[3], t[4], t[5], t[47], t[48], t[8],
          t[11], t[12], t[13], t[14], t[15], t[16], t[9], t[10], t[17],
          t[6], t[7], t[0], t[21], t[22], t[23], t[24], t[25], t[26],
          t[27], t[28], t[29], t[30], t[18], t[19], t[20], t[34], t[35],
          t[36], t[37], t[38], t[39], t[40], t[41], t[42], t[43], t[44],
          t[45], t[46], t[31], t[32], t[33], t[50], t[51], t[52], t[53],
      )
    elif movement == 'L':
      return (
          t[0], t[1], t[51], t[52], t[45], t[5], t[6], t[7], t[8],
          t[9], t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17],
          t[18], t[19], t[20], t[21], t[2], t[3], t[4], t[25], t[26],
          t[22], t[23], t[24], t[30], t[31], t[32], t[33], t[34], t[35],
          t[42], t[43], t[36], t[37], t[38], t[39], t[40], t[41], t[44],
          t[29], t[46], t[47], t[48], t[49], t[50], t[27], t[28], t[53],
      )
    elif movement == 'L2':
      return (
          t[0], t[1], t[27], t[28], t[29], t[5], t[6], t[7], t[8],
          t[9], t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17],
          t[18], t[19], t[20], t[21], t[51], t[52], t[45], t[25], t[26],
          t[2], t[3], t[4], t[30], t[31], t[32], t[33], t[34], t[35],
          t[40], t[41], t[42], t[43], t[36], t[37], t[38], t[39], t[44],
          t[24], t[46], t[47], t[48], t[49], t[50], t[22], t[23], t[53],
      )
    elif movement == 'L\'':
      return (
          t[0], t[1], t[22], t[23], t[24], t[5], t[6], t[7], t[8],
          t[9], t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17],
          t[18], t[19], t[20], t[21], t[27], t[28], t[29], t[25], t[26],
          t[51], t[52], t[45], t[30], t[31], t[32], t[33], t[34], t[35],
          t[38], t[39], t[40], t[41], t[42], t[43], t[36], t[37], t[44],
          t[4], t[46], t[47], t[48], t[49], t[50], t[2], t[3], t[53],
      )
    elif movement == 'F':
      return (
          t[40], t[41], t[42], t[3], t[4], t[5], t[6], t[7], t[8],
          t[2], t[10], t[11], t[12], t[13], t[14], t[0], t[1], t[17],
          t[24], t[25], t[18], t[19], t[20], t[21], t[22], t[23], t[26],
          t[27], t[28], t[15], t[16], t[9], t[32], t[33], t[34], t[35],
          t[36], t[37], t[38], t[39], t[29], t[30], t[31], t[43], t[44],
          t[45], t[46], t[47], t[48], t[49], t[50], t[51], t[52], t[53],
      )
    elif movement == 'F2':
      return (
          t[29], t[30], t[31], t[3], t[4], t[5], t[6], t[7], t[8],
          t[42], t[10], t[11], t[12], t[13], t[14], t[40], t[41], t[17],
          t[22], t[23], t[24], t[25], t[18], t[19], t[20], t[21], t[26],
          t[27], t[28], t[0], t[1], t[2], t[32], t[33], t[34], t[35],
          t[36], t[37], t[38], t[39], t[15], t[16], t[9], t[43], t[44],
          t[45], t[46], t[47], t[48], t[49], t[50], t[51], t[52], t[53],
      )
    elif movement == 'F\'':
      return (
          t[15], t[16], t[9], t[3], t[4], t[5], t[6], t[7], t[8],
          t[31], t[10], t[11], t[12], t[13], t[14], t[29], t[30], t[17],
          t[20], t[21], t[22], t[23], t[24], t[25], t[18], t[19], t[26],
          t[27], t[28], t[40], t[41], t[42], t[32], t[33], t[34], t[35],
          t[36], t[37], t[38], t[39], t[0], t[1], t[2], t[43], t[44],
          t[45], t[46], t[47], t[48], t[49], t[50], t[51], t[52], t[53],
      )
    elif movement == 'B':
      return (
          t[0], t[1], t[2], t[3], t[11], t[12], t[13], t[7], t[8],
          t[9], t[10], t[33], t[34], t[27], t[14], t[15], t[16], t[17],
          t[18], t[19], t[20], t[21], t[22], t[23], t[24], t[25], t[26],
          t[38], t[28], t[29], t[30], t[31], t[32], t[36], t[37], t[35],
          t[4], t[5], t[6], t[39], t[40], t[41], t[42], t[43], t[44],
          t[51], t[52], t[45], t[46], t[47], t[48], t[49], t[50], t[53],
      )
    elif movement == 'B2':
      return (
          t[0], t[1], t[2], t[3], t[33], t[34], t[27], t[7], t[8],
          t[9], t[10], t[36], t[37], t[38], t[14], t[15], t[16], t[17],
          t[18], t[19], t[20], t[21], t[22], t[23], t[24], t[25], t[26],
          t[6], t[28], t[29], t[30], t[31], t[32], t[4], t[5], t[35],
          t[11], t[12], t[13], t[39], t[40], t[41], t[42], t[43], t[44],
          t[49], t[50], t[51], t[52], t[45], t[46], t[47], t[48], t[53],
      )
    elif movement == 'B\'':
      return (
          t[0], t[1], t[2], t[3], t[36], t[37], t[38], t[7], t[8],
          t[9], t[10], t[4], t[5], t[6], t[14], t[15], t[16], t[17],
          t[18], t[19], t[20], t[21], t[22], t[23], t[24], t[25], t[26],
          t[13], t[28], t[29], t[30], t[31], t[32], t[11], t[12], t[35],
          t[33], t[34], t[27], t[39], t[40], t[41], t[42], t[43], t[44],
          t[47], t[48], t[49], t[50], t[51], t[52], t[45], t[46], t[53],
      )
    elif movement == 'U':
      return (
          t[6], t[7], t[0], t[1], t[2], t[3], t[4], t[5], t[8],
          t[49], t[50], t[51], t[12], t[13], t[14], t[15], t[16], t[17],
          t[11], t[19], t[20], t[21], t[22], t[23], t[9], t[10], t[26],
          t[27], t[28], t[29], t[30], t[31], t[32], t[33], t[34], t[35],
          t[36], t[37], t[24], t[25], t[18], t[41], t[42], t[43], t[44],
          t[45], t[46], t[47], t[48], t[38], t[39], t[40], t[52], t[53],
      )
    elif movement == 'U2':
      return (
          t[4], t[5], t[6], t[7], t[0], t[1], t[2], t[3], t[8],
          t[38], t[39], t[40], t[12], t[13], t[14], t[15], t[16], t[17],
          t[51], t[19], t[20], t[21], t[22], t[23], t[49], t[50], t[26],
          t[27], t[28], t[29], t[30], t[31], t[32], t[33], t[34], t[35],
          t[36], t[37], t[9], t[10], t[11], t[41], t[42], t[43], t[44],
          t[45], t[46], t[47], t[48], t[24], t[25], t[18], t[52], t[53],
      )
    elif movement == 'U\'':
      return (
          t[2], t[3], t[4], t[5], t[6], t[7], t[0], t[1], t[8],
          t[24], t[25], t[18], t[12], t[13], t[14], t[15], t[16], t[17],
          t[40], t[19], t[20], t[21], t[22], t[23], t[38], t[39], t[26],
          t[27], t[28], t[29], t[30], t[31], t[32], t[33], t[34], t[35],
          t[36], t[37], t[49], t[50], t[51], t[41], t[42], t[43], t[44],
          t[45], t[46], t[47], t[48], t[9], t[10], t[11], t[52], t[53],
      )
    elif movement == 'D':
      return (
          t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8],
          t[9], t[10], t[11], t[12], t[20], t[21], t[22], t[16], t[17],
          t[18], t[19], t[42], t[43], t[36], t[23], t[24], t[25], t[26],
          t[33], t[34], t[27], t[28], t[29], t[30], t[31], t[32], t[35],
          t[47], t[37], t[38], t[39], t[40], t[41], t[45], t[46], t[44],
          t[13], t[14], t[15], t[48], t[49], t[50], t[51], t[52], t[53],
      )
    elif movement == 'D2':
      return (
          t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8],
          t[9], t[10], t[11], t[12], t[42], t[43], t[36], t[16], t[17],
          t[18], t[19], t[45], t[46], t[47], t[23], t[24], t[25], t[26],
          t[31], t[32], t[33], t[34], t[27], t[28], t[29], t[30], t[35],
          t[15], t[37], t[38], t[39], t[40], t[41], t[13], t[14], t[44],
          t[20], t[21], t[22], t[48], t[49], t[50], t[51], t[52], t[53],
      )
    elif movement == 'D\'':
      return (
          t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8],
          t[9], t[10], t[11], t[12], t[45], t[46], t[47], t[16], t[17],
          t[18], t[19], t[13], t[14], t[15], t[23], t[24], t[25], t[26],
          t[29], t[30], t[31], t[32], t[33], t[34], t[27], t[28], t[35],
          t[22], t[37], t[38], t[39], t[40], t[41], t[20], t[21], t[44],
          t[42], t[43], t[36], t[48], t[49], t[50], t[51], t[52], t[53],
      )
    else:
      return t

  def apply_movements(self, state, movements):
    for movement in movements:
      state = self.apply_movement(state, movement)
    return state

  def get_movements(self):
    random.shuffle(self.movements)
    return self.movements

  def is_valid_movement(self, state, movement):
    return True

  def is_redundant(self, movements, movement):
    if len(movements) == 0:
      return False
    elif len(movements) == 1:
      return movements[0][0] == movement[0] # if last move was R' then R, R' and R2 are redundant
    else:
      return (movements[-1][0] == movement[0]) or ((RubiksCube.get_opposite_face(movements[-1]) == movement[0]) and (movements[-2][0] == movement[0]))

  @staticmethod
  def get_reverse_movement(movement):
    d = {
        'R': 'R\'', 'R\'': 'R', 'R2': 'R2',
        'L': 'L\'', 'L\'': 'L', 'L2': 'L2',
        'U': 'U\'', 'U\'': 'U', 'U2': 'U2',
        'D': 'D\'', 'D\'': 'D', 'D2': 'D2',
        'F': 'F\'', 'F\'': 'F', 'F2': 'F2',
        'B': 'B\'', 'B\'': 'B', 'B2': 'B2',
    }
    return d[movement]

  @staticmethod
  def get_reverse_movements(movements):
    reverse_movements = []
    for movement in movements[::-1]:
      reverse_movements.append(RubiksCube.get_reverse_movement(movement))
    return reverse_movements

  @staticmethod
  def get_opposite_face(movement):
    d = {
        'R': 'L', 'L': 'R',
        'U': 'D', 'D': 'U',
        'F': 'B', 'B': 'F',
    }
    return d[movement[0]]

  def to_string(self, state):
    t = state
    return f'\
                {t[45]} {t[46]} {t[47]}\n\
                {t[52]} {t[53]} {t[48]}\n\
                {t[51]} {t[50]} {t[49]}\n\
    {t[33]} {t[34]} {t[27]} {t[36]} {t[37]} {t[38]} {t[4]} {t[5]} {t[6]} {t[11]} {t[12]} {t[13]}\n\
    {t[32]} {t[35]} {t[28]} {t[43]} {t[44]} {t[39]} {t[3]} {t[8]} {t[7]} {t[10]} {t[17]} {t[14]}\n\
    {t[31]} {t[30]} {t[29]} {t[42]} {t[41]} {t[40]} {t[2]} {t[1]} {t[0]} {t[9]} {t[16]} {t[15]}\n\
                {t[24]} {t[25]} {t[18]}\n\
                {t[23]} {t[26]} {t[19]}\n\
                {t[22]} {t[21]} {t[20]}\n\
                '

  def from_string(self, string):
    t = string.replace(' ', '').upper()
    return (
          t[35], t[34], t[33], t[30], t[27], t[28], t[29], t[32], t[31],
          t[42], t[39], t[36], t[37], t[38], t[41], t[44], t[43], t[40],
          t[47], t[50], t[53], t[52], t[51], t[48], t[45], t[46], t[49],
          t[11], t[14], t[17], t[16], t[15], t[12], t[9], t[10], t[13],
          t[18], t[19], t[20], t[23], t[26], t[25], t[24], t[21], t[22],
          t[0], t[1], t[2], t[5], t[8], t[7], t[6], t[3], t[4],
      )