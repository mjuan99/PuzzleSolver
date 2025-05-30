from puzzles.puzzle_interface import Puzzle
import random

# Rubiks Cube 2x2x2 implementation
# states are represented as a tuple of 24 'stickers' (the different colors each piece can have coded as Y, B, R, W, G and O for Yellow, Blue, Red, White, Green and Orange)
#
# The solved state can be represented in 2D as
#         G G
#         G G
# W W O O Y Y R R
# W W O O Y Y R R
#         B B
#         B B
#
# The indexes for each sticker in the tuple are:
#               20 21 
#               23 22    
#  15 12 16 17   2  3  5  6 
#  14 13 19 18   1  0  4  7 
#               11  8
#               10  9
#                     
class Rubiks222Cube(Puzzle):
  def __init__(self):
    self.movements = [
      "U", "U'", "U2",
      "F", "F'", "F2",
      "R", "R'", "R2"
    ]

  def new_puzzle(self, total_movements=0):
    puzzle = (
      'Y', 'Y', 'Y', 'Y',
      'R', 'R', 'R', 'R',
      'B', 'B', 'B', 'B',
      'W', 'W', 'W', 'W',
      'O', 'O', 'O', 'O',
      'G', 'G', 'G', 'G',
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

  def from_string(self, string):
    t = string.replace(' ', '').upper()
    return (
      t[15], t[14], t[12], t[13],
      t[18], t[16], t[17], t[19],
      t[21], t[23], t[22], t[20],
      t[5], t[7], t[6], t[4],
      t[8], t[9], t[11], t[10],
      t[0], t[1], t[3], t[2]
    )

  def to_string(self, state):
    t = state
    return f'\
            {t[20]} {t[21]}\n\
            {t[23]} {t[22]}\n\
    {t[15]} {t[12]} {t[16]} {t[17]} {t[2]} {t[3]} {t[5]} {t[6]}\n\
    {t[14]} {t[13]} {t[19]} {t[18]} {t[1]} {t[0]} {t[4]} {t[7]}\n\
            {t[11]} {t[8]}\n\
            {t[10]} {t[9]}\n\
                '

  def apply_movements(self, state, movements):
    for movement in movements:
      state = self.apply_movement(state, movement)
    return state

  def get_movements(self):
    random.shuffle(self.movements)
    return self.movements

  def is_valid_movement(self, state, movement):
    # Every movement is always valid to every state
    return True

  def is_redundant(self, movements, movement):
    if len(movements) == 0:
      return False
    else:
      return movements[-1][0] == movement[0]

  @staticmethod
  def get_reverse_movement(movement):
    d = {
        'R': 'R\'', 'R\'': 'R', 'R2': 'R2',
        'U': 'U\'', 'U\'': 'U', 'U2': 'U2',
        'F': 'F\'', 'F\'': 'F', 'F2': 'F2',
    }
    return d[movement]

  @staticmethod
  def get_reverse_movements(movements):
    reverse_movements = []
    for movement in movements[::-1]:
      reverse_movements.append(Rubiks222Cube.get_reverse_movement(movement))
    return reverse_movements

  def apply_movement(self, state, movement=''):
    t = state
    if movement == 'R':
      return (
          t[9], t[1], t[2], t[8],
          t[7], t[4], t[5], t[6],
          t[14], t[15], t[10], t[11],
          t[12], t[13], t[21], t[22],
          t[16], t[17], t[18], t[19],
          t[20], t[3], t[0], t[23],
      )
    elif movement == 'R2':
      return (
          t[15], t[1], t[2], t[14],
          t[6], t[7], t[4], t[5],
          t[21], t[22], t[10], t[11],
          t[12], t[13], t[3], t[0],
          t[16], t[17], t[18], t[19],
          t[20], t[8], t[9], t[23],
      )
    elif movement == 'R\'':
      return (
          t[22], t[1], t[2], t[21],
          t[5], t[6], t[7], t[4],
          t[3], t[0], t[10], t[11],
          t[12], t[13], t[8], t[9],
          t[16], t[17], t[18], t[19],
          t[20], t[14], t[15], t[23],
      )
    elif movement == 'F':
      return (
          t[18], t[19], t[2], t[3],
          t[1], t[5], t[6], t[0],
          t[11], t[8], t[9], t[10],
          t[12], t[7], t[4], t[15],
          t[16], t[17], t[13], t[14],
          t[20], t[21], t[22], t[23],
      )
    elif movement == 'F2':
      return (
          t[13], t[14], t[2], t[3],
          t[19], t[5], t[6], t[18],
          t[10], t[11], t[8], t[9],
          t[12], t[0], t[1], t[15],
          t[16], t[17], t[7], t[4],
          t[20], t[21], t[22], t[23],
      )
    elif movement == 'F\'':
      return (
          t[7], t[4], t[2], t[3],
          t[14], t[5], t[6], t[13],
          t[9], t[10], t[11], t[8],
          t[12], t[18], t[19], t[15],
          t[16], t[17], t[0], t[1],
          t[20], t[21], t[22], t[23],
      )
    elif movement == 'U':
      return (
          t[3], t[0], t[1], t[2],
          t[22], t[23], t[6], t[7],
          t[5], t[9], t[10], t[4],
          t[12], t[13], t[14], t[15],
          t[16], t[11], t[8], t[19],
          t[20], t[21], t[17], t[18],
      )
    elif movement == 'U2':
      return (
          t[2], t[3], t[0], t[1],
          t[17], t[18], t[6], t[7],
          t[23], t[9], t[10], t[22],
          t[12], t[13], t[14], t[15],
          t[16], t[4], t[5], t[19],
          t[20], t[21], t[11], t[8],
      )
    elif movement == 'U\'':
      return (
          t[1], t[2], t[3], t[0],
          t[11], t[8], t[6], t[7],
          t[18], t[9], t[10], t[17],
          t[12], t[13], t[14], t[15],
          t[16], t[22], t[23], t[19],
          t[20], t[21], t[4], t[5],
      )
    else:
      return t