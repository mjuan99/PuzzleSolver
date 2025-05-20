import heapq
import random
import datetime

class PuzzleSolver:
  def __init__(self, puzzle, model):
     self.puzzle = puzzle
     self.model = model

  def generate_states(self, max_depth=20, max_level_size=5000, visited_check=True, verbose=True):
    final_state = self.puzzle.new_puzzle()
    level_states = {0: [(final_state, [])]}  # Level 0 (the solved state)
    if visited_check:
      visited = set()  # Set to store visited states
      visited.add(final_state)

    # BFS: Expand the state space by applying all movements
    for level in range(1, max_depth + 1):
      if verbose:
        #level_start_time = datetime.now().strftime('%H:%M:%S')
        print(f"> Generating level {level}...")

      level_states[level] = []

      for state, movements_list in level_states[level - 1]:
        for movement in self.puzzle.get_movements():
          # Skip inverse/redundant movements
          if (not self.puzzle.is_valid_movement(state, movement)) or (self.puzzle.is_redundant(movements_list, movement)):
              continue

          # Apply the move
          new_state = self.puzzle.apply_movement(state, movement)

          # If the state hasn't been visited, add it
          if visited_check:
            if new_state not in visited:
              visited.add(new_state)
              level_states[level].append((new_state, movements_list + [movement]))
          else:
            level_states[level].append((new_state, movements_list + [movement]))

      # If level exceeds max level size, sample from the next level
      if len(level_states[level]) > max_level_size:
        level_states[level] = random.sample(level_states[level], max_level_size)

      if verbose:
        print(f"> Level size: {len(level_states[level])}")
        print()

    self.level_states_movements = level_states

    flattened_states = []
    for level, states_movements in level_states.items():
      for state, movements in states_movements:
        flattened_states.append((state, level))
    self.states = flattened_states


    return self.states

  def train_model(self):
    self.model.train_model(self.states)

  def evaluate_model(self):
    self.model.evaluate_model()

  def solve(self, start_state, max_depth=10):
    final_state = self.puzzle.new_puzzle()
    best_cost = {}

    start_node = Node(start_state, g=0, move_sequence=[])
    start_node.h = self.model.predict(start_state)
    start_node.f = start_node.g + start_node.h

    queue = [start_node]
    heapq.heapify(queue)

    while queue:
        node = heapq.heappop(queue)

        if node.state == final_state:  # goal check
            return node.move_sequence, len(best_cost)

        if (max_depth > 0 and node.g >= max_depth):
            continue

        if node.state in best_cost and best_cost[node.state] <= node.g:
            continue
        else:
            best_cost[node.state] = node.g

        for movement in self.puzzle.get_movements():
            new_state = self.puzzle.apply_movement(node.state, movement)

            child_node = Node(new_state, g=node.g + 1, move_sequence=node.move_sequence + [movement])
            child_node.h = self.model.predict(new_state)
            child_node.f = child_node.g + child_node.h
            heapq.heappush(queue, child_node)

    return None, len(best_cost)  # no solution found within depth limit



class Node:
    def __init__(self, state, g, move_sequence):
        self.state = state                      # cube state as sticker tuple
        self.g = g                              # cost so far
        self.h = 0                              # heuristic (to be computed)
        self.f = 0                              # total cost = g + h
        self.move_sequence = move_sequence      # list of movements to reach this state

    def __lt__(self, other):  # required for heapq to compare nodes
        return self.f < other.f