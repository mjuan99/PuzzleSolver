import heapq
import random
import time
from collections import deque, defaultdict

class PuzzleSolver:
  def __init__(self, puzzle, model=None):
    self.puzzle = puzzle
    self.model = model

  def set_states(self, states):
    self.states = states

  def set_model(self, model):
    self.model = model

  # Returns a list of pairs (state, distance) generating a tree starting from the final state
  # and applying the available movements repeteadly to each state in each level.
  # max_level_size and max_depth control the growth of the states tree
  def generate_states(self, max_depth=10, max_level_size=5000, upsample_levels=True, balance_levels_to=0, visited_check=True, verbose=True):
    final_state = self.puzzle.new_puzzle()
    level_states = {0: [(final_state, [])]}  # Level 0 contains only the final state

    if visited_check:
      visited = set()  # Set to store visited states
      visited.add(final_state)

    # For each level generate all the possible states (not already generated) at one movement from the current level states
    level = 1
    while len(level_states[level - 1]) > 0 and not ((max_depth > 0) and ((level > max_depth))):
      if verbose:
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

      # If level exceeds max level size, sample from the next level (could be stopped before but this way adds randomness)
      if (max_level_size > 0) and (len(level_states[level]) > max_level_size):
        level_states[level] = random.sample(level_states[level], max_level_size)

      if verbose:
        print(f"> Level size: {len(level_states[level])}")
        print()
      
      level += 1

    self.level_states_movements = level_states

    flattened_states = []
    for level, states_movements in level_states.items():
      for state, movements in states_movements:
        flattened_states.append((state, level))
    self.states = flattened_states

    if upsample_levels:
      self.states = self.upsample_levels()
    else:
      if balance_levels_to > 0:
        self.states = self.balance_levels(balance_levels_to)

    return self.states
  
  
  def upsample_levels(self):
    level_to_states = defaultdict(list)
    for state, level in self.states:
      level_to_states[level].append((state, level))

    max_count = max(len(samples) for samples in level_to_states.values())

    balanced_states = []
    for level, samples in level_to_states.items():
      if len(samples) < max_count:
        needed = max_count - len(samples)
        samples_to_add = random.choices(samples, k=needed)
        samples.extend(samples_to_add)

      balanced_states.extend(samples)

    random.shuffle(balanced_states)
    self.states = balanced_states
    return self.states
  
  def balance_levels(self, level_size):
    level_to_states = defaultdict(list)
    for state, level in self.states:
      level_to_states[level].append((state, level))

    balanced_states = []
    for level, samples in level_to_states.items():
      if len(samples) < level_size:
        needed = level_size - len(samples)
        samples_to_add = random.choices(samples, k=needed)
        samples.extend(samples_to_add)
      else:
        samples = random.sample(samples, level_size)

      balanced_states.extend(samples)

    random.shuffle(balanced_states)
    self.states = balanced_states
    return self.states

  def train_model(self):
    self.model.train_model(self.states)

  def evaluate_model(self):
    self.model.evaluate_model()

  # Applies the A* algorithm to solve the puzzle, using the model as herustic function
  def solve(self, start_state, max_depth=0):
    start_time = time.time()

    final_state = self.puzzle.new_puzzle()
    best_cost = {}

    start_node = Node(start_state, g=0, move_sequence=[])
    start_node.h = self.model.predict(start_state)
    start_node.f = start_node.g + start_node.h

    queue = [start_node]
    heapq.heapify(queue)

    while queue:
      node = heapq.heappop(queue)

      # Goal check
      if node.state == final_state:
        return node.move_sequence, len(best_cost), time.time() - start_time

      # Depth limit check
      if (max_depth > 0 and node.g >= max_depth):
        continue

      # Check if state was already visisted with a better path
      if node.state in best_cost and best_cost[node.state] <= node.g:
        continue
      else:
        best_cost[node.state] = node.g

      # Generate neighbor nodes
      for movement in self.puzzle.get_movements():
        new_state = self.puzzle.apply_movement(node.state, movement)

        child_node = Node(new_state, g=node.g + 1, move_sequence=node.move_sequence + [movement])
        child_node.h = self.model.predict(new_state)
        child_node.f = child_node.g + child_node.h
        heapq.heappush(queue, child_node)

    return None, len(best_cost), time.time() - start_time  # no solution found within depth limit

  def bfs_solve(self, start_state, max_depth=0):
    start_time = time.time()
    final_state = self.puzzle.new_puzzle()
    visited = set()
    queue = deque()
    queue.append((start_state, []))
    visited.add(start_state)

    while queue:
        state, path = queue.popleft()

        if state == final_state:  # goal
            return path, len(visited), time.time() - start_time

        if (max_depth > 0) and (len(path) >= max_depth):
            continue

        for movement in self.puzzle.get_movements():
            if not self.puzzle.is_redundant(path, movement):
                new_state = self.puzzle.apply_movement(state, movement)

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [movement]))

    return None, len(visited), time.time() - start_time


class Node:
    def __init__(self, state, g, move_sequence):
        self.state = state
        self.g = g
        self.h = 0
        self.f = 0
        self.move_sequence = move_sequence

    # Required for heapq to compare nodes
    def __lt__(self, other):
        return self.f < other.f
    


