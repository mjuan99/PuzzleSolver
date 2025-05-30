> ⚠️ **Work in Progress**  
> This project is still under active development. Some features may be incomplete or non-functional, and the documentation may not reflect the most recent changes.

---

# Heuristic Model A* Puzzle Solver

A machine learning–augmented puzzle solver that learns how to solve any* puzzle by exploring the puzzle's state space knowing only the final state and the available movements.

---

## 🧩 Overview

This project presents an **experimental puzzle solver** that combines machine learning and A* search to solve puzzles with minimum knowledge of the puzzle mechanics.

The key idea is for the Puzzle Solver to learn by exploring any puzzle's state space, starting from the known final state and using only the available set of valid movements. Then, the puzzle knowledge gained by exploring the state space will be used alongside the A* search algorithm to solve new, unseen configurations.

---

## 🚦 Supported Puzzle Requirements

To be solvable by this system, a puzzle must satisfy the following constraints:

* ✅ A single, unique final (solved) state.
* ✅ A finite and known set of valid movements.
* ✅ Each movement must be **reversible**, i.e., for every move `m` there exists an inverse `m'` such that: `m'(m(s)) = s`

---

## ⚙️ How It Works
The Puzzle Solver architecture can be divided in two stages: the Learning Stage and the Solving Stage. The first step in the Learning Stage is the States Generation, where the Puzzle Solver takes as input the Puzzle definition (final stage and valid momvents) and some generation restrictions (max depth and max level size) and it starts exploring the state space starting from the final state and moving level by level with the available movements (level = distance to the final state), the output is a list of pairs (state, distance_to_final_state). Then, the next step in the Learning Stage is the Model Training, here a Machine Learning Model is trained to predict the distance of any unsolved state to the final state. Finally, in the Solving Stage, the trained Model is used as Heuristic in the A* alogrithm to efficiently find solutions to any unsolved state.

Throughout the documentation, the Rubik's Cube will be used as example, but all steps can be adapted to different puzzles. Rubik's Cube images generated with https://rubiks-cube-solver.com/.

![Puzzle Solver Diagram](images/puzzle_solver_diagram.png)


### 1. State Generation
Starting from the final (solved) state, the system generates a subset of all the valid states by applying all valid moves to each known state, keeping track of how many steps away each state is from the goal. The output will be a list of pairs (state, distance). Along with the puzzle definition (containing the final state and valid movements) some generation restrictions are used as input to restrict the state space exploration, which is indispensable when learning to solve puzzles that have a very large amount of possible states, like the Rubik's Cube. These restrictions are max_depth and max_level_size. max_depth defines how far from the final state should the Puzzle Solver explore and max_level_size defines how many states should be explored for each level.

The next image ilustrates the State Generation process for a generic puzzle with a given final_state, \[m_0, m_1\] as the available movements, max_depth = 4 and max_level_size = 6.

![Generic States Generation Diagram](images/generic_states_generation.png)

Notice that in level 3 (distance = 3) the two leftmost states are excluded, as max_level_size indicates that only 6 states per level should be included (not necessarily the first 6) and there are 8 states in level 3. Then, once reached level 4 and selected up to 6 states for that level, the State Generation process stop.

The next image pictures what the State Generation output would look like applied to the Rubik's Cube.

![Rubik's Cube States Generation Diagram](images/rubiks_states_generation.png)

It's important to note that if the States Generation process doesn't explore every state in the state space it's possible that a state skipped in a level near the final state is found later and considered to be further away than it actually is, producing inaccurate data for the Model Training step and potentially degrading the performance of the Puzzle Solver. For that reason, the Puzzle interface (that any puzzle should implement) defines an abstract function `is_redundant(prev_movements, movement)` that allows the Puzzle Solver to avoid movements that would reduce the distance to the final state instead of increasing it. A correct and robust implementation of this function can be challenging and require strong knowledge of the puzzle, but nevertheless a simple implementation can also help get a cleaner and more accurate states dataset.

### 2. Model Training
In this step a machine learning model is trained, using the output from the previous step, to predict the number of movements from any state to the final state. The model selection and configuration along with the evaluation is crucial to produce an effective and efficient Puzzle Solver. During testing XGBoost, MLP and CNN models were mainly used to predict distances from any state to the final state. Also some dummy models were implemented for quick testing and comparison, including BFSModel (always returns 0, making the A* algorithm equivalent to BFS search), RandomModel (returns a random value) and RandomNormalModel (returns a random normally distributed value).

Note that if the model perfectly predicts the distance from any state to the final state, the A* algorithm will be able to find the optimal solution very quickly, without visiting any state that isn't in the optimal path to the solved state. Nevertheless a good enough model with some wrong predictions can also be used to efficiently find soultions, although the solution is not guaranteed to be optimal.

![Heuristic Predictions Diagram](images/heuristic_predictions.png)

### 3. A* Search:
When solving a new puzzle configuration, the trained model is used as the heuristic function (`h(n)`) for A*, guiding the search efficiently.

A* is a graph traversal and pathfinding algorithm that finds the path from a start state to a goal by combining the actual cost to reach a node and a heuristic estimate of the remaining cost. At each step, it selects the state with the lowest estimated total cost.

The next image shows a simplified diagram of how the A* algorithm would work solving a Rubik's Cube with a perfect heuristic function (it always return the exact amount of moves needed to solve any state). Starting from the initial state, it has cost 0 (it didn't need any movement to get to that state), heuristic 4 (the heuristic function predicts that state can be solved in 4 movements) and total 4 (adding the cost and the heuristic to estimate how many movements would be needed to solve the puzzle). Then all the neighbors are generated, applying every possible movement to the current state, and calculating their cost, heuristic and total (in the image we are only seeing 3 neighbors for each visited state but in this case there should be 18). Then the state with the smaller total (without considering already visited oned) is selected and the process repeats, generating all the neighbors and calculating their values. In this case the next selected state has cost 1 (it's 1 movement away from the initial state), heuristic 3 (can be solved in 3 movements) and total 4 (solving the cube passing through that state is predicted to take 4 movements). The process is repeated until finding the goal (final state) and the solution is the path (movements) from the initial state to the goal. Notice that as the heuristic function is perfect the algorithm always choose a state in the optimal path to the goal (states where total = 4) because any other path would increment the total (increasing the cost to get to that state without decreasing the heuristic) and therefore won't be chosen by the algorithm. Also notice that every displayed state outside the path to the goal has a total of 6, that makes sense because they are all one wrong movement away from the optimal path, so their total is 4 (best solution) + 1 (wrong movement) + 1 (undoing the wrong movement).

![A* algorithm solving Rubik's cube](images/rubik_a_star.png)

---

## 🚀 Puzzle Solver in action
Now let's see the Puzzle Solver working!
First we create the Puzzle Solver with the Puzzle it will learn to solve (Rubik's Cube) and the Model it will use to learn (MLP Model). Implementation of these classes can be found on the `src` directory.
```
puzzle = RubiksCube()
model = RubiksCubeMLPModel()
solver = PuzzleSolver(puzzle, model)
```
Then we use the Puzzle Solver to generate a list of states, in this case up to 15 movements from the final state and up to 10.000 states per level.
```
solver.generate_states(max_depth=15, max_level_size=10000)
```
Once the states are generated, the Puzzle Solver will train its model (the MLP model defined previously) to predict distances to the final state.
```
solver.train_model()
```
Then we can evaluate the model, the `evaluate_model()` implementation is up to the user, in this case we are computing MAE, RMSE and a scatterplot of true vs predicted distances.
```
solver.evaluate_model()
```
```
# OUTPUT:
Test MAE: 1.1794
Test RMSE: 1.7206
```
![True vs Predicted Distances scatterplot](images/evaluation_scatterplot.png)

The error metrics and the scatterplot shows that even though the model predictions are far from perfect they might be good enough for the A\* algorithm

Then we can get a new unsolved state (in this case 5 movements away from the final state) and print it in a 2D representation.
```
c = puzzle.new_puzzle(total_movements=5)
print("Puzzle to solve:")
print(puzzle.to_string(c))
```
```
#  OUTPUT:
Puzzle to solve:
                W Y Y
                W G Y
                R G Y
    R R B O O G Y O O B B B
    R W B O O W G Y W R R B
    R W O G G W G Y W R R B
                O O G
                Y B G
                Y B W
```
Finally we can tell the Puzzle Solver to solve it.
```
solution, visited_nodes, elapsed_time = solver.solve(c)
print("Heuristic Model Puzzle Solver:\n")
print(f"Solution found: {solution}")
print(f"Visited Nodes:  {visited_nodes}")
print(f"Elapsed Time:   {elapsed_time:.4f}s")
```
```
# OUTPUT:
Heuristic Model Puzzle Solver:

Solution found: ["L'", "R'", 'B', 'U', "R'"]
Visited Nodes:  6
Elapsed Time:   0.0779s
```
Also, for comparison we can solve the same state using BFS. BFS is a brute force search algorithm that checks all the states at distance 1 from the initial state, then all the states at distance 2, and so on...that way it always finds an optimal solution (shortest path) but it may take a lot of time depending on the size of the state space and the distance to the final state.
```
solution, visited_nodes, elapsed_time = solver.bfs_solve(c)
print("BFS Solver:\n")
print(f"Solution found: {solution}")
print(f"Visited Nodes:  {visited_nodes}")
print(f"Elapsed Time:   {elapsed_time:.4f}s")
```
```
# OUTPUT:
BFS Solver:

Solution found: ["R'", "L'", 'B', 'U', "R'"]
Visited Nodes:  1936500
Elapsed Time:   11.4638s
```
We can see that both algorithms found a solution with 5 movements, so both are optimal solutions in this case. But A\* visited 6 nodes in 0.0779 seconds to find the solution while BFS visited 1936500 in 11.4638 seconds, so in this case the Puzzle Solver solution proved to be more efficient. If we use unsolved states more complex the BFS solution quickly becomes unacceptably slow and the Puzzle Solver A\* algorithm works fine up to around 10 movements, further than that it becomes highly unstable.
```
c = puzzle.new_puzzle(total_movements=10)
solution, visited_nodes, elapsed_time = solver.solve(c)
print(f"Solution found: {solution}")
print(f"Visited Nodes:  {visited_nodes}")
print(f"Elapsed Time:   {elapsed_time:.4f}s")
```
```
# OUTPUT:
Solution found: ["F'", 'R2', "L'", 'D', 'F2', 'R', 'F2', 'R2', "B'", 'U2']
Visited Nodes:  105
Elapsed Time:   0.3625s
```
---

## 🗂️ Project Structure

```
project-root/
├── src/
│   ├── models/                  # Model classes: MLP, XGBoost, CNN, etc.
│   │   └── model_interface.py   # Interface for all heuristic models
│   ├── puzzles/                 # Puzzle definitions (e.g., Rubik's Cube)
│   │   └── puzzle_interface.py  # Interface for all puzzles
│   ├── puzzle_solver.py         # Core state generation, training and A* solver logic
│   └── tests.py                 # Test scripts for training and solving
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ▶️ How to Run

1. **Clone the repo**

   ```bash
   git clone https://github.com/mjuan99/PuzzleSolver.git
   cd PuzzleSolver
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run test script**

   ```bash
   python src/tests.py
   ```

---

## 🛠️ Main Technologies Used

* Python
* PyTorch
* Scikit-learn
* NumPy

---

## 🚧 Limitations

While the Puzzle Solver is designed to be general-purpose and adaptable to a wide range of puzzles, its performance heavily depends on the quality of the trained heuristic model. The effectiveness and efficiency of the solver are directly influenced by the choice of model architecture, the quality and representativeness of the training data, and the tuning of model parameters. As such, selecting and training the right model is the most critical factor in achieving strong performance when solving a particular puzzle. 

In addition, the state generation process—used to create the dataset for training the heuristic model—relies on backward exploration from the solved state. Without strong domain knowledge of the puzzle (such as understanding the true maximum distance from the solved state or identifying non-obvious "backward" moves that reduce that distance), the generation process may introduce inaccuracies. These inaccuracies could lead to noisy or misleading training data, which may negatively impact the learned heuristic and ultimately degrade the solver’s performance—sometimes negligibly, but potentially significantly depending on the puzzle.

Finally, the solution found by the Puzzle Solver may not be optimal (i.e., it may involve more moves than the minimum required). This happens because the learned heuristic function is not guaranteed to be **admissible** — it may overestimate the true cost to the goal. As a result, the A* search might not always find the shortest possible path.

---

## 📄 License

This project is open-source and available under the MIT License.

---
