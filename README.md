> ⚠️ **Work in Progress**  
> This project is still under active development. Some features may be incomplete or non-functional, and the documentation may not reflect the most recent changes.

---

# Heuristic Model A* Puzzle Solver

A machine learning–augmented puzzle solver that learns how to solve any* puzzle by training a heuristic function from the final state alone.

---

## 🧩 Overview

This project presents an **experimental puzzle solver** that combines machine learning and A* search to solve puzzles.

The key idea is to automatically generate a dataset of puzzle states and their respective distance to the final (solved) state. Then train a model that can estimate the distance from any puzzle state to the final state. Finally, This model is used as a heuristic in the A* algorithm, allowing it to efficiently find solutions to new, unseen configurations.

---

## 🚦 Supported Puzzle Requirements

To be solvable by this system, a puzzle must satisfy the following constraints:

* ✅ A single, unique final (solved) state.
* ✅ A finite and known set of valid movements.
* ✅ Each movement must be **reversible**, i.e., for every move `m` there exists an inverse `m'` such that: `m'(m(s)) = s`

---

## ⚙️ How It Works

1. **State Generation:**
   Starting from the solved state, the system generates a subset of all the valid states by applying random valid moves, keeping track of how many steps away each state is from the goal.

2. **Model Training:**
   A machine learning model is trained to predict the number of steps from a given state to the goal using the generated data.

3. **A\* Search:**
   When solving a new puzzle configuration, the trained model is used as the heuristic function (`h(n)`) for A*, guiding the search efficiently.

---

## 🗂️ Project Structure

```
project-root/
├── models/                  # Model classes: MLP, XGBoost, CNN, etc.
│   └── model_interface.py   # Interface for all heuristic models
├── puzzles/                 # Puzzle definitions (e.g., Rubik's Cube)
│   └── puzzle_interface.py  # Interface for all puzzles
├── puzzle_solver.py         # Core state generation and A* solver logic
├── tests.py                 # Test scripts for training and solving
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
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

4. **Run training and testing**

   ```bash
   python tests.py
   ```

---

## 🧠 Model Options

You can plug in different heuristic models by implementing the `Model` interface, like the ones implemented:

* 🔢 **MLP (Multi-Layer Perceptron)**
* 🌲 **XGBoost Regressor**
* 🧠 **CNN (Convolutional Neural Network)**

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
