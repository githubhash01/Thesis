# Thesis Project

## Overview
This project is focused on simulating and analyzing dynamical systems in MuJoCo using both default and modified solvers. The project structure is organized into different components for experiments, solvers, models, and data storage. Additionally, the project includes a research group's repository, `mjx`, which serves as a dependency for handling MuJoCo simulations.

## Project Structure
```
Thesis/
│── solvers/                   # Contains solvers for MuJoCo simulations
│   │── original_solver.py     # Copy of the default MuJoCo solver
│   │── modified_solver.py     # Modified MuJoCo solver with implicit solvers
│── experiments/                  # Contains all experiment-related scripts and data
│   │── analysis.py               # Script for analyzing experimental results
│   │── main.py                   # Main script to run experiments
│   │── simulation.py              # Handles simulation logic, including state transitions and Jacobians
│   │── trajectory_optimisation.py # Trajectory optimisation on ball rebounding from the wall 
│   │── xmls/                      # Stores models in XML format
│   │   │── one_bounce.xml         # Model for one-bounce experiment
│   │   │── two_cart.xml           # Model for two-cart experiment
│   │   │── finger.xml             # Model for finger experiment
│   │── stored_data/               # Stores experiment results (states & Jacobians)
│       │── one_bounce/            # Data for one-bounce experiment
│       │── two_cart/              # Data for two-cart experiment
│       │── finger/                # Data for finger experiment
│
│── mjx/                           # Clone of the `mjx_fitted_iteration` repository from the research group
│── Report                         # Contains the final report for the project 
│── README.md                      # Project overview and instructions
```

## Components

### `experiments/`
This directory contains all the code related to running experiments, simulations, and analysis. The main scripts include:

- **`main.py`**: The primary entry point for running experiments. It initializes simulations and calls relevant functions from `simulation.py`.
- **`simulation.py`**: Defines simulation functions, including state transitions and Jacobian computations, using different MuJoCo solvers.
- **`analysis.py`**: Handles result processing and analysis, extracting insights from the stored simulation data.
- **`trajectory_optimisation.py`**: Contains functions for optimizing the trajectory of a ball rebounding from a wall 
- - works well with implicit solvers, doesn't work with the default solver, and problems implementing the finite difference method 

### `solvers/`
This directory contains solver implementations:
- **`original_solver.py`**: A copy of the default MuJoCo solver.
- **`modified_solver.py`**: A modified solver that incorporates implicit solvers for improved stability and accuracy.

### `xmls/`
This directory contains XML files defining MuJoCo models for different experiments:
- `one_bounce.xml` – Single object bouncing.
- `two_cart.xml` – Two-cart system.
- `finger.xml` – Robotic finger dynamics.
- `rebound.xml` - Ball rebounding from the wall

### `stored_data/`
Contains simulation results for each experiment. The results include:
- **States**: The evolution of the system's state over time.
- **Jacobians**: State transition Jacobians computed during simulation.

### `mjx/`
This directory is a cloned repository (`mjx_fitted_iteration`) from the research group. It serves as a dependency for running MuJoCo simulations and processing experimental data.

## Running Experiments
To run an experiment, use:
```sh
mjpython experiments/main.py
```
This will execute the main experiment pipeline, which includes setting up the simulation, running the solver, and storing results.

## Analysis
Once experiments are completed, results can be analyzed using:
```sh
python3 analysis.py
```
This script loads stored data and processes the results to extract insights from the simulations.

## Running Trajectory Optimisation
To run the trajectory optimisation experiment, use:
```sh
mjpython experiments/trajectory_optimisation.py
```
This will optimize the trajectory of a ball rebounding from a wall using gradient mode of your choice.

## Dependencies
- Python 3.x
- JAX
- MuJoCo
- NumPy
- Other dependencies from `mjx_fitted_iteration`

## Contact
For any questions or issues, refer to the research group’s documentation or reach out to the maintainers of this project.

