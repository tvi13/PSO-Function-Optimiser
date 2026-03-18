## Particle Swarm Optimization (PSO) Dashboard

This project is an interactive implementation of the **Particle Swarm Optimization (PSO)** algorithm.

## Features
- **Interactive UI**: Built with Streamlit to allow real-time hyperparameter tuning.
- **Dynamic Optimization**: Minimize or Maximize user-defined functions.
- **Visual Convergence**: Live plotting of particle movement and global best trajectory.

## Complexity Analysis
* **Time Complexity**: $O(T \cdot P \cdot D)$, where $T$ is iterations, $P$ is population (particles), and $D$ is dimensions. This is highly efficient compared to brute-force grid searches.
* **Space Complexity**: $O(P \cdot D)$ to store the positions, velocities, and personal bests of the swarm.

## Real-World Applications
1.  **Antenna Design**: Optimizing the shape and parameters for maximum signal gain.
2.  **Power Systems**: Economic Dispatch problems to minimize fuel costs in power plants.
3.  **Financial Modeling**: Portfolio optimization to minimize risk for a given expected return.

## Installation
1. Clone the repo: `git clone https://github.com/tvi13/PSO-Function-Optimiser`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
