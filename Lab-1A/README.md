# Lab Project 1A

This README provides an overview and instructions for Lab Project 1A of the CI-4001 course.
## Project Overview
In this lab, we focus on the fundamentals of programming and algorithm design, using external clusters to perform computations. The project runs a NEURON model simulation for the fear response, and outputs the results to CI-4001/Lab-1A/fear_simulation/output/ directory and to the micro:bit display.
## Repository Structure
- `fear_simulation/`: Contains the NEURON model simulation code and related files.
- `output/`: Directory where simulation results are stored.
- `network/`: Configuration files for connecting to external clusters.
- `components/`: Contains reusable NEURON components for the simulation.
## File Descriptions
- `fear_simulation/build_network.py`: Script to build the neural network for the simulation.
- `fear_simulation/update_configs.py`: Script to update configuration files for the simulation.
- `fear_simulation/run_bionet.py`: Main script to run the NEURON simulation.
- `fear_simulation/check_output.py`: Script to analyze the output and compute oscillation frequency.
- `fear_simulation/parameters.py`: File to set simulation parameters.
- `network/cluster_config.json`: Configuration file for connecting to external clusters.
## Getting Started
To get started with the lab project, follow the setup and run instructions provided in the `fear_simulation/README.md` file.
