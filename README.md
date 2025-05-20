# A* Pathfinding Algorithm Comparison

This project presents a detailed examination and implementation of the A* (A-star) pathfinding algorithm compared with other fundamental pathfinding algorithms such as Dijkstra's Algorithm and Breadth-First Search (BFS).

## Project Overview

A* is a widely used graph traversal and pathfinding algorithm known for its completeness, optimality, and efficiency. This project:
- Implements A* and compares it with Dijkstra's algorithm on a real road network (Küçükçekmece, Istanbul)
- Analyzes the mathematical formulation and complexity of A*
- Visualizes the search space exploration of both algorithms
- Demonstrates the efficiency gains of using A* with a proper heuristic

## Algorithms Implemented

1. **A* Algorithm** - An informed search algorithm using a heuristic function to guide exploration
2. **Dijkstra's Algorithm** - An uninformed search algorithm for finding shortest paths
3. **Path Reconstruction** - Method to construct the final optimal path

## Key Findings

- **Performance Comparison:** A* expanded only ~120 nodes to find the optimal path, while Dijkstra's algorithm expanded ~984 nodes for the same path
- **Path Optimality:** Both algorithms found the same optimal path of 5.47 km
- **Search Efficiency:** A* provides significant performance improvements by focusing the search toward the goal
- **Visualization:** The project includes visualizations that clearly show A*'s directed search pattern versus Dijkstra's uniform expansion

## Implementation Details

The project uses:
- OSMnx library to download and process real-world map data
- Heapq for priority queue implementations
- Matplotlib for visualization and animation
- Euclidean distance as an admissible heuristic for A*

## Running the Code

The main Python script (`Midterm_Samed_Tok.py`) performs the following actions:
1. Downloads and processes the map of Küçükçekmece, Istanbul
2. Implements A* and Dijkstra's algorithm
3. Runs both algorithms on the same start and end points
4. Visualizes the explored nodes and final path
5. Creates an animation comparing the exploration patterns

## Visualizations

The project generates several visualizations:
- Maps showing nodes expanded by each algorithm
- The final optimal path
- Animated comparison of how A* and Dijkstra explore the graph

## Theoretical Foundation

The project includes a thorough theoretical analysis of:
- A*'s mathematical formulation using f(n) = g(n) + h(n)
- Admissible heuristics and their role in guaranteeing optimality
- Time and space complexity analysis
- Comparative advantages over other pathfinding algorithms

## Author

Samed Tok

## Course Information

- Computer Engineering Department
- CMPE304 Graph Theory
- Kadir Has University
- Spring 2025
