# Quantum Circuit Optimization Using Evolutionary Algorithms (Qiskit)

This project implements an advanced **quantum circuit optimization engine** using **evolutionary algorithms**.
It automatically reduces gate depth and circuit complexity while maintaining high fidelity to a target quantum circuit.

The project was originally developed as part of **Jugend forscht**, where I received a **certificate for scientific achievement**.
However, I was not allowed to officially participate in the competition because my supervising teacher did not approve the submission.

---

## 🚀 Overview

The optimizer takes a quantum circuit and applies:

- **Genetic Algorithms (GA)**
- **Adaptive mutation and crossover**
- **Tournament selection**
- **Hardware-aware optimization using DAG analysis**
- **Multi-objective scoring (depth + fidelity)**
- **Visualization of optimization progress**
- **QASM import/export support**

The system works with any Qiskit circuit or QASM file and can be configured for different backends.

---

## 🔬 Features

### 🧬 Evolutionary Optimization
- Intelligent crossover that preserves gate compatibility
- Multiple smart mutation strategies:
  - Gate reordering
  - Layer reversing
  - Layer shuffling
  - Gate removal
  - Local transpilation optimization

### ⚙ Hardware-Aware Logic
- Quantum DAG extraction
- Commutation-based rearrangement
- Graph partitioning via NetworkX
- Basis-gate analysis using Qiskit transpilation

### 📉 Circuit Evaluation
- Circuit depth
- Gate count
- Fidelity (statevector comparison)
- Weighted fitness scoring

### 📊 Visualization
- Fitness over generations
- Circuit depth reduction
- Fidelity progression

---

## 🧠 Technologies Used

- **Qiskit**
- **Python**
- **NetworkX**
- **NumPy**
- **Matplotlib**

---

## 🏁 Getting Started

### Install dependencies:

```bash
pip install qiskit networkx numpy matplotlib

