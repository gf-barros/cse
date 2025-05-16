# 🧠 Computational Science and Engineering

Welcome to my personal study repository on **Computational Science and Engineering (CSE)**!  
Here, you'll find implementations, notes, and experiments with numerical methods and solvers that form the backbone of scientific computing.

The goal of this repository is to **learn by doing** — every topic is explored through code (in **Python** and **C++**) with clear documentation, examples, and, whenever possible, visualizations.

---

## 📚 Topics Covered

This repository is organized by topic, each in its own folder with detailed explanations and examples:

### 🔧 Discretization Methods
- **Finite Difference Method (FDM)**  
  Basic concepts, 1D/2D problems, boundary conditions, stencil operators, etc.

- **Finite Volume Method (FVM)**  
  Conservative formulations, fluxes, 1D/2D meshes, applications to transport equations.

- **Finite Element Method (FEM)**  
  Weak form derivation, 1D/2D implementations, shape functions, mesh handling.

### ⏱️ Time Integration Methods
- Explicit and implicit schemes (Euler, Runge-Kutta, Crank-Nicolson)
- Stability analysis and time step selection
- Coupling with spatial discretizations

### 🧮 Solvers and Linear Algebra
- Direct solvers (LU, Cholesky)
- Iterative solvers (Jacobi, Gauss-Seidel, Conjugate Gradient)
- Sparse matrix representations and efficiency tips

### 📈 Bonus Topics (Coming Soon)
- Adaptive Mesh Refinement (AMR)
- Multigrid methods
- Spectral methods
- Parallel computing and performance tips

---

## 🚀 Structure of the Repository

```
cse-study/
├── fdm/
│   ├── python/
│   └── cpp/
├── fvm/
├── fem/
├── time_integration/
├── solvers/
└── README.md
```

Each method folder contains:
- `python/` — Clean, readable code with NumPy and matplotlib
- `cpp/` — Efficient C++ implementations, often with CMake build instructions
- `README.md` — Concepts, references, and usage instructions for the folder

---

## 🛠️ Getting Started

### Requirements (for Python)
- Python 3.8+
- NumPy
- Matplotlib
- Jupyter (optional, for interactive notebooks)

You can install the Python dependencies with:

```bash
pip install -r requirements.txt
```

### Building the C++ Code
Most C++ examples use **CMake**. From the folder you're working on:

```bash
mkdir build && cd build
cmake ..
make
./your_executable
```

---

## ✨ Why This Exists

Computational science is at the heart of engineering, physics, and even machine learning. This repo is my way of consolidating knowledge, practicing numerical thinking, and bridging theory with implementation. If you're a fellow student or enthusiast, I hope you find it helpful!

<!-- ---

## 📘 References & Further Reading

- **Numerical Recipes** – Press et al.
- **Finite Element Procedures** – Bathe
- **Computational Fluid Dynamics** – Versteeg & Malalasekera
- **An Introduction to Computational Science** – Landau & Páez -->

---

## 🤝 Contributions

While this is a personal learning repo, you're welcome to open issues, ask questions, or suggest improvements!

---

## 📬 Contact

Feel free to reach out on [LinkedIn](https://www.linkedin.com/in/gabrielfbarros/) or open an issue if you'd like to collaborate or chat about numerical methods and CSE.
