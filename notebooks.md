---
layout: page
title: Interactive Notebooks
permalink: /notebooks/
---

# Interactive Jupyter Notebooks

Hands-on tutorials and examples for learning Bayesian PDE inverse problems.

## Tutorial Series

### 🎓 **Beginner Level**

**[01. Introduction to Bayesian PDE Inverse Problems](notebooks/01_introduction.html)**
- Fundamental concepts and motivation
- Comparison with traditional point estimates
- Simple 1D heat equation example
- Visualization of uncertainty

**[02. PDE Solvers Demonstration](notebooks/02_pde_solvers_demo.html)** 
- Finite difference vs finite element methods
- Boundary condition handling
- Mesh convergence analysis
- Performance comparisons

### 🔬 **Intermediate Level**

**[03. Bayesian Inference Methods](notebooks/03_bayesian_inference.html)**
- MCMC sampling with Metropolis-Hastings
- Hamiltonian Monte Carlo implementation
- Variational inference techniques
- Convergence diagnostics and chain analysis

**[04. Certified Uncertainty Quantification](notebooks/04_uncertainty_quantification.html)**
- Concentration inequalities theory
- PAC-Bayes bounds implementation
- Coverage validation experiments
- Comparison with bootstrap methods

### 📊 **Visualization & Analysis**

**[05. Visualization Gallery](notebooks/05_visualization_gallery.html)**
- Publication-quality figure generation
- PDE solution plotting techniques
- Posterior distribution analysis
- Interactive dashboards with widgets

### 🚀 **Advanced Applications**

**[06. Complete Workflow Demonstration](notebooks/06_complete_workflow.html)**
- End-to-end problem solving
- 2D heat conduction case study
- Adaptive MCMC strategies
- Model validation and diagnostics

**[07. Advanced Research Examples](notebooks/07_advanced_examples.html)**
- Multi-physics coupled systems
- Time-dependent parameter estimation
- High-dimensional model selection
- Real-world engineering applications

## Running the Notebooks

### Local Setup
```bash
# Install Jupyter and dependencies
pip install jupyter matplotlib seaborn plotly

# Clone repository and start Jupyter
git clone https://github.com/bayesian-pde-solver/bayesian-pde-solver.git
cd bayesian-pde-solver/notebooks
jupyter notebook
```

### Online Access
- **Binder**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bayesian-pde-solver/bayesian-pde-solver/main?filepath=notebooks)
- **Google Colab**: [Open in Colab](https://colab.research.google.com/github/bayesian-pde-solver/bayesian-pde-solver/blob/main/notebooks/)

## Learning Path Recommendations

### For Students
1. Start with **01_introduction** for basic concepts
2. Work through **02_pde_solvers_demo** to understand numerical methods
3. Progress to **03_bayesian_inference** for statistical foundations
4. Complete **06_complete_workflow** for practical application

### For Researchers  
1. Review **04_uncertainty_quantification** for theoretical foundations
2. Explore **07_advanced_examples** for cutting-edge applications
3. Use **05_visualization_gallery** for publication graphics
4. Adapt examples for your specific research domain

### For Practitioners
1. Begin with **06_complete_workflow** for immediate application
2. Reference **02_pde_solvers_demo** for numerical implementation details
3. Use **05_visualization_gallery** for result presentation
4. Consult **04_uncertainty_quantification** for reliability analysis

## Interactive Features

Each notebook includes:
- **Live Code**: Editable examples with immediate execution
- **Mathematical Derivations**: Step-by-step theory with LaTeX rendering  
- **Interactive Plots**: Dynamic visualizations with parameter controls
- **Exercises**: Hands-on problems with solutions
- **Real Data**: Examples using actual experimental measurements

## Dataset Downloads

Example datasets used in notebooks:
- **Thermal imaging data**: Electronics cooling applications
- **Flow measurements**: Pipe flow with unknown roughness  
- **Material properties**: Composite thermal conductivity
- **Synthetic problems**: Controlled test cases with known solutions

## Contributing New Notebooks

We welcome contributions! See our [contributing guide](contributing.html) for:
- Notebook style guidelines
- Mathematical notation standards
- Code quality requirements  
- Review process details

## Notebook Dependencies

All notebooks are designed to run with minimal dependencies:
```
numpy >= 1.19
scipy >= 1.6  
matplotlib >= 3.3
jupyter >= 1.0
bayesian-pde-solver >= 1.0
```

Optional enhancements:
```
plotly >= 4.0      # Interactive plots
seaborn >= 0.11    # Statistical visualization
ipywidgets >= 7.0  # Interactive controls
```