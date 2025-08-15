---
layout: page
title: Methodology
permalink: /methodology/
nav_order: 2
---

# Mathematical Methodology

This page details the mathematical framework underlying our Bayesian approach to PDE inverse problems with certified uncertainty quantification.

## Problem Formulation

### Forward Problem
Consider a partial differential equation of the form:

$$\mathcal{L}(\theta) u = f \quad \text{in } \Omega$$

$$\mathcal{B} u = g \quad \text{on } \partial\Omega$$

where:
- $\mathcal{L}(\theta)$ is a differential operator parameterized by $\theta \in \mathbb{R}^d$
- $u: \Omega \rightarrow \mathbb{R}$ is the solution field
- $f$ is the source term
- $\mathcal{B}$ represents boundary conditions
- $\Omega \subset \mathbb{R}^n$ is the spatial domain

### Inverse Problem
Given noisy observations $\mathbf{y} = (y_1, \ldots, y_m)^T$ at locations $\mathbf{x}_1, \ldots, \mathbf{x}_m$:

$$y_i = \mathcal{G}(\theta)(\mathbf{x}_i) + \epsilon_i$$

where:
- $\mathcal{G}(\theta): \Omega \rightarrow \mathbb{R}$ is the parameter-to-observable map
- $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ is observation noise

**Goal**: Estimate $\theta$ and quantify uncertainty with statistical guarantees.

## Bayesian Framework

### Prior Distribution
We specify a prior distribution $\pi(\theta)$ that encodes our knowledge about the parameters before observing data. Common choices include:

**Gaussian Prior**:
$$\pi(\theta) = \mathcal{N}(\mu_0, \Sigma_0)$$

**Log-normal Prior** (for positive parameters):
$$\log \theta_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$$

**Uniform Prior** (for bounded parameters):
$$\theta_i \sim \mathcal{U}(a_i, b_i)$$

### Likelihood Function
Assuming independent Gaussian noise:

$$\mathcal{L}(\mathbf{y}|\theta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathcal{G}(\theta)(\mathbf{x}_i))^2}{2\sigma^2}\right)$$

### Posterior Distribution
By Bayes' theorem:

$$\pi(\theta|\mathbf{y}) = \frac{\mathcal{L}(\mathbf{y}|\theta)\pi(\theta)}{\int \mathcal{L}(\mathbf{y}|\theta')\pi(\theta') d\theta'}$$

The posterior is typically intractable analytically, requiring computational methods.

## Computational Methods

### Markov Chain Monte Carlo (MCMC)

#### Metropolis-Hastings Algorithm
1. Initialize $\theta^{(0)}$
2. For $t = 1, 2, \ldots, T$:
   - Propose $\theta^* \sim q(\cdot|\theta^{(t-1)})$
   - Compute acceptance probability:
     $$\alpha = \min\left(1, \frac{\pi(\theta^*|\mathbf{y})q(\theta^{(t-1)}|\theta^*)}{\pi(\theta^{(t-1)}|\mathbf{y})q(\theta^*|\theta^{(t-1)})}\right)$$
   - Accept $\theta^{(t)} = \theta^*$ with probability $\alpha$, otherwise $\theta^{(t)} = \theta^{(t-1)}$

#### Hamiltonian Monte Carlo (HMC)
HMC uses gradient information to propose efficient moves:

1. Augment with momentum: $(\theta, p) \sim \mathcal{N}(0, M)$
2. Evolve Hamiltonian dynamics:
   $$\frac{d\theta}{dt} = M^{-1}p, \quad \frac{dp}{dt} = -\nabla_\theta U(\theta)$$
   where $U(\theta) = -\log \pi(\theta|\mathbf{y})$
3. Use leapfrog integration for numerical approximation
4. Accept/reject based on energy conservation

### Variational Inference

Approximate the posterior with a tractable family $\mathcal{Q}$:

$$q^*(\theta) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(\theta) \| \pi(\theta|\mathbf{y}))$$

Equivalently, maximize the Evidence Lower BOund (ELBO):

$$\text{ELBO}(q) = \mathbb{E}_q[\log \pi(\mathbf{y}, \theta)] - \mathbb{E}_q[\log q(\theta)]$$

#### Mean-Field Variational Inference
Assume factorized posterior:
$$q(\theta) = \prod_{i=1}^d q_i(\theta_i)$$

Update each factor iteratively:
$$\log q_j^*(\theta_j) = \mathbb{E}_{q_{-j}}[\log \pi(\mathbf{y}, \theta)] + \text{const}$$

## Uncertainty Quantification

### Concentration Inequalities

#### Hoeffding's Inequality
For bounded random variables $X_i \in [a, b]$:

$$P(|\bar{X}_n - \mathbb{E}[X]| \geq t) \leq 2\exp\left(-\frac{2nt^2}{(b-a)^2}\right)$$

**Application**: With probability $1-\delta$:
$$|\bar{\theta}_n - \mathbb{E}[\theta]| \leq (b-a)\sqrt{\frac{\log(2/\delta)}{2n}}$$

#### Bernstein's Inequality
For sub-exponential random variables:

$$P(|\bar{X}_n - \mathbb{E}[X]| \geq t) \leq 2\exp\left(-\frac{nt^2}{2(\sigma^2 + bt/3)}\right)$$

where $\sigma^2$ is the variance and $b$ bounds the sub-exponential parameter.

### PAC-Bayes Bounds

#### McAllester Bound
For any posterior $\rho$ and prior $\pi$:

$$P\left(R(\rho) \leq \hat{R}(\rho) + \sqrt{\frac{\text{KL}(\rho\|\pi) + \log(2\sqrt{n}/\delta)}{2n}}\right) \geq 1-\delta$$

where:
- $R(\rho)$ is the true risk
- $\hat{R}(\rho)$ is the empirical risk
- $\text{KL}(\rho\|\pi)$ is the KL divergence between posterior and prior

#### Seeger Bound (Tighter)
The tightest known PAC-Bayes bound solves:

$$\hat{R}(\rho) + \frac{\text{KL}(\rho\|\pi) + \log(2\sqrt{n}/\delta)}{2n(1-R(\rho))} = R(\rho)$$

### Coverage Analysis

#### Empirical Coverage
For confidence intervals $[L_i, U_i]$ and true values $\theta_i^*$:

$$\text{Coverage} = \frac{1}{m}\sum_{i=1}^m \mathbf{1}[\theta_i^* \in [L_i, U_i]]$$

#### Calibration Metrics
- **Average Coverage Error**: $|\text{Coverage} - (1-\alpha)|$
- **Interval Width**: $\mathbb{E}[U_i - L_i]$
- **Sharpness**: Conditional coverage given interval width

## Forward PDE Solvers

### Finite Difference Methods

#### 1D Elliptic Problem
For $-\frac{d}{dx}\left(D(x)\frac{du}{dx}\right) + R(x)u = f(x)$:

**Discretization**:
$$-\frac{D_{i+1/2}(u_{i+1} - u_i) - D_{i-1/2}(u_i - u_{i-1})}{h^2} + R_i u_i = f_i$$

where $D_{i+1/2} = \frac{1}{2}(D_i + D_{i+1})$.

#### 2D Elliptic Problem
For $-\nabla \cdot (D \nabla u) + Ru = f$:

**5-point stencil**:
$$-D_{i,j}\left(\frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h_x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h_y^2}\right) + R_{i,j}u_{i,j} = f_{i,j}$$

### Finite Element Methods

#### Weak Formulation
Find $u \in H^1_0(\Omega)$ such that:
$$a(u,v) = \langle f, v \rangle \quad \forall v \in H^1_0(\Omega)$$

where $a(u,v) = \int_\Omega D \nabla u \cdot \nabla v + Ruv \, dx$.

#### Galerkin Discretization
Approximate $u \approx \sum_{j=1}^N U_j \phi_j(x)$ where $\{\phi_j\}$ are basis functions.

**System**: $\mathbf{A}\mathbf{U} = \mathbf{b}$

$$A_{ij} = a(\phi_j, \phi_i), \quad b_i = \langle f, \phi_i \rangle$$

## Model Selection and Validation

### Information Criteria

#### Deviance Information Criterion (DIC)
$$\text{DIC} = -2\log p(\mathbf{y}|\hat{\theta}) + 2p_D$$

where $p_D$ is the effective number of parameters.

#### Watanabe-Akaike Information Criterion (WAIC)
$$\text{WAIC} = -2\sum_{i=1}^n \log\left(\frac{1}{S}\sum_{s=1}^S p(y_i|\theta^{(s)})\right) + 2\sum_{i=1}^n \text{Var}_s(\log p(y_i|\theta^{(s)}))$$

### Cross-Validation
- **K-fold CV**: Split data into K folds, train on K-1, test on 1
- **Leave-one-out CV**: Special case with K = n
- **Time series CV**: Respect temporal ordering in data splits

## Computational Considerations

### Scalability
- **Parallel MCMC**: Multiple chains, parallel tempering
- **Sparse Linear Algebra**: Exploit PDE structure
- **Adaptive Meshing**: Refine mesh based on posterior uncertainty

### Convergence Diagnostics
- **Gelman-Rubin Statistic**: $\hat{R} = \sqrt{\frac{\text{Var}^+(\psi)}{W}}$
- **Effective Sample Size**: Account for autocorrelation
- **Trace Plots**: Visual inspection of mixing

### Numerical Stability
- **Preconditioning**: Improve condition number of linear systems
- **Regularization**: Add small diagonal terms for stability
- **Robust Statistics**: Use median and MAD instead of mean and variance