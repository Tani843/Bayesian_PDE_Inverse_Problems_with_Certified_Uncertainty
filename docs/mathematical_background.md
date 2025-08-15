# Mathematical Background

This document provides the mathematical foundation for Bayesian inverse problems in partial differential equations with certified uncertainty quantification.

## Problem Formulation

### Forward Problem

Consider a general PDE in domain Ω ⊂ ℝᵈ:

```
ℒ[u; θ] = f   in Ω
```

where:
- u(x) is the unknown solution field
- θ ∈ Θ ⊂ ℝᵖ are unknown parameters
- ℒ is a differential operator depending on θ
- f is the source term

**Common PDE Types:**

1. **Elliptic**: -∇·(D(x,θ)∇u) + R(x,θ)u = f(x,θ)
2. **Parabolic**: ∂u/∂t - ∇·(D(x,θ)∇u) + R(x,θ)u = f(x,θ)
3. **Hyperbolic**: ∂²u/∂t² - ∇·(D(x,θ)∇u) + R(x,θ)u = f(x,θ)

### Inverse Problem

Given noisy observations y = [y₁, ..., yₙ] at locations {x₁, ..., xₙ}, find θ such that:

```
yᵢ = G(θ)(xᵢ) + εᵢ,  i = 1, ..., n
```

where:
- G(θ): Θ → X is the parameter-to-solution map
- εᵢ ~ N(0, σ²) are independent noise terms

## Bayesian Framework

### Bayes' Theorem

The posterior distribution is:

```
π(θ|y) = π(y|θ)π(θ) / π(y)
```

where:
- π(θ) is the prior distribution
- π(y|θ) is the likelihood function
- π(y) is the marginal likelihood (evidence)

### Likelihood Function

For Gaussian noise:

```
π(y|θ) = (2πσ²)^(-n/2) exp(-1/(2σ²) ||y - G(θ)||²)
```

Taking logarithms:

```
log π(y|θ) = -n/2 log(2πσ²) - 1/(2σ²) Σᵢ(yᵢ - G(θ)(xᵢ))²
```

### Prior Distributions

**Common Prior Choices:**

1. **Log-normal**: θᵢ ~ LogNormal(μᵢ, σᵢ²) for positive parameters
2. **Gaussian**: θ ~ N(μ, Σ) for unbounded parameters  
3. **Gamma**: θᵢ ~ Gamma(α, β) for positive parameters
4. **Uniform**: θᵢ ~ Uniform(a, b) for bounded parameters

## Numerical Methods

### Finite Difference Method

**2D Second-Order Centered Differences:**

For the elliptic operator -∇·(D∇u):

```
-D(∂²u/∂x² + ∂²u/∂y²) ≈ -D((uᵢ₊₁,ⱼ - 2uᵢ,ⱼ + uᵢ₋₁,ⱼ)/h² + (uᵢ,ⱼ₊₁ - 2uᵢ,ⱼ + uᵢ,ⱼ₋₁)/h²)
```

This leads to the linear system:
```
Au = f
```

where A is a sparse matrix with 5-point stencil structure.

### Finite Element Method

**Weak Formulation:**

Find u ∈ H₀¹(Ω) such that:
```
∫_Ω D∇u·∇v dx + ∫_Ω Ruv dx = ∫_Ω fv dx  ∀v ∈ H₀¹(Ω)
```

**Galerkin Discretization:**

Using basis functions {φⱼ}:
```
u_h = Σⱼ uⱼφⱼ
```

leads to the system:
```
Ku = F
```

where:
- K is the stiffness matrix: Kᵢⱼ = ∫_Ω D∇φᵢ·∇φⱼ + Rφᵢφⱼ dx
- F is the load vector: Fᵢ = ∫_Ω fφᵢ dx

## MCMC Methods

### Metropolis-Hastings Algorithm

1. **Proposal**: θ* ~ q(θ*|θ⁽ᵗ⁾)
2. **Acceptance ratio**: α = min(1, π(θ*|y)q(θ⁽ᵗ⁾|θ*) / (π(θ⁽ᵗ⁾|y)q(θ*|θ⁽ᵗ⁾)))
3. **Accept/Reject**: θ⁽ᵗ⁺¹⁾ = θ* with probability α, otherwise θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾

**Random Walk Proposals:**
```
θ* = θ⁽ᵗ⁾ + ε,  ε ~ N(0, σ²I)
```

### Hamiltonian Monte Carlo

Introduces auxiliary momentum variables p:
```
H(θ, p) = U(θ) + K(p) = -log π(θ|y) + ½p^T M⁻¹p
```

Hamilton's equations:
```
dθ/dt = ∂H/∂p = M⁻¹p
dp/dt = -∂H/∂θ = -∇U(θ)
```

## Variational Inference

### Mean-Field Approximation

Approximate the posterior with a factorized distribution:
```
q(θ) = ∏ᵢ qᵢ(θᵢ)
```

### Evidence Lower Bound (ELBO)

Maximize:
```
ℒ(q) = 𝔼_q[log π(θ, y)] - 𝔼_q[log q(θ)]
      = 𝔼_q[log π(θ|y)] + log π(y) - 𝔼_q[log q(θ)]
```

### Gaussian Variational Families

**Mean-field Gaussian**: q(θ) = ∏ᵢ N(θᵢ; μᵢ, σᵢ²)

**Full-rank Gaussian**: q(θ) = N(θ; μ, Σ)

## Uncertainty Quantification

### Concentration Inequalities

#### Hoeffding's Inequality

For bounded random variables Xᵢ ∈ [aᵢ, bᵢ]:
```
ℙ(|S̄ₙ - 𝔼[S̄ₙ]| ≥ t) ≤ 2exp(-2n²t² / Σᵢ(bᵢ - aᵢ)²)
```

#### Bernstein's Inequality

For sub-exponential random variables:
```
ℙ(|S̄ₙ - 𝔼[S̄ₙ]| ≥ t) ≤ 2exp(-nt² / (2σ² + 2bt/3))
```

#### McDiarmid's Inequality

For functions with bounded differences:
```
ℙ(|f(X₁,...,Xₙ) - 𝔼[f(X₁,...,Xₙ)]| ≥ t) ≤ 2exp(-2t² / Σᵢcᵢ²)
```

### PAC-Bayes Bounds

#### McAllester Bound

For any posterior Q and prior P:
```
R(Q) ≤ R̂(Q) + √((KL(Q||P) + log(4n/δ)) / (2n))
```

where:
- R(Q) is the true risk
- R̂(Q) is the empirical risk
- KL(Q||P) is the KL divergence

#### Seeger Bound

```
R(Q) ≤ R̂(Q) + √((KL(Q||P) + log(2√n/δ)) / (2n-1))
```

#### Catoni Bound

For bounded losses in [0,1]:
```
R(Q) ≤ (1/λ) log 𝔼_Q[exp(λ(R̂(θ) + √((KL(Q||P) + log(1/δ)) / n)))]
```

## Convergence Theory

### MCMC Convergence

#### Ergodicity Conditions

1. **Irreducibility**: Can reach any set with positive probability
2. **Aperiodicity**: No cyclical behavior
3. **Positive Recurrence**: Expected return time is finite

#### Central Limit Theorem

For ergodic chains:
```
√n(θ̄ₙ - 𝔼[θ]) →ᵈ N(0, Σ)
```

where Σ depends on the autocorrelation structure.

### VI Convergence

#### ELBO Convergence

Under mild conditions:
```
lim_{n→∞} ℒₙ(q*) = ℒ(q*)
```

where q* minimizes KL(q||π).

## Error Analysis

### Discretization Error

**Finite Differences**: O(h²) for second-order schemes
**Finite Elements**: O(hᵖ⁺¹) for degree p elements

### Statistical Error

**MCMC**: O(1/√n) for sample mean estimates
**VI**: Depends on approximation quality and optimization

### Total Error Decomposition

```
||θ - θ̂|| ≤ ||θ - θₕ|| + ||θₕ - θ̂ₕ|| + ||θ̂ₕ - θ̂||
```

where:
- ||θ - θₕ||: Discretization error
- ||θₕ - θ̂ₕ||: Statistical error
- ||θ̂ₕ - θ̂||: Computational error

## Advanced Topics

### Multilevel Methods

Use hierarchy of discretizations to reduce computational cost:
```
𝔼[Q] ≈ Q₀ + Σₗ₌₁ᴸ (Qₗ - Qₗ₋₁)
```

### Dimension Reduction

**Proper Orthogonal Decomposition (POD)**:
Find basis {φᵢ} minimizing:
```
Σⱼ||u⁽ʲ⁾ - Σᵢ₌₁ʳ aᵢ⁽ʲ⁾φᵢ||²
```

### Adaptive Methods

**Error Indicators**: Use a posteriori estimates to guide refinement
**Goal-Oriented**: Minimize error in quantities of interest

## Implementation Considerations

### Computational Complexity

- **Forward solve**: O(N³) direct, O(N log N) with multigrid
- **MCMC**: O(M × cost_per_sample) where M is chain length
- **VI**: O(K × N_samples) where K is iterations

### Numerical Stability

- Use appropriate preconditioning
- Monitor condition numbers
- Implement adaptive timestepping for parabolic PDEs

### Parallel Computing

- Domain decomposition for PDE solves
- Multiple chains for MCMC
- Vectorized operations for VI

This mathematical framework provides the theoretical foundation for understanding and implementing the Bayesian PDE inverse problem solver with certified uncertainty quantification.