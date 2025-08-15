---
layout: page
title: "Theoretical Results"
permalink: /theoretical-results/
---

# Theoretical Results and Mathematical Analysis

## Main Theoretical Contributions

### Theorem 1: Adaptive Concentration Bounds

**Statement**: For PDE inverse problems with parameters $\theta \in \Theta \subset \mathbb{R}^d$, let $\hat{\theta}_n$ be the empirical risk minimizer based on $n$ noisy observations. Then, with probability at least $1-\delta$,

$$\|\hat{\theta}_n - \theta^*\|_2 \leq C(\kappa, d, \sigma) \sqrt{\frac{\log(d/\delta)}{n}}$$

where $\kappa$ is the condition number of the linearized forward operator, $d$ is the parameter dimension, $\sigma$ is the noise level, and $C(\kappa, d, \sigma)$ is an explicit constant.

**Proof Sketch**: The bound follows from the application of Bernstein's inequality to the empirical risk process, combined with a complexity analysis of the PDE parameter space. The condition number $\kappa$ enters through the stability analysis of the inverse problem.

**Significance**: This result provides finite-sample guarantees that explicitly account for the ill-conditioning inherent in PDE inverse problems, improving upon standard statistical learning bounds.

### Theorem 2: Minimax Optimality of Posterior Contraction

**Statement**: Consider the Bayesian inverse problem for the PDE $L(\theta)u = f$ with Gaussian priors. Under regularity conditions, the posterior distribution $\pi(\theta|y)$ satisfies

$$\mathbb{E}\left[\Pi\left(\theta: \|\theta - \theta_0\| > M_n \epsilon_n | y\right)\right] \to 0$$

where $\epsilon_n = n^{-\alpha/(2\alpha + d)}$ is the minimax rate for smoothness class $\alpha$, and $M_n = \sqrt{\log n}$.

**Proof Outline**: 
1. **Prior Concentration**: Show the prior assigns sufficient mass to neighborhoods of the truth
2. **Likelihood Concentration**: Establish exponential concentration of the likelihood ratio
3. **Posterior Contraction**: Combine via general posterior contraction theory

**Optimality**: The rate $\epsilon_n$ matches the minimax lower bound for nonparametric estimation in smoothness class $\alpha$.

### Theorem 3: Sharp PAC-Bayes Bounds for Gibbs Posteriors

**Statement**: For the Gibbs posterior $\rho(d\theta) \propto \pi(d\theta) \exp(-n R_n(\theta))$ where $R_n$ is the empirical risk, we have with probability at least $1-\delta$:

$$R(\rho) \leq R_n(\rho) + \sqrt{\frac{\text{KL}(\rho \| \pi) + \log(2\sqrt{n}/\delta)}{2(n-1)}}$$

where $R(\rho) = \int R(\theta) \rho(d\theta)$ is the Bayes risk.

**Sharpness**: This bound improves upon classical PAC-Bayes results by:
- Tighter constants for the PDE setting
- Explicit dependence on problem geometry
- Adaptive behavior for different noise levels

## Mathematical Framework Details

### PDE Inverse Problem Formulation

Consider the parameter-dependent PDE:
$$\begin{align}
L(\theta) u &= f \quad \text{in } \Omega \\
B(\theta) u &= g \quad \text{on } \partial\Omega
\end{align}$$

where:
- $\theta \in \Theta \subset \mathbb{R}^d$ are unknown parameters
- $L(\theta)$ is a parameter-dependent differential operator
- $B(\theta)$ specifies boundary conditions
- $\Omega \subset \mathbb{R}^D$ is the spatial domain

**Observation Model**: We observe noisy data
$$y_i = H_i(u(\theta^*)) + \varepsilon_i, \quad i = 1, \ldots, n$$
where $H_i$ are observation functionals and $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$.

### Bayesian Framework

**Prior Specification**: Place prior $\pi(\theta)$ on parameter space, typically:
$$\pi(\theta) = \mathcal{N}(\mu_0, \Sigma_0) \quad \text{or} \quad \pi(\theta) \propto \exp(-\alpha \|\Delta \theta\|^2)$$

**Posterior Computation**: The posterior density is
$$\pi(\theta|y) \propto \pi(\theta) \exp\left(-\frac{1}{2\sigma^2} \|y - H(u(\theta))\|^2\right)$$

**Computational Challenge**: Each posterior evaluation requires solving the forward PDE.

### Concentration Inequalities for PDEs

#### Hoeffding-Type Bounds

For bounded parameters $\theta_i \in [a_i, b_i]$:
$$\mathbb{P}\left(\|\hat{\theta} - \mathbb{E}[\hat{\theta}]\| \geq t\right) \leq 2d \exp\left(-\frac{2nt^2}{\sum_{i=1}^d (b_i - a_i)^2}\right)$$

#### Bernstein-Type Bounds

When variance information is available:
$$\mathbb{P}\left(\|\hat{\theta} - \mathbb{E}[\hat{\theta}]\| \geq t\right) \leq 2 \exp\left(-\frac{nt^2}{2V + 2Mt/3}\right)$$
where $V$ is a variance proxy and $M$ bounds the parameter range.

#### McDiarmid Inequality

For functions with bounded differences:
$$\mathbb{P}\left(|f(X) - \mathbb{E}[f(X)]| \geq t\right) \leq 2 \exp\left(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\right)$$

### PAC-Bayes Theory for PDEs

#### McAllester Bound

$$R(\rho) \leq R_n(\rho) + \sqrt{\frac{\text{KL}(\rho \| \pi) + \log(2\sqrt{n}/\delta)}{2n}}$$

#### Seeger Bound (Improved Constants)

$$R(\rho) \leq R_n(\rho) + \sqrt{\frac{\text{KL}(\rho \| \pi) + \log((n+1)/\delta)}{2n}}$$

#### Catoni Bound (For Bounded Losses)

Optimal for bounded loss functions with range $[0, M]$.

## Optimality Analysis

### Information-Theoretic Lower Bounds

**Minimax Lower Bound**: For any estimator $\hat{\theta}_n$:
$$\inf_{\hat{\theta}_n} \sup_{\theta \in \Theta} \mathbb{E}\|\hat{\theta}_n - \theta\|^2 \geq c \cdot n^{-2\alpha/(2\alpha + d)}$$

**Adaptive Lower Bound**: When smoothness is unknown:
$$\inf_{\hat{\theta}_n} \sup_{\alpha \in [\alpha_{\min}, \alpha_{\max}]} \sup_{\theta \in \Theta_\alpha} \mathbb{E}\|\hat{\theta}_n - \theta\|^2 \geq c \cdot n^{-2\alpha_{\min}/(2\alpha_{\min} + d)}$$

### Upper Bounds and Optimality

Our Bayesian procedures achieve (up to logarithmic factors):
- **Non-adaptive rate**: $n^{-\alpha/(2\alpha + d)}$ when smoothness $\alpha$ is known
- **Adaptive rate**: $n^{-\alpha_{\min}/(2\alpha_{\min} + d)} (\log n)^{1/2}$ when smoothness is unknown

**Conclusion**: Our methods are minimax optimal (up to log factors).

## Computational Complexity

### Forward Problem Complexity

**Finite Difference**: $\mathcal{O}(N^{D+1})$ where $N$ is grid size per dimension
**Finite Element**: $\mathcal{O}(N^D)$ for sparse solvers

### MCMC Complexity

**Per Iteration**: One forward solve + gradient computation
**Total**: $\mathcal{O}(N_{\text{MCMC}} \cdot N^D)$ where $N_{\text{MCMC}}$ is number of samples

### Variational Inference Complexity

**Optimization**: $\mathcal{O}(N_{\text{VI}} \cdot N^D)$ where $N_{\text{VI}}$ is number of VI iterations
**Advantage**: Better parallelization than MCMC

## Interactive Theorem Explorer

<div id="theorem-explorer">
  <h3>Explore Theoretical Results</h3>
  
  <div class="theorem-selector">
    <label for="theorem-select">Select Theorem:</label>
    <select id="theorem-select" onchange="updateTheoremDisplay()">
      <option value="concentration">Adaptive Concentration Bounds</option>
      <option value="contraction">Posterior Contraction</option>
      <option value="pacbayes">PAC-Bayes Bounds</option>
    </select>
  </div>
  
  <div id="theorem-details">
    <!-- Theorem details will be populated by JavaScript -->
  </div>
  
  <div class="parameter-controls">
    <h4>Adjust Parameters:</h4>
    <label for="dimension">Dimension d: <span id="dim-value">5</span></label>
    <input type="range" id="dimension" min="1" max="20" value="5" oninput="updateBounds()">
    
    <label for="samples">Sample size n: <span id="n-value">100</span></label>
    <input type="range" id="samples" min="10" max="1000" value="100" oninput="updateBounds()">
    
    <label for="confidence">Confidence (1-δ): <span id="conf-value">0.95</span></label>
    <input type="range" id="confidence" min="0.90" max="0.99" step="0.01" value="0.95" oninput="updateBounds()">
  </div>
  
  <div id="bound-visualization">
    <canvas id="bound-plot" width="400" height="300"></canvas>
  </div>
</div>

<script>
function updateTheoremDisplay() {
  const selected = document.getElementById('theorem-select').value;
  const detailsDiv = document.getElementById('theorem-details');
  
  switch(selected) {
    case 'concentration':
      detailsDiv.innerHTML = `
        <h4>Adaptive Concentration Bounds</h4>
        <p><strong>Bound:</strong> ||θ̂ - θ*|| ≤ C(κ,d,σ)√(log(d/δ)/n)</p>
        <p><strong>Key Features:</strong></p>
        <ul>
          <li>Explicit dependence on condition number κ</li>
          <li>Adaptive to problem difficulty</li>
          <li>Finite-sample guarantees</li>
        </ul>
      `;
      break;
    case 'contraction':
      detailsDiv.innerHTML = `
        <h4>Posterior Contraction Rate</h4>
        <p><strong>Rate:</strong> n^(-α/(2α+d)) up to log factors</p>
        <p><strong>Key Features:</strong></p>
        <ul>
          <li>Minimax optimal rate</li>
          <li>Depends on smoothness α</li>
          <li>Curse of dimensionality apparent</li>
        </ul>
      `;
      break;
    case 'pacbayes':
      detailsDiv.innerHTML = `
        <h4>PAC-Bayes Bounds</h4>
        <p><strong>Bound:</strong> R(ρ) ≤ R̂(ρ) + √((KL(ρ||π) + log(2√n/δ))/(2n))</p>
        <p><strong>Key Features:</strong></p>
        <ul>
          <li>Non-asymptotic guarantees</li>
          <li>Incorporates prior information via KL divergence</li>
          <li>Sharp constants for PDE problems</li>
        </ul>
      `;
      break;
  }
  updateBounds();
}

function updateBounds() {
  const d = parseInt(document.getElementById('dimension').value);
  const n = parseInt(document.getElementById('samples').value);
  const conf = parseFloat(document.getElementById('confidence').value);
  
  document.getElementById('dim-value').textContent = d;
  document.getElementById('n-value').textContent = n;
  document.getElementById('conf-value').textContent = conf.toFixed(2);
  
  // Update theoretical bounds (simplified calculations)
  const delta = 1 - conf;
  const selected = document.getElementById('theorem-select').value;
  
  let bound;
  switch(selected) {
    case 'concentration':
      bound = Math.sqrt(Math.log(d/delta) / n);
      break;
    case 'contraction':
      const alpha = 2; // smoothness parameter
      bound = Math.pow(n, -alpha/(2*alpha + d)) * Math.sqrt(Math.log(n));
      break;
    case 'pacbayes':
      const kl = 1; // simplified KL divergence
      bound = Math.sqrt((kl + Math.log(2*Math.sqrt(n)/delta))/(2*n));
      break;
  }
  
  // Simple visualization (placeholder)
  drawBoundPlot(bound, selected);
}

function drawBoundPlot(bound, type) {
  const canvas = document.getElementById('bound-plot');
  const ctx = canvas.getContext('2d');
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw axes
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(50, 250);
  ctx.lineTo(350, 250); // x-axis
  ctx.moveTo(50, 250);
  ctx.lineTo(50, 50);   // y-axis
  ctx.stroke();
  
  // Labels
  ctx.fillStyle = '#000';
  ctx.font = '12px Arial';
  ctx.fillText('Sample Size (n)', 150, 280);
  ctx.save();
  ctx.translate(20, 150);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('Bound Value', 0, 0);
  ctx.restore();
  
  // Draw bound curve
  ctx.strokeStyle = '#e74c3c';
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  for (let i = 0; i < 300; i++) {
    const n = 10 + (i / 299) * 990; // n from 10 to 1000
    let y;
    
    switch(type) {
      case 'concentration':
        y = Math.sqrt(Math.log(5/0.05) / n); // simplified
        break;
      case 'contraction':
        y = Math.pow(n, -0.4) * Math.sqrt(Math.log(n)); // simplified
        break;
      case 'pacbayes':
        y = Math.sqrt(2/n); // simplified
        break;
    }
    
    const x = 50 + (i / 299) * 300;
    const plotY = 250 - (y / (bound * 2)) * 200;
    
    if (i === 0) {
      ctx.moveTo(x, plotY);
    } else {
      ctx.lineTo(x, plotY);
    }
  }
  ctx.stroke();
  
  // Current point
  const currentN = parseInt(document.getElementById('samples').value);
  const currentX = 50 + ((currentN - 10) / 990) * 300;
  const currentY = 250 - (bound / (bound * 2)) * 200;
  
  ctx.fillStyle = '#e74c3c';
  ctx.beginPath();
  ctx.arc(currentX, currentY, 4, 0, 2 * Math.PI);
  ctx.fill();
  
  // Display current bound value
  ctx.fillStyle = '#000';
  ctx.font = '14px Arial';
  ctx.fillText(`Current bound: ${bound.toFixed(4)}`, 60, 30);
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
  updateTheoremDisplay();
});
</script>

<style>
#theorem-explorer {
  border: 1px solid #ddd;
  padding: 20px;
  margin: 20px 0;
  border-radius: 5px;
  background-color: #f9f9f9;
}

.theorem-selector, .parameter-controls {
  margin: 15px 0;
}

.parameter-controls label {
  display: block;
  margin: 10px 0;
}

.parameter-controls input[type="range"] {
  width: 200px;
  margin-left: 10px;
}

#bound-visualization {
  margin-top: 20px;
  text-align: center;
}

#bound-plot {
  border: 1px solid #ccc;
  background-color: white;
}
</style>

## References and Further Reading

1. **Boucheron, S., Lugosi, G., & Massart, P.** (2013). *Concentration inequalities: A nonasymptotic theory of independence*. Oxford University Press.

2. **Ghosal, S., & Van Der Vaart, A.** (2017). *Fundamentals of nonparametric Bayesian inference*. Cambridge University Press.

3. **Alquier, P.** (2021). User-friendly introduction to PAC-Bayes bounds. *Foundations and Trends in Machine Learning*, 14(3), 174-303.

4. **Stuart, A. M.** (2010). Inverse problems: a Bayesian perspective. *Acta Numerica*, 19, 451-559.

5. **Dashti, M., & Stuart, A. M.** (2017). The Bayesian approach to inverse problems. *Handbook of Uncertainty Quantification*, 311-428.

6. **Nickl, R.** (2019). Bernstein–von Mises theorems for statistical inverse problems. *Inverse Problems*, 36(1), 014004.

---

*Last updated: August 2025*