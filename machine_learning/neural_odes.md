# Neural Ordinary Differential Equations (Neural ODEs)

Neural Ordinary Differential Equations (Neural ODEs) introduced by [Chen et al.](https://arxiv.org/abs/1806.07366) in 2018, model the evolution of a system's state over time using a continuous function parameterized by a neural network.

In a standard neural network, the hidden state is represented as a discrete sequence of layers as shown below. These updates are like Euler's discretization of a continuous transformation.

$$
\boldsymbol{h}_{t+1} = \boldsymbol{h}_t + f(\boldsymbol{h}_t, \theta_t); \;\; t = 0, 1, ..., T
$$

In the limit as the step size goes to zero, this can be represented as a continuous transformation of the hidden state:

$$
\frac{d\boldsymbol{h}}{dt} = f(h(t), t, \theta)
$$

Using the neural ODE framework, a neural network is used to find the function $f$ that determines the dynamics (or the velocity field) of the system. Then, the ODE relating the hidden state $\boldsymbol{h}$ with function $f$ is solved using an ODE solver (e.g., `torchdiffeq.odeint`) to compute the output $\boldsymbol{h}(T)$ from the input $\boldsymbol{h}(0)$.

Inherently, ODE solver enforces the following:

$$
X(t_k) = X(0) + \int_{0}^{t_k} f(X(s), s, \theta) ds
$$

> $\boldsymbol{X}(0)$ is the input layer, and $\boldsymbol{X}(T)$ is the output layer.

## Advantages of Neural ODEs

According to the original paper:

- **Memory Efficiency**: Neural ODEs can be more memory efficient than traditional deep networks, as they do not require storing intermediate activations for backpropagation. Instead, they use the adjoint method to compute gradients, there by having a constant memory cost as a function of the network depth.
- **Adaptive Computation**: Depending on the complexity of the input, and the desired accuracy, Neural ODEs can adaptively choose the number of function evaluations needed to compute the output, which can lead to faster inference for simpler inputs.
- **Continuous-Time Modeling**: Neural ODEs naturally model continuous-time processes, containing observations at *irregular* time intervals.
- **Scalable and invertible normalizing flows**

## Standard NN $\leftrightarrow$ Neural ODEs

| Feature                | Standard NN                                            | Neural ODEs                                                                             |
| ---------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| Depth                  | Number of layers                                       | Integration time, number of function evaluations determined by the solver tolerance     |
| Continuity             | Discrete, $h_{t+1} = f(h_t, \theta)$                   | Continuous, $\frac{dh}{dt} = f(h(t), t, \theta)$                                        |
| Gradient               | Backpropagation through layers                         | Adjoint method for gradient computation                                                 |
| Step size              | Fixed                                                  | Adaptive                                                                                |
| Forward pass           | through layers                                         | through ODE solver, Ex: using `torchdiffeq.odeint`                                      |
| Backward pass          | through layers                                         | through ODE solver, using adjoint method                                                |
| Memory cost            | $O(L)$                                                 | $O(1)$, it just stores the current state and the end points of the integration interval |
| Function $\mathcal{f}$ | represents transformation mapping from input to output | veclocity field that defines the dynamics of the system                                 |

## A simple example of a Neural ODE

### Breathing Ellipse Example

Let's build a model that learns velocity field that governs the motion of points on a 2D curve such that the overall shape follows a specific geometric evolution. In this example, we will use a simple ellipse that expands and contracts over time, simulating a breathing motion whose aspect ratio changes according to the following function:

```math
\beta(t) = \beta_0 + \frac{1 - \beta_0 ^ 2}{\beta_{0} \sin{\omega T}} \sin(\omega t), \;\; t = 0, 1, ..., T
```

#### Assumptions

- **Initial state** ($t=0$): An ellipse with aspect ratio $\beta_0$
- **Intermediate states**: We have the observations of the curve at different time steps, possibly at irregular intervals. We want to learn the underlying velocity field that governs this motion.
- **Final state** ($t=T$): The ellipse reaches opposite aspect ratio $\beta_T = 1/\beta_0$.
- The area of the ellipse is conserved over time.
- Each point on the curve maps to a new position at each time step, following the velocity field defined by the Neural ODE.

#### Expected Results

- Non-autonomous vector field (explicit time dependence) that governs the motion of points on the curve.
- Conservation of topology (no crossing of trajectories)
- Resolution invariance (the same curve should be obtained regardless of the number of points sampled on the curve)

#### Implementation in PyTorch

- Define a neural network to find the function $f_{\theta}(X, t):\mathbb{R}^2 \times \mathbb{R} \rightarrow \mathbb{R}^2$, that determines the velocity field.
- This leads to the ODE: $\frac{dX}{dt} = f_{\theta}(X, t)$, where $X$ is the set of points on the curve.
- Use an ODE solver (e.g., `torchdiffeq.odeint`) to compute the output $X(T)$ from the input $X(0)$.
- Loss function: Mean squared error between the predicted curve at time $T$ and the true curve at time $T$.

```math
\mathcal{L}(\theta) = \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} \| X_i^{\theta}(t_k) - X_i^{\text{true}}(t_k) \|^2
```

#### Appendix: Derivation of velocity field

```math
\dfrac{x^2}{a(t)^2} + \dfrac{y^2}{b(t)^2} = 1 \\
```

Assuming the ellipse is centered at the origin, parametrized by the angle $\theta \in [0, 2\pi)$, the coordinates of points on the ellipse can be expressed as:

```math
x(t, \theta) = a(t) \cos{\theta} \\
y(t, \theta) = b(t) \sin{\theta}
```

Where $a(t)$ and $b(t)$ are the semi-major and semi-minor axes of the ellipse at time $t$. The aspect ratio $\beta(t)$ is defined as:

```math
\beta(t) = \frac{a(t)}{b(t)} \\
\beta(0) = \beta_0 \\
\beta(T) = \frac{1}{\beta_0}
```

As the area of the ellipse is conserved, we have:

```math
\begin{align*}
A &= \pi a(t) b(t) = \text{constant} \\
\Rightarrow a(t) b(t) &= A / \pi = \xi ^ 2 \\
\Rightarrow a(t) &= \xi \sqrt{\beta(t)} \\
\Rightarrow b(t) &= \frac{\xi}{\sqrt{\beta(t)}} \\
x(t, \theta) &= \xi \sqrt{\beta(t)} \cos{\theta} \\
y(t, \theta) &= \frac{\xi}{\sqrt{\beta(t)}} \sin{\theta} \\
\end{align*}
```

Differentiating the coordinates with respect to time to get the velocity field:

```math
\begin{align*}
\frac{dx}{dt} &= \frac{d}{dt} (\xi \sqrt{\beta(t)} \cos{\theta}) \\
&= \xi \cos{\theta} \frac{d}{dt} \sqrt{\beta(t)} \\
&= \frac{\xi}{2} \cos{\theta} \frac{\beta'(t)}{\sqrt{\beta(t)}} \\
\Rightarrow \frac{dx}{dt} &= \frac{x}{2} \frac{\beta'(t)}{\beta(t)} \\
\end{align*}
```

Similarly for $y$:

```math
\begin{align*}
\frac{dy}{dt} &= \frac{d}{dt} \left( \frac{\xi}{\sqrt{\beta(t)}} \sin{\theta} \right) \\
&= -\frac{\xi}{2} \sin{\theta} \frac{\beta'(t)}{\beta(t)\sqrt{\beta(t)}} \\
\Rightarrow \frac{dy}{dt} &= -\frac{y}{2} \frac{\beta'(t)}{\beta(t)}
\end{align*}
```

Hence, the velocity field governing the motion of points on the curve can be expressed as:

```math
\begin{align*}
\frac{d}{dt} \begin{pmatrix} x \\ y \end{pmatrix} &= \alpha(t) \begin{pmatrix} x \\ -y \end{pmatrix} \\
&= \frac{\beta'(t)}{2\beta(t)} \begin{pmatrix} x \\ -y \end{pmatrix}
\end{align*}
```

The neural ODE should learn this time-varying velocity field that governs the motion of points on the curve, allowing it to reconstruct the breathing ellipse dynamics from the observed data.

## Scope/Limitations of Neural ODEs

According to the original paper:

- **Minibatching**
- **Unique Solutions**
- **Manual tolerance Setting**
- **Errors in the reconstruction of forward trajectories**
