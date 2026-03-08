# Neural ODEs

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

## Normalising inputs, time and targets

Neural ODEs can be sensitive to the scale of the inputs and targets, as in the case of any machine learning model. Normalizing the inputs and targets can help improve the training stability and convergence of the model. For example, if we use standardization:

with standardisation,

$$
\begin{align}
\mathbf{\tilde{X}} &= \frac{\mathbf{X} - \mu_{\mathbf{X}}}{\sigma_{\mathbf{X}}}
\quad \rightarrow \quad \mathbf{X} = \sigma_{\mathbf{X}} \mathbf{\tilde{X}} + \mu_{\mathbf{X}} \\
\mathbf{\tilde{t}} &= \frac{\mathbf{t} - \mathbf{t}_{\text{min}}}{\mathbf{t}_{\text{max}} - \mathbf{t}_{\text{min}}} \quad \rightarrow \quad \mathbf{t} = \mathbf{t}_{\text{min}} + \mathbf{\tilde{t}} (\mathbf{t}_{\text{max}} - \mathbf{t}_{\text{min}})
\end{align}
$$

With these transformations, the ODE can be rewritten as:

$$
\begin{align}
\frac{d\mathbf{X}}{dt} &= f_{\theta} \left( \mathbf{X}, \mathbf{t} \right) \\
\rightarrow \frac{\sigma_{\mathbf{X}}}{\mathbf{t}_{\text{max}} - \mathbf{t}_{\text{min}}} \frac{d\mathbf{\tilde{X}}}{d{\mathbf{\tilde{t}}}} &= f_{\theta} \left( \sigma_{\mathbf{X}} \mathbf{\tilde{X}} + \mu_{\mathbf{X}}, \sigma_{\mathbf{t}} \mathbf{\tilde{t}} + \mu_{\mathbf{t}} \right) \\
\rightarrow \frac{d\mathbf{\tilde{X}}}{d{\mathbf{\tilde{t}}}} &= \frac{\mathbf{t}_{\text{max}} - \mathbf{t}_{\text{min}}}{\sigma_{\mathbf{X}}} f_{\theta} \left( \sigma_{\mathbf{X}} \mathbf{\tilde{X}} + \mu_{\mathbf{X}}, \sigma_{\mathbf{t}} \mathbf{\tilde{t}} + \mu_{\mathbf{t}} \right) \\
\rightarrow \frac{d\mathbf{\tilde{X}}}{d{\mathbf{\tilde{t}}}} &= \frac{\mathbf{t}_{\text{max}} - \mathbf{t}_{\text{min}}}{\sigma_{\mathbf{X}}} f_{\theta} \left(\mathbf{X}, \mathbf{t} \right) \\
\rightarrow \frac{d\mathbf{\tilde{X}}}{ d{\mathbf{\tilde{t}}} } &= \tilde{f}_{\theta} \left( \mathbf{\tilde{X}}, \mathbf{\tilde{t}} \right)
\end{align} \\
$$

and,

$$
\frac{dX}{dt} = \frac{\sigma_{\mathbf{X}}}{\mathbf{t}_{\text{max}} - \mathbf{t}_{\text{min}}}\frac{d\mathbf{\tilde{X}}}{d{\mathbf{\tilde{t}}}}
$$

From above equations,

- If the network $f_{\theta}$ is trained on the original data, then its output should be scaled by the factor ${(\mathbf{t}_{\text{max}} - \mathbf{t}_{\text{min}})}/{\sigma_{\mathbf{X}}}$ before passing it to the ODE solver. Then, the output of the ODE solver will be in the normalized space, and we can transform it back to the original space using the inverse of the normalization transformation.
- Otherwise, *even better*, if the network $f_{\theta}$ is trained on the normalized data, then it can be directly used in the ODE solver without any scaling, as it already represents the dynamics in the normalized space. Then, the output of the ODE solver will be in the normalized space, and we can transform it back to the original space using the inverse of the normalization transformation.

So, in summary, adopting the second approach:

1. **Normalize** the *input* (i.e., the initial state of the system), *time* and *target trajectory* (i.e., the observed states at different time steps) using standardization or any other normalization technique.
2. **ODESolver** takes the normalised input and time, along with the neural network $\tilde{f}_{\theta}$ (where we do not scale the output of the network), to compute the predicted states at different time steps in the normalized space.
3. **Train** the model with the loss computed in the normalized space.
4. During **inference**, we can pass the trained model along with given initial state and future time values in the same normalised scale (i.e., for example to zero mean and unit variance, if standardization is used during training), to the ODE solver to compute the predicted states in the normalized space. Finally, transform the predicted states back to the original space using the inverse of the normalization transformation.

<!-- 
## A simple example of a Neural ODE
### Breathing Ellipse Example

Let's build a model that learns velocity field that governs the motion of points on a 2D curve such that the overall shape follows a specific geometric evolution. In this example, we will use a simple ellipse that expands and contracts over time, simulating a breathing motion whose aspect ratio changes according to the following function:

$$
\beta(t) = \frac{1}{2}\left[(\beta_0 + \frac{1}{\beta_0}) + (\beta_0 - \frac{1}{\beta_0})\cos(\frac{\pi t}{T}) \right]; \quad t = 0, 1, ..., T \\
$$

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

$$
\mathcal{L}(\theta) = \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} \| X_i^{\theta}(t_k) - X_i^{\text{true}}(t_k) \|^2
$$

#### Appendix: Derivation of velocity field

$$
\dfrac{x^2}{a(t)^2} + \dfrac{y^2}{b(t)^2} = 1 \\
$$

Assuming the ellipse is centered at the origin, parametrized by the angle $\theta \in [0, 2\pi)$, the coordinates of points on the ellipse can be expressed as:

$$
x(t, \theta) = a(t) \cos{\theta} \\
y(t, \theta) = b(t) \sin{\theta}
$$

Where $a(t)$ and $b(t)$ are the semi-major and semi-minor axes of the ellipse at time $t$. The aspect ratio $\beta(t)$ is defined as:

$$
\beta(t) = \frac{a(t)}{b(t)} \\
\beta(0) = \beta_0 \\
\beta(T) = \frac{1}{\beta_0}
$$

As the area of the ellipse is conserved, we have:

$$
\begin{align*}
A &= \pi a(t) b(t) = \text{constant} \\
\Rightarrow a(t) b(t) &= A / \pi = \xi ^ 2 \\
\Rightarrow a(t) &= \xi \sqrt{\beta(t)} \\
\Rightarrow b(t) &= \frac{\xi}{\sqrt{\beta(t)}} \\
x(t, \theta) &= \xi \sqrt{\beta(t)} \cos{\theta} \\
y(t, \theta) &= \frac{\xi}{\sqrt{\beta(t)}} \sin{\theta} \\
\end{align*}
$$

Differentiating the coordinates with respect to time to get the velocity field:

$$
\begin{align*}
\frac{dx}{dt} &= \frac{d}{dt} (\xi \sqrt{\beta(t)} \cos{\theta}) \\
&= \xi \cos{\theta} \frac{d}{dt} \sqrt{\beta(t)} \\
&= \frac{\xi}{2} \cos{\theta} \frac{\beta'(t)}{\sqrt{\beta(t)}} \\
\Rightarrow \frac{dx}{dt} &= \frac{x}{2} \frac{\beta'(t)}{\beta(t)} \\
\end{align*}
$$

Similarly for $y$:

$$
\begin{align*}
\frac{dy}{dt} &= \frac{d}{dt} \left( \frac{\xi}{\sqrt{\beta(t)}} \sin{\theta} \right) \\
&= -\frac{\xi}{2} \sin{\theta} \frac{\beta'(t)}{\beta(t)\sqrt{\beta(t)}} \\
\Rightarrow \frac{dy}{dt} &= -\frac{y}{2} \frac{\beta'(t)}{\beta(t)}
\end{align*}
$$

Hence, the velocity field governing the motion of points on the curve can be expressed as:

$$
\begin{align*}
\frac{d}{dt} \begin{pmatrix} x \\ y \end{pmatrix} &= \alpha(t) \begin{pmatrix} x \\ -y \end{pmatrix} \\
&= \frac{\beta'(t)}{2\beta(t)} \begin{pmatrix} x \\ -y \end{pmatrix}
\end{align*}
$$

The neural ODE should learn this time-varying velocity field that governs the motion of points on the curve, allowing it to reconstruct the breathing ellipse dynamics from the observed data. -->

## Scope/Limitations of Neural ODEs

According to the original paper:

- **Minibatching**
- **Unique Solutions**
- **Manual tolerance Setting**
- **Errors in the reconstruction of forward trajectories**
