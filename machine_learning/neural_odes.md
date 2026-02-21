# Neural Ordinary Differential Equations (Neural ODEs)

Neural Ordinary Differential Equations (Neural ODEs) introduced by [Chen et al.](https://arxiv.org/abs/1806.07366) in 2018, model the evolution of a system's state over time using a continuous function parameterized by a neural network.

In a standard neural network, the hidden state is represented as a discrete sequence of layers as shown below. These updates are like Euler's discretization of a continuous transformation.

$$
\bm{h}_{t+1} = \bm{h}_t + f(\bm{h}_t, \theta_t); \;\; t = 0, 1, ..., T
$$

In the limit as the step size goes to zero, this can be represented as a continuous transformation of the hidden state:

$$
\frac{d\bm{h}}{dt} = f(h(t), t, \theta)
$$

> $\bm{h}(0)$ is the input layer, and $\bm{h}(T)$ is the output layer.

## Advantages of Neural ODEs

According to the original paper:

- **Memory Efficiency**: Neural ODEs can be more memory efficient than traditional deep networks, as they do not require storing intermediate activations for backpropagation. Instead, they use the adjoint method to compute gradients, there by having a constant memory cost as a function of the network depth.
- **Adaptive Computation**: Depending on the complexity of the input, and the desired accuracy, Neural ODEs can adaptively choose the number of function evaluations needed to compute the output, which can lead to faster inference for simpler inputs.
- **Continuous-Time Modeling**: Neural ODEs naturally model continuous-time processes, containing observations at *irregular* time intervals.
- **Scalable and invertible normalizing flows**

## Scope/Limitations of Neural ODEs

According to the original paper:

- **Minibatching**
- **Unique Solutions**
- **Manual tolerance Setting**
- **Errors in the reconstruction of forward trajectories**

## Standard NN $\leftrightarrow$ Neural ODEs

| Feature                | Standard NN                          | Neural ODEs                                                                         |
| ---------------------- | ------------------------------------ | ----------------------------------------------------------------------------------- |
| Depth                  | Number of layers                     | Integration time, number of function evaluations determined by the solver tolerance |
| Continuity             | Discrete, $h_{t+1} = f(h_t, \theta)$ | Continuous, $\frac{dh}{dt} = f(h(t), t, \theta)$                                    |
| Gradient               | Backpropagation through layers       | Adjoint method for gradient computation                                             |
| Step size              | Fixed                                | Adaptive                                                                            |
| Forward pass           | through layers                       | through ODE solver, Ex: using `torchdiffeq.odeint`                                  |
| Memory cost            | $O(L)$                               | $O(1)$, just store the current state and the end points of the integration interval |
| Function $\mathcal{N}$ | represents transformation            | $\mathcal{N}$ represents a veclocity field that defines the dynamics of the system  |

## A simple example of a Neural ODE

### Breathing Ellipse Example

Let's build a model that learns velocity field that governs the motion of points on a 2D curve such that the overall shape follows a specific geometric evolution. In this example, we will use a simple ellipse that expands and contracts over time, simulating a breathing motion.

#### Assumptions

- Initial state ($t=0$): An ellipse with aspect ratio $\beta_0$
- Dynamics: The aspect ratio $\beta(t)$ changes over time according to a simple sinusoidal function, simulating the breathing motion.
- Intermediate state: At some $t_c$, the aspect ratio becomes unity, resulting in a circle.
- Final state ($t=T$): The ellipse reaches opposite aspect ratio $\beta_T = 1/\beta_0$.
- Each point on the curve must be mapped to a new position at each time step, following the velocity field defined by the Neural ODE.

#### Mathematical Formulation

Todo

#### Expected Results

- Non-autonomous vector field
- Conservation of topology (no crossing of trajectories)
- Resolution invariance (the same curve should be obtained regardless of the number of points sampled on the curve)
