# Residual network block

Resnet block adds a skip connection to the output of a sequence of neural network layers, designed for addressing the vanishing gradient problem in deep neural networks.

The Resnet block can be represented mathematically as:

$$
y = F(x) + x
$$

Where $F(x)$ is the output of the sequence of layers, and $x$ is the input to those layers. If there is a need to learn a more complex function, $F(x)$ will capture that complexity. If not, $F(x)$ can be zero, and the output will simply be the input $x$, allowing for an identity mapping.

Here, the word "residual" refers to the fact that the sequence of layers is learning the residual function $F(x)$, which is the difference between the desired output and the input. This allows the network to learn more effectively, as it can focus on learning the residuals rather than trying to learn the entire mapping from input to output.

The gradient of the output with respect to the input can be expressed as:
$$
\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + 1
$$

In some cases, the function $F(x)$ may be very small, leading to very small gradients. Here, however the `+1` term ensures becomes the bridge that allows the gradient to flow through the network, even when $F(x)$ is small. This is one of the key reasons why Resnet blocks help mitigate the vanishing gradient problem in deep neural networks.
Allowing smaller changes with $F(x)$, while still maintaining a strong gradient flow through the skip connection, enables feature refinement and learning in deeper networks.
