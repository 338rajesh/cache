# Monte Carlo Methods for Uncertainty Quantification

Monte Carlo (MC) methods are a class of computational algorithms that rely on repeated random sampling to obtain numerical results. They are widely used in uncertainty quantification to estimate the expected value, variance, and other statistical properties of a model's output when the input parameters are uncertain. The basic idea is to:

- Generate random samples in the input space
- Run (deterministic) simulations or calculations for each of these random samples
- Summarize the results using statistics (e.g., mean, variance, confidence intervals) to quantify the uncertainty in the output

## Law of Large Numbers

MC methods rest on the principle of the law of large numbers, which states that as the number of samples increases, the sample **average converges to the expected value**. This means that with enough samples, we can get an accurate estimate of the uncertainty in our model's predictions.

- Strong Law of Large Numbers: The sample average converges to the expected value almost surely (with probability 1).

```math
Pr(\lim_{n \to \infty} \bar{X}_n = \mu) = 1 \quad \text{almost surely}
```

- Weak Law of Large Numbers: The probability of *the absolute difference between the sample average and the expected* value being *smaller than any small positive value $\epsilon$* approaches 1 as the number of samples goes to infinity.

```math
\lim_{n \to \infty} Pr(|\bar{X}_n - \mu| < \epsilon) = 1 \quad \text{for a small value } \epsilon > 0
```

Let $X_1, X_2, ..., X_n$ be a sequence of *independent and identically distributed (i.i.d.)* random variables with expected value $\mu$ and variance $\sigma^2$. According to LLN, as $n$ increases,  the sample average $\bar{X}_n$ converges to $\mu$.

```math
\bar{X}_n = \frac{1}{n} \sum_{i=1}^{n} X_i
```

Variance of the sample average can be used to understand the convergence rate. The following formula shows that the variance of the sample average decreases as the number of samples increases:

```math
Var(\bar{X}_n) = Var(\frac{1}{n} \sum_{i=1}^{n} X_i) = \frac{1}{n^2} \sum_{i=1}^{n} Var(X_i) = \frac{\sigma^2}{n}
```

## Monte Carlo Simulation

From law of large numbers:

```math
\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_{X} (x) dx \approx \frac{1}{N} \sum_{i=1}^{N} X_i
```

Where $X_i$ are i.i.d. samples drawn from the distribution of $X$. The accuracy of this approximation improves as $N$ increases, and the variance of the estimate decreases as $\sigma^2 / N$.

**Monte Carlo simulation** is nothing but the process of simulating a large number of realizations of a random variable and then evaluating its average value to estimate the expected value.

Here, the random variable $X$ can be function of another random variable or a vector of random variables. Let $g$ is a function that maps a random variable $X$ to another random variable $Y$, i.e., $Y = g(X)$. When $g$ takes a simple analytical form and invertible, we can derive the distribution of $Y$ from the distribution of $X$ using methods like change of variables. However, when $g$ is complex or non-invertible, it becomes difficult to derive the distribution of $Y$ analytically. In such cases, we can use Monte Carlo simulation to estimate the expected value of $Y$.

```math
\mathbb{E}[Y] = \mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) f_{X} (x) dx \approx \frac{1}{N} \sum_{i=1}^{N} g(X_i)
```

Where $X_i$ are i.i.d. samples drawn from the distribution of $X$. The accuracy of this approximation improves as $N$ increases, and the variance of the estimate decreases as $\sigma^2 / N$, where $\sigma^2$ is the variance of $g(X)$.

## Refined Standard steps

For any functionor model $g: \mathbb{R}^n \rightarrow Y$, that takes random variables as input with known probability distribution, one can estimate the expected value of the output distribution using the following Monte Carlo steps:

- Generate $N$ i.i.d. samples $\boldsymbol{x}_1, \boldsymbol{x}_2, ..., \boldsymbol{x}_N$ from the known joint density function $f_{\boldsymbol{X}}(\boldsymbol{x})$ of the input random vector $\boldsymbol{X}$.
- Function/model evaluation: For each sample $\boldsymbol{x}_i$, compute the corresponding output $y_i = g(\boldsymbol{x}_i)$. This step involves running the deterministic model or function for each of the generated samples.
- Apply Monte Carlo averaging to estimate the expected value of the output distribution:

```math
\mathbb{E}[Y] = \int_{-\infty}^{\infty} y f_{Y} (y) dy \approx \frac{1}{N} \sum_{i=1}^{N} g(\boldsymbol{x}_i)
```
