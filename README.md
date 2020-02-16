# normalizing-flows

Implementations of normalizing flows for variational inference using Python (3.6+) and Tensorflow (2.0+).

## Installation

    pip install git+https://github.com/bgroenks96/normalizing-flows
    
Please be aware that this library is currently under **heavy development** and all APIs are subject to change.

## Introduction and background

Bayesian methods are valuable tools for statistical inference thanks to their ability to treat model parameters as drawn from their own distribution rather than being fixed estimates. This allows for incorporation of known information in the form of a *prior* distribution, as well as calculation of parameter uncertainty. Unfortunately, computational cost is commonly cited as a significant limitation of Bayesian inference. This typically results from the intractability of the integrations required to compute the marginal distribution p(X) for the posterior p(Z | X). Markov Chain Monte Carlo algorithms are often used to approximate the posterior through random walks but often suffer from long convergence times, especially in very high dimensional spaces. Variational inference attempts to instead approximate p(Z | X) by constructing a lower bound on the log marginal likelihood, p(X). This is done using a known, tractable *variational* distribution q(Z). The goal is then to minimize the dissimilarity between $(Z | X) and the chosen variational distribution q(Z).

### Variational lower bound

The variational or evidential lower bound (ELBO) is defined as

$$
\begin{split}
    \log p_\theta(X) &\geq \log p_\theta(X) - \text{KL}\big [q(Z|X) || p_\theta(Z|X)\big ] \\
                     & = \underset{q}{\mathbb{E}}\big [ \log p_\theta(X|Z)\big ] - \text{KL}\big [ q(Z|X) || p(Z) \big]
\end{split}
$$

where theta is the model parameters and KL is the Kullback-Leibler divergence between distributions p and q. Maximizing the first term is equivalent to minimizing the log likelihood of the data given theta and Z, while minimizing the KL term attempts to bring the learned posterior distribution q(Z|X) closer to the variational prior p(Z).

### Normalizing flows

For highly complex posterior distributions, simple variational distributions like the spherical Gaussian will result in a loose lower bound (large divergence between $\log P(X)$ and ELBO). Thus, there is a need for more flexible class of distributions for p(Z).

Let $f: \mathbb{R}^d \rightarrow \mathbb{R}^d$ be a smooth, invertible mapping over a probability density p(z) such that $z = f^{-1}(f(z))$. Then it follows that the resulting random variable $z' = f(z)$ has a density given by:
$$
    p(z') = p(z)|\det{\mathbb{J}_{f^{-1}}(z')}| = p(z)|\det{\mathbb{J}_f(z)}|^{-1}
$$

where the last equality follows from the inverse function theorem. In general, the cost of computing the Jacobian is $\mathrm{O}(n^3)$. It is possible, however, to construct transformations where the Jacobian can be more efficiently computed.

We can construct arbitrarily complex densities by chaining a series of transformations f_i and successively applying the aforementioned transformation:
$$
    z_k = f_k \circ f_{k-1} \circ \cdots \circ f_1(z_0)
$$
Then the log density is given by:
$$
    \log p_k(z_k) = \log p_0(z_0) - \sum_{i=1}^k{\log |\det{\mathbb{J}_{f_{i}}(z_{i-1})}}|
$$
The path formed by random variables $z_0, z_1, \dots, z_k$ is called a *flow* and the path formed by their successive distributions $p_0, p_1, \dots, p_k$ a *normalizing flow* (Rezende et al. 2015).
