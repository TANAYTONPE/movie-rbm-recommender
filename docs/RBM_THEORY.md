# Mathematical Foundations of Restricted Boltzmann Machines (RBM)
## B.Tech CSE (AIML) — Deep Learning-II Academic Reference

---

## 1. Introduction to Energy-Based Models (EBMs)

A Restricted Boltzmann Machine (RBM) is an **Energy-Based Probabilistic Generative Neural Network**. Instead of calculating an output via simple forward matrix multiplications like a feedforward multi-layer perceptron (MLP), an RBM assigns an scalar "energy" value $E(v, h)$ to every joint configuration of visible variables $v$ and hidden variables $h$.

In statistical physics (specifically Boltzmann-Gibbs distributions), lower energy states correspond to configurations that are **more probable** and harmonious with the training data distribution:

$$P(v, h) = \frac{e^{-E(v, h)}}{Z}$$

Where:
- $v \in \{0, 1\}^N$: Vector of visible units (representing $N = 1,682$ movie ratings).
- $h \in \{0, 1\}^M$: Vector of hidden units (representing $M = 100$ latent preference factors).
- $Z$: The **partition function** (normalizing constant):
  $$Z = \sum_{v} \sum_{h} e^{-E(v, h)}$$

Calculating $Z$ directly requires summing over $2^{N + M} = 2^{1682 + 100} \approx 2^{1782}$ states, which is computationally intractable. This is why Contrastive Divergence and Markov Chain Monte Carlo (MCMC) approximations are required.

---

## 2. The Energy Function

The joint energy of the visible units $v$ and hidden units $h$ is defined as:

$$E(v, h) = - a^T v - b^T h - v^T W h$$

Expanded into scalar summation:

$$E(v, h) = - \sum_{i=1}^{N} a_i v_i - \sum_{j=1}^{M} b_j h_j - \sum_{i=1}^{N} \sum_{j=1}^{M} v_i W_{ij} h_j$$

### Symbol Reference Table

| Symbol | Mathematical Domain | Physical / Computational Meaning |
|---|---|---|
| $v_i$ | $\{0, 1\}$ | State of visible neuron $i$ ($1$ = liked movie $i$, $0$ = disliked/unliked). |
| $h_j$ | $\{0, 1\}$ | State of hidden neuron $j$ ($1$ = latent preference active, $0$ = inactive). |
| $W_{ij}$ | $\mathbb{R}$ | Weight coupling visible movie $i$ with latent preference $j$. |
| $a_i$ | $\mathbb{R}$ | Bias of visible unit $i$ (captures overall popularity of movie $i$). |
| $b_j$ | $\mathbb{R}$ | Bias of hidden unit $j$ (base activation propensity of latent factor $j$). |
| $\sigma(x)$ | $[0, 1]$ | Standard Sigmoid activation function $\frac{1}{1 + e^{-x}}$. |

---

## 3. The "Restricted" Architecture & Conditional Independence

A standard Boltzmann Machine allows connections between all pairs of neurons (including hidden-to-hidden and visible-to-visible). This causes intractable dependencies.

In a **Restricted** Boltzmann Machine, connections exist **strictly between visible and hidden layers** (bipartite graph). **There are no intra-layer connections.**

$$\text{Cov}(h_j, h_k \mid v) = 0 \quad \forall j \neq k$$
$$\text{Cov}(v_i, v_l \mid h) = 0 \quad \forall i \neq l$$

Because of this property:
1. Given visible states $v$, all hidden units $h_j$ are **conditionally independent**.
2. Given hidden states $h$, all visible units $v_i$ are **conditionally independent**.

---

## 4. Conditional Probabilities

### 4.1 Hidden Layer Conditional Probability $P(h_j = 1 \mid v)$

To compute the probability that hidden unit $h_j$ turns ON given visible vector $v$:

$$P(h_j = 1 \mid v) = \frac{P(h_j = 1, v)}{P(h_j = 0, v) + P(h_j = 1, v)}$$

Using the energy equation:

$$P(h_j = 1 \mid v) = \frac{e^{b_j + \sum_i v_i W_{ij}}}{1 + e^{b_j + \sum_i v_i W_{ij}}} = \sigma\left(b_j + \sum_{i=1}^{N} v_i W_{ij}\right)$$

In vector notation:

$$P(h = 1 \mid v) = \sigma(b + v W)$$

### 4.2 Visible Layer Conditional Probability $P(v_i = 1 \mid h)$

Symmetrically, to compute the reconstructed probability that visible unit $v_i$ is 1 given hidden vector $h$:

$$P(v_i = 1 \mid h) = \sigma\left(a_i + \sum_{j=1}^{M} h_j W_{ij}\right)$$

In vector notation:

$$P(v = 1 \mid h) = \sigma(a + h W^T)$$

---

## 5. Free Energy

The marginal probability of a visible vector $v$ is obtained by marginalizing out all hidden states $h$:

$$P(v) = \sum_{h} P(v, h) = \frac{1}{Z} \sum_{h} e^{-E(v, h)} = \frac{e^{-F(v)}}{Z}$$

Where $F(v)$ is the **Free Energy**:

$$F(v) = - a^T v - \sum_{j=1}^{M} \ln\left(1 + e^{b_j + \sum_i v_i W_{ij}}\right)$$

In PyTorch, to prevent numerical overflow from large positive activations, we compute $\ln(1 + e^x)$ using the numerically stable `torch.nn.functional.softplus(x)`:

$$F(v) = - a^T v - \sum_{j=1}^{M} \text{softplus}(b_j + v W_{:, j})$$

---

## 6. Contrastive Divergence Learning (CD-k)

### 6.1 The Maximum Likelihood Gradient

The objective of training is to maximize the log-likelihood of the observed training users $\mathcal{D}$:

$$\mathcal{L}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{v \in \mathcal{D}} \ln P(v)$$

The derivative of the log-likelihood with respect to connection weights $W_{ij}$ is:

$$\frac{\partial \ln P(v)}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$

Where:
- $\langle v_i h_j \rangle_{\text{data}}$: **Positive Phase expectation** — expectation under the conditional distribution $P(h \mid v)$ clamped on actual training data $v_0$.
- $\langle v_i h_j \rangle_{\text{model}}$: **Negative Phase expectation** — expectation under the model's equilibrium distribution $P(v, h)$ over all possible states.

### 6.2 The CD-k Approximation (Geoffrey Hinton, 2002)

Because running a Markov chain to complete equilibrium is too slow, Geoffrey Hinton proved that running Gibbs sampling for just $k$ steps (typically $k=1$) initialized at the data vector $v_0$ yields a remarkably accurate learning signal:

$$\Delta W_{ij} \approx \eta \left( v_{0, i} P(h_{0, j}=1 \mid v_0) - v_{k, i} P(h_{k, j}=1 \mid v_k) \right)$$

### 6.3 Complete Parameter Update Equations with Momentum and Weight Decay

$$\Delta W^{(t)} = \gamma \Delta W^{(t-1)} + \eta \left( \frac{1}{B} v_0^T P(h_0 \mid v_0) - \frac{1}{B} v_k^T P(h_k \mid v_k) - \lambda W \right)$$

$$\Delta a^{(t)} = \gamma \Delta a^{(t-1)} + \eta \left( \frac{1}{B} \sum_{u=1}^{B} (v_{0, u} - v_{k, u}) \right)$$

$$\Delta b^{(t)} = \gamma \Delta b^{(t-1)} + \eta \left( \frac{1}{B} \sum_{u=1}^{B} (P(h_{0, u} \mid v_{0, u}) - P(h_{k, u} \mid v_{k, u})) \right)$$

Where:
- $\eta$: Learning rate (e.g., $0.05$).
- $\gamma$: Momentum coefficient (e.g., $0.8$).
- $\lambda$: Weight decay coefficient (e.g., $0.0001$).
- $B$: Batch size (e.g., $64$).

---

## 7. Handling Missing Ratings: Conditional Masking

In MovieLens collaborative filtering, over 93% of movie ratings are missing for any given user.

If unrated movies were treated as $0$ without masking, the model would incorrectly learn that the user *actively disliked* those movies.

To solve this:
1. Define a binary observation mask $M \in \{0, 1\}^N$, where:
   $$M_i = \begin{cases} 1 & \text{if user rated movie } i \\ 0 & \text{if movie } i \text{ is unrated} \end{cases}$$
2. During the negative phase of Contrastive Divergence, reconstructed negative sample $v_k$ is multiplied element-wise by the mask $M$:
   $$v_k^{\text{masked}} = v_k \odot M$$
3. Gradients and reconstruction loss are computed **strictly over rated items**:
   $$\mathcal{L}_{\text{recon}} = \frac{1}{\sum_i M_i} \sum_{i=1}^{N} M_i (v_{0, i} - P(v_{k, i}=1 \mid h))^2$$

---

## 8. Summary Comparison of Core Concepts

| Concept | Purpose | Input $\to$ Output | Mathematical Operation |
|---|---|---|---|
| **Forward Pass (Positive Phase)** | Infer latent user preference profile | $v_0 \to P(h_0 \mid v_0)$ | $\sigma(b + v_0 W)$ |
| **Gibbs Sampling** | Explore model's internal distribution | $h_{t-1} \to v_t \to h_t$ | Alternating Bernoulli draws from $\sigma(a + h W^T)$ and $\sigma(b + v W)$ |
| **Contrastive Divergence (CD-1)** | Efficient parameter update rule | $v_0, v_1, h_0, h_1 \to \Delta W$ | $\eta \cdot (\langle v_0 h_0 \rangle - \langle v_1 h_1 \rangle)$ |
| **Reconstruction** | Estimate preference for unseen movies | $v_{\text{user}} \to P(v_{\text{unseen}}=1 \mid h)$ | $P(v = 1 \mid \sigma(b + v_{\text{user}} W))$ |
| **Recommendation Ranking** | Select Top-K unseen movies | $P(v \mid h) \to \text{Top-K list}$ | $\text{argsort}(P(v \mid h)) \setminus \text{rated items}$ |

