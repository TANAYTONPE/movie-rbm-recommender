# Beginner-Friendly Project Explanation
## Movie Recommendation System Using Restricted Boltzmann Machine (RBM)

---

## 1. Project Analogy: The Coffee Barista & Movie Tastes

Imagine you walk into a cafe you have never visited before. You tell the barista:
- *"I like dark roast Colombian espresso."*
- *"I enjoy a touch of cinnamon and oat milk."*
- *"I dislike overly sugary caramel syrups."*

A great barista does not just memorize those 3 specific drinks. Instead, the barista forms a **mental concept of your hidden taste profile**:
- **Hidden Concept 1:** Enjoys bold, aromatic, roasted flavors.
- **Hidden Concept 2:** Prefers creamy non-dairy texture.
- **Hidden Concept 3:** Low tolerance for artificial sweetness.

Using these **hidden concepts**, the barista recommends a drink you have never even heard of: *"You will love our Nitro Cold Brew with Spiced Almond Milk!"*

### How This Maps to our RBM Movie Recommender

| Cafe Barista Analogy | RBM Recommendation System |
|---|---|
| The drinks you told the barista you like or dislike | **Visible Layer ($v$):** Movies you rated 1–5 stars. |
| The barista's mental concepts (bold, creamy, nutty) | **Hidden Layer ($h$):** 100 latent preference neurons (sci-fi pacing, cerebral mysteries, dry comedy). |
| The barista's intuition connecting drinks to concepts | **Weight Matrix ($W$):** Connection strengths between movies and latent preference factors. |
| The new drink the barista recommends | **Reconstruction / Top-K Recommendations:** Movies with highest reconstructed probability. |

---

## 2. Deep Dive Into Every Major Component

Below is a breakdown of every fundamental concept in the project. For each concept, you will find:
1. What it is.
2. Why it is used.
3. How it works.
4. A real-world analogy.
5. A small numeric example.
6. How to explain it during a viva exam.

---

### Component 1: Visible Layer ($v$)

1. **What it is:** A vector of 1,682 neurons where each neuron represents one movie in the MovieLens catalog.
2. **Why it is used:** It provides the input/output interface for user interactions.
3. **How it works:** When a user rates movies, rated movies with rating $\ge 3$ are set to $1.0$ (liked), movies rated $< 3$ are set to $0.0$ (disliked), and an observation mask flags which movies were actually rated.
4. **Real-world analogy:** The row of switches on an airplane dashboard. Each switch corresponds to one specific feature (lights, air conditioning, oxygen mask).
5. **Small example:**
   If a user rated *Toy Story* ($5$ stars) and *GoldenEye* ($2$ stars), their input vector is:
   $$v = [1, 0, 0, 0, \dots, 0]$$
   $$\text{mask} = [1, 1, 0, 0, \dots, 0]$$
6. **Viva explanation:**
   > *"Visible neurons represent the observable data—in our case, the 1,682 movies in the MovieLens 100K dataset. They receive the user's ratings and later display the reconstructed preferences."*

---

### Component 2: Hidden Layer ($h$)

1. **What it is:** A layer of 100 stochastic neurons representing latent (hidden) user tastes.
2. **Why it is used:** Humans do not rate movies purely at random; our choices are driven by underlying preferences such as pacing, director style, tone, and humor. The hidden layer discovers these factors without human supervision.
3. **How it works:** It aggregates signals from all visible units through connection weights and computes activation probabilities via the Sigmoid function:
   $$P(h_j = 1 \mid v) = \sigma\left(b_j + \sum_i v_i W_{ij}\right)$$
4. **Real-world analogy:** Musical genres. Nobody explicitly told you the exact mathematical formula for "Jazz", but your brain recognizes the underlying patterns of syncopated rhythm and brass instruments.
5. **Small example:**
   Suppose neuron $h_{12}$ detects "Epic Space Opera". If *Star Wars* ($v_{50}=1$) is active and has weight $W_{50, 12} = +2.4$, the sum exceeds the threshold, turning neuron $h_{12}$ ON with probability $92\%$.
6. **Viva explanation:**
   > *"Hidden neurons represent latent variables. They are not directly observed in the data. Instead, they capture abstract patterns and user taste archetypes in an unsupervised fashion."*

---

### Component 3: Weight Matrix ($W$) and Biases ($a, b$)

1. **What it is:**
   - $W \in \mathbb{R}^{1682 \times 100}$: Connection strengths between visible and hidden units.
   - $a \in \mathbb{R}^{1682}$: Visible biases (baseline popularity of each movie).
   - $b \in \mathbb{R}^{100}$: Hidden biases (base tendency of latent factors to turn ON).
2. **Why it is used:** Weights store the collaborative intelligence learned from all 943 users and 100,000 ratings.
3. **How it works:** Positive weights indicate that liking movie $i$ strongly implies latent factor $j$ is active. Negative weights indicate an inverse relationship.
4. **Real-world analogy:** Gravitational pull between stars in a solar system. Stronger weights exert a stronger pull on connected neurons.
5. **Small example:**
   $W_{\text{Star Wars}, \text{Sci-Fi}} = +3.1$ (strong positive affinity).
   $W_{\text{Barbie}, \text{Dark Horror}} = -2.5$ (strong negative affinity).
6. **Viva explanation:**
   > *"The weight matrix $W$ captures the bilateral correlation between movies and latent preferences. The biases $a$ and $b$ act as baseline activation thresholds independent of inputs."*

---

### Component 4: Gibbs Sampling

1. **What it is:** A Markov Chain Monte Carlo (MCMC) sampling technique where neurons in one layer are sampled alternately conditioned on the current states of the other layer.
2. **Why it is used:** Because the exact joint distribution $P(v, h)$ requires summing over $2^{1782}$ states (impossible to compute directly), Gibbs sampling allows us to approximate this distribution efficiently.
3. **How it works:**
   - Step 1: Given $v_0$, sample hidden states $h_0 \sim P(h \mid v_0)$.
   - Step 2: Given $h_0$, sample visible states $v_1 \sim P(v \mid h_0)$.
   - Step 3: Given $v_1$, sample hidden states $h_1 \sim P(h \mid v_1)$.
   - Repeat for $k$ iterations.
4. **Real-world analogy:** A tennis rally between two players (visible player and hidden player). Player A hits the ball to Player B, who returns it, refining their positions on the court with each bounce.
5. **Small example:**
   Start with input $v_0 = [1, 0, 1]$.
   Forward pass samples $h_0 = [1, 1]$.
   Backward pass samples $v_1 = [1, 1, 1]$ (the middle movie turned ON because both hidden units supported it!).
6. **Viva explanation:**
   > *"Gibbs sampling is an MCMC method where we alternate between sampling the hidden units given visible units and sampling visible units given hidden units. Thanks to the restricted bipartite structure, units within a layer are conditionally independent, allowing all neurons in a layer to be sampled in parallel."*

---

### Component 5: Contrastive Divergence (CD-k)

1. **What it is:** The efficient learning algorithm introduced by Geoffrey Hinton in 2002 to train Restricted Boltzmann Machines.
2. **Why it is used:** Traditional maximum likelihood requires running the Gibbs sampling chain until it reaches thermal equilibrium (which takes hundreds of steps). CD-1 stops after just **1 step** and still provides an accurate approximation of the gradient.
3. **How it works:**
   - **Positive Phase:** Clamp the visible units to actual training data $v_0$. Compute hidden activations $P(h_0 \mid v_0)$. This represents what the real data looks like ($\langle v_0 h_0 \rangle$).
   - **Negative Phase:** Reconstruct $v_1$ from $h_0$, then compute $P(h_1 \mid v_1)$. This represents what the model currently imagines ($\langle v_1 h_1 \rangle$).
   - **Update Rule:**
     $$\Delta W = \eta \cdot \left( v_0^T h_0 - v_1^T h_1 \right)$$
     If data has stronger co-occurrence than the model's reconstruction, increase $W$. If the model over-predicts, decrease $W$.
4. **Real-world analogy:** Practicing a song on the piano with a teacher.
   - Positive phase: You listen to the teacher play the sheet music correctly ($v_0$).
   - Negative phase: You try playing it from memory ($v_1$).
   - Learning step: You adjust your finger placement based on the difference between the teacher's version and your version.
5. **Small example:**
   $v_{0, i} \cdot h_{0, j} = 1.0 \times 0.9 = 0.90$.
   $v_{1, i} \cdot h_{1, j} = 0.0 \times 0.4 = 0.00$.
   $\Delta W_{ij} = 0.05 \times (0.90 - 0.00) = +0.045$ (weight increases).
6. **Viva explanation:**
   > *"Contrastive Divergence (CD-1) approximates the gradient of log-likelihood using a 1-step Gibbs chain. It compares the data-dependent expectation in the positive phase with the model's fantasy expectation in the negative phase to update weights."*

---

### Component 6: Reconstruction and Top-K Recommendation

1. **What it is:** The inference mechanism that takes a new user's ratings and outputs personalized movie suggestions.
2. **Why it is used:** The core deliverable of the project is helping a user discover new movies they will enjoy.
3. **How it works:**
   - 1. Take user's known ratings and feed them to the visible layer $v$.
   - 2. Compute hidden layer activations: $h = \sigma(b + v W)$.
   - 3. Reconstruct all visible units: $P(v_{\text{all}} = 1 \mid h) = \sigma(a + h W^T)$.
   - 4. Set the score of all already-rated movies to negative infinity so they are excluded.
   - 5. Sort remaining unseen movies descending by probability score.
   - 6. Return the top 10 movies along with their genres and explanations.
4. **Real-world analogy:** A doctor diagnosing your symptoms. You tell the doctor 3 symptoms ($v$). The doctor identifies the underlying condition ($h$). From that condition, the doctor predicts other symptoms you might soon experience ($P(v \mid h)$).
5. **Small example:**
   User likes *Toy Story* ($v_1 = 1$).
   Hidden unit $h_3$ (Animated Family Comedy) activates with probability $0.95$.
   Visible unit for *A Close Shave* ($v_{408}$) reconstructs with probability $0.98$ and predicted stars of $4.9 / 5.0$.
6. **Viva explanation:**
   > *"During inference, we pass the user's vector of known ratings through the weights to infer their latent taste vector. Then, we reconstruct the probabilities for all 1,682 visible units, filter out movies the user has already rated, and rank the unseen movies to generate the top recommendations."*

---

## 3. How to Explain the Entire Project in 2 Minutes (Viva Elevator Pitch)

> *"Good morning, Professors. My project is a **Movie Recommendation System using a Restricted Boltzmann Machine (RBM)** trained on the MovieLens 100K benchmark dataset.*
>
> *Traditional collaborative filtering methods like user-based KNN or matrix factorization are purely heuristic or linear. In contrast, an RBM is an **energy-based stochastic neural network** capable of learning non-linear probability distributions over human preferences.*
>
> *Our architecture has **1,682 visible neurons** representing movies and **100 hidden neurons** representing latent viewer taste factors. It is a **bipartite graph**, meaning there are no connections between neurons in the same layer. This restriction guarantees conditional independence and allows parallel Gibbs sampling.*
>
> *We train the RBM using **Contrastive Divergence (CD-1)** with momentum and conditional masking to account for the 94% sparsity of the rating matrix. On a held-out test set of 20,000 ratings, the model achieves a **Test RMSE of 0.3734** and a **Hit Rate@20 of 30.28%**.*
>
> *Finally, we deployed the model with an interactive **Streamlit web application** featuring movie search, genre filtering, real-time Gibbs inference, dynamic rating cards, and visual weight heatmaps."*

