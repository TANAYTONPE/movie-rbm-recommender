# Comprehensive Viva Voce Questions & Answers
## Subject: Deep Learning-II (B.Tech CSE AIML)
### Project: Movie Recommendation System Using Restricted Boltzmann Machine (RBM)

---

### Category 1: Deep Learning & Energy-Based Models

#### Q1. What is Deep Learning?
**Ans:** Deep Learning is a subfield of Machine Learning based on artificial neural networks with multiple representation layers, capable of automatically learning hierarchical representations from raw data without manual feature engineering.

#### Q2. What is an Energy-Based Model (EBM)?
**Ans:** An EBM is a probabilistic framework that associates a scalar energy value $E(x)$ to configurations of variables. Desirable, highly probable configurations are assigned low energy, while unlikely configurations are assigned high energy.

#### Q3. What is a Boltzmann Machine?
**Ans:** A Boltzmann Machine is an undirected recurrent neural network with stochastic binary units governed by a Boltzmann probability distribution derived from statistical physics.

#### Q4. Why is a standard Boltzmann Machine rarely used in practice?
**Ans:** Because neurons within the visible and hidden layers are all mutually interconnected. This full connectivity causes intractable dependencies, making exact inference and learning exponentially complex ($O(2^N)$).

#### Q5. What does the term "Restricted" mean in Restricted Boltzmann Machine?
**Ans:** It means that **no connections are permitted between neurons in the same layer** (no visible-to-visible or hidden-to-hidden links). The graph is strictly bipartite.

#### Q6. What is the main mathematical advantage of the "Restricted" structure?
**Ans:** Conditional independence. Given the visible layer, all hidden units can be computed independently and in parallel; given the hidden layer, all visible units can be computed independently and in parallel.

---

### Category 2: RBM Architecture & Components

#### Q7. What are visible neurons in this project?
**Ans:** Visible neurons ($N = 1,682$) represent the observable features—in this case, all movies in the MovieLens catalog. They receive user ratings and output reconstructed recommendations.

#### Q8. What are hidden neurons in this project?
**Ans:** Hidden neurons ($M = 100$) represent unobserved latent factors or taste profiles (such as preference for high-budget sci-fi, dark psychological thrillers, or family animations).

#### Q9. What is a latent variable?
**Ans:** A latent variable is a hidden, unobserved variable inferred indirectly from observable data that explains underlying patterns or correlations among visible variables.

#### Q10. What is the role of the weight matrix $W$?
**Ans:** The weight matrix $W \in \mathbb{R}^{1682 \times 100}$ couples the visible movie units with the latent preference units, encoding the bilateral correlation learned across all users.

#### Q11. What is the role of the visible bias $a$?
**Ans:** The visible bias vector $a \in \mathbb{R}^{1682}$ captures the baseline popularity of each movie independent of user preferences (e.g., blockbusters have higher baseline bias).

#### Q12. What is the role of the hidden bias $b$?
**Ans:** The hidden bias vector $b \in \mathbb{R}^{100}$ controls the baseline activation threshold of each latent factor.

#### Q13. Which activation function is used in RBM and why?
**Ans:** The **Sigmoid function** $\sigma(z) = \frac{1}{1 + e^{-z}}$. It maps real-valued linear inputs into a valid probability range $[0, 1]$ suitable for Bernoulli stochastic sampling.

#### Q14. Are RBM neurons deterministic or stochastic?
**Ans:** They are **stochastic**. The Sigmoid outputs an activation probability, and the neuron's binary state ($0$ or $1$) is drawn via a Bernoulli random trial: $h_j \sim \text{Bernoulli}(P(h_j=1 \mid v))$.

---

### Category 3: Gibbs Sampling & Contrastive Divergence

#### Q15. What is Gibbs Sampling?
**Ans:** Gibbs Sampling is a Markov Chain Monte Carlo (MCMC) algorithm where each variable is sampled conditioned on the current values of all other variables. In an RBM, we alternate between sampling hidden units given visible units ($v \to h$) and visible units given hidden units ($h \to v$).

#### Q16. Why do we need Gibbs Sampling in RBM?
**Ans:** Because the partition function $Z$ involves summing over $2^{1782}$ states, making exact expectation calculation impossible. Gibbs sampling allows us to generate representative samples from the model's joint distribution.

#### Q17. What is Contrastive Divergence (CD)?
**Ans:** Contrastive Divergence is an approximation algorithm introduced by Geoffrey Hinton in 2002 that replaces the infinitely long Gibbs chain with just $k$ steps of alternating Gibbs sampling initialized at the training data $v_0$.

#### Q18. What is the difference between CD-1 and CD-k?
**Ans:** CD-1 runs Gibbs sampling for exactly 1 step ($v_0 \to h_0 \to v_1 \to h_1$). CD-k runs Gibbs sampling for $k$ steps ($k > 1$), offering higher asymptotic fidelity at the cost of more computation.

#### Q19. What is the Positive Phase in CD?
**Ans:** The positive phase clamps the visible units to actual training data ($v_0$) and samples hidden states ($h_0$). It calculates the data-dependent correlations $\langle v_0 h_0 \rangle$.

#### Q20. What is the Negative Phase in CD?
**Ans:** The negative phase generates reconstructed visible states ($v_k$) from hidden states and then samples $h_k$. It calculates the model's internal "fantasy" correlations $\langle v_k h_k \rangle$.

#### Q21. What is the parameter update equation for weights in CD-1?
**Ans:**
$$\Delta W = \eta \left( \frac{1}{B} v_0^T h_0 - \frac{1}{B} v_1^T h_1 \right) - \lambda W$$
where $\eta$ is learning rate, $B$ is batch size, and $\lambda$ is weight decay.

#### Q22. What does it mean when the positive correlation equals the negative correlation?
**Ans:** It means the model's reconstructed distribution perfectly matches the training data distribution, so gradients become zero and the model has converged.

#### Q23. What is Free Energy in an RBM?
**Ans:** Free Energy $F(v)$ is the energy of a visible configuration after integrating out (marginalizing) all hidden states: $F(v) = -a^T v - \sum_j \ln(1 + e^{b_j + v W_{:, j}})$.

---

### Category 4: Dataset & Collaborative Filtering

#### Q24. Which dataset is used in this project?
**Ans:** The **MovieLens 100K** benchmark dataset collected by the GroupLens Research Project at the University of Minnesota.

#### Q25. What are the key statistics of the MovieLens 100K dataset?
**Ans:** 100,000 ratings (scale 1 to 5), 943 unique users, 1,682 unique movies, spanning 19 genres.

#### Q26. What is the sparsity of the MovieLens 100K dataset?
**Ans:** Sparsity is approximately **93.7%** (calculated as $1 - \frac{100,000}{943 \times 1682} = 93.7\%$).

#### Q27. What is a sparse matrix?
**Ans:** A matrix in which the vast majority of elements are unobserved or zero. In recommendation systems, no single user has watched even 10% of the entire movie catalog.

#### Q28. How does this project handle missing (unrated) movies during training?
**Ans:** Using **conditional masking**. A binary mask $M_{u, i}$ tracks whether user $u$ rated movie $i$. During Gibbs sampling and gradient computation, unrated items are masked out so the model does not treat unrated movies as disliked.

#### Q29. How are 1–5 star ratings converted for the binary RBM?
**Ans:** Ratings $\ge 3.0$ are converted to $1$ (liked/favorable), while ratings $< 3.0$ are converted to $0$ (disliked/unfavorable).

#### Q30. How is the RBM output converted into a 5-star rating for the user?
**Ans:** Using the linear mapping formula:
$$\text{Star Rating} = 1.0 + 4.0 \times P(v_i = 1 \mid h)$$
For instance, a probability of $0.90$ maps to $1.0 + 4.0(0.90) = 4.6$ stars.

---

### Category 5: Recommendation Logic & Evaluation

#### Q31. How does the recommendation inference work step-by-step?
**Ans:**
1. The user's ratings are converted into a visible vector $v$.
2. The forward pass computes hidden activations $P(h \mid v)$.
3. The backward pass reconstructs visible probabilities $P(v \mid h)$ for all 1,682 movies.
4. Already-rated movies are filtered out.
5. The remaining unseen movies are ranked descending by score, and the top 10 are returned.

#### Q32. Why must already-rated movies be excluded from recommendations?
**Ans:** A recommender must assist in **discovery** of new, unseen content. Recommending a movie the user already watched provides zero discovery utility.

#### Q33. What is the Cold-Start Problem?
**Ans:** The cold-start problem occurs when a new user or new movie has zero interaction history, making collaborative filtering inference impossible. In our project, we handle cold-start users with a popularity-based baseline filter.

#### Q34. What is Reconstruction Error?
**Ans:** The difference between the user's actual ratings and the model's reconstructed ratings on held-out test data, measured using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

#### Q35. What is Precision@K?
**Ans:** The proportion of recommended items in the top-$K$ list that are actually relevant to the user in the test set: $\text{Precision@K} = \frac{|\text{Top-K} \cap \text{Test Relevant}|}{K}$.

#### Q36. What is Recall@K?
**Ans:** The proportion of all relevant items in the test set that were successfully retrieved in the top-$K$ list: $\text{Recall@K} = \frac{|\text{Top-K} \cap \text{Test Relevant}|}{|\text{Test Relevant}|}$.

#### Q37. What is Hit Rate@K?
**Ans:** The percentage of users who have **at least one** relevant recommendation in their top-$K$ list.

#### Q38. What real results did your model achieve on the test set?
**Ans:**
- **Test RMSE:** $0.3734$
- **Test MAE:** $0.2624$
- **Hit Rate@20:** $30.28\%$
- **Precision@10:** $3.44\%$
- **MRR@10:** $0.1105$

---

### Category 6: Comparisons & Theoretical Distinctions

#### Q39. What is the difference between an RBM and an Autoencoder?
**Ans:**
- An **Autoencoder** is deterministic, trained via backpropagation using gradient descent on reconstruction error, and acts as a non-linear dimensionality reducer.
- An **RBM** is an energy-based, stochastic generative model trained via Contrastive Divergence, learning an explicit probability distribution over states.

#### Q40. What is the difference between an RBM and a Generative Adversarial Network (GAN)?
**Ans:**
- A **GAN** uses two competing networks (Generator and Discriminator) in a minimax game without explicitly defining an energy function.
- An **RBM** uses a single bipartite network with symmetric undirected weights defined by an explicit energy function and trained via MCMC sampling.

#### Q41. What is the difference between Collaborative Filtering and Content-Based Filtering?
**Ans:**
- **Collaborative Filtering** recommends items based on patterns across many users (user-item interaction matrix) without needing item attributes.
- **Content-Based Filtering** recommends items based on similarities between item features (e.g., plot keywords, actors, genre metadata).

#### Q42. Why is RBM well-suited for Collaborative Filtering?
**Ans:** Because user movie preferences are naturally multi-modal (a user can like both 90s action and animated comedies). RBM's distributed latent representations can model overlapping preferences simultaneously without forcing users into a single rigid cluster.

#### Q43. What are the main limitations of an RBM recommender?
**Ans:**
1. High sensitivity to hyperparameter tuning (learning rate, momentum, CD steps).
2. Suffers from the cold-start problem when no interaction data exists.
3. Scaling to millions of items requires large visible layers or hierarchical architectures.

#### Q44. What is a Deep Belief Network (DBN)?
**Ans:** A deep neural network formed by stacking multiple RBMs on top of each other, where the hidden layer of one RBM acts as the visible layer for the next RBM, trained layer-by-layer in a greedy unsupervised fashion.

#### Q45. How would you improve this project in future work?
**Ans:** Implement a hybrid model combining RBM collaborative filtering with text embeddings of movie synopses (content-based), expand to the 25M MovieLens dataset, and implement a conditional Softmax rating RBM.

