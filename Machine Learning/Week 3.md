# **Probabilistic Models in Machine Learning**

## **1. Fundamentals of Probability**

### **1.1 Basic Concepts**

In machine learning, we often approach problems through the lens of probability. Probabilistic methods allow us to express uncertainty about outcomes rather than making hard yes/no distinctions.

**Probability** refers to the measure of the likelihood that an event will occur. It is expressed as a number between 0 and 1, where:
- 0 indicates impossibility
- 1 indicates certainty

The key advantages of probabilistic approaches in machine learning include:
- Making probabilistic predictions instead of binary decisions
- Incremental effect of each training example on estimated probabilities
- Ability to incorporate prior knowledge with observed data

However, these approaches also have disadvantages:
- Need for initial probability estimates (which may be unknown)
- Potentially high computational cost

### **1.2 Bayes' Theorem**

**Bayes' theorem** is a fundamental concept in probabilistic machine learning that describes the probability of an event based on prior knowledge of conditions related to the event.

The theorem is expressed as:

$$P(h|D) = \frac{P(D|h)P(h)}{P(D)}$$

Where:
- $P(h|D)$ is the **posterior probability** of hypothesis $h$ given data $D$
- $P(D|h)$ is the **likelihood** - probability of observing data $D$ given that hypothesis $h$ is true
- $P(h)$ is the **prior probability** of hypothesis $h$ before seeing any data
- $P(D)$ is the **evidence** - prior probability of observing the data regardless of the hypothesis

This relationship allows us to update our beliefs about hypotheses based on observed evidence.

**Example: The Wrong Underwear Problem**

To illustrate Bayes' theorem, consider finding unknown underwear in your drawer after a business trip:
- Hypothesis $h$: Your partner is cheating
- Data $D$: Strange underwear in drawer

We need to compute:
1. $P(D|h)$ - If they cheated, how likely is strange underwear to appear? (Assume 50%)
2. $P(h)$ - Prior probability they cheated (Studies suggest about 4% for married partners per year)
3. $P(D|\neg h)$ - Probability of innocent explanation (Assume 5%)

Computing $P(h|D)$ using Bayes' theorem:

$$P(h|D) = \frac{P(D|h)P(h)}{P(D|h)P(h) + P(D|\neg h)P(\neg h)}$$

$$P(h|D) = \frac{0.5 \times 0.04}{0.5 \times 0.04 + 0.05 \times (1 - 0.04)} = 0.294$$

This indicates that despite the suspicious evidence, there's only about a 29.4% chance of cheating because of the low prior probability.

However, if it happens again, the prior probability becomes 0.294:

$$P(h|D) = \frac{0.5 \times 0.294}{0.5 \times 0.294 + 0.05 \times (1 - 0.294)} = 0.806$$

This shows how new evidence updates our beliefs systematically through Bayes' theorem.

### **1.3 Prior and Posterior Probabilities**

When applying Bayesian learning to machine learning problems, we work with several key probability concepts:

**Prior probability** ($P(h)$):
- The probability of hypothesis $h$ before seeing any data
- Represents our initial beliefs or background knowledge
- May reflect domain expertise or assumptions

**Likelihood** ($P(D|h)$):
- The probability of observing data $D$ given hypothesis $h$ is true
- Measures how well the hypothesis explains the observed data

**Posterior probability** ($P(h|D)$):
- The probability of hypothesis $h$ after observing data $D$
- What we're typically trying to calculate in machine learning
- Used to make predictions about new data

**Evidence** ($P(D)$):
- The marginal probability of observing data $D$ regardless of hypothesis
- Can be calculated as $P(D) = P(D|h)P(h) + P(D|\neg h)P(\neg h)$ when we have two hypotheses

In Bayesian learning, we often seek the **Maximum a Posteriori (MAP)** hypothesis:

$$h_{MAP} = \arg\max_{h \in H} P(h|D) = \arg\max_{h \in H} \frac{P(D|h)P(h)}{P(D)} = \arg\max_{h \in H} P(D|h)P(h)$$

When we assume that all hypotheses are equally likely a priori (uniform prior), we simplify to finding the **Maximum Likelihood (ML)** hypothesis:

$$h_{ML} = \arg\max_{h \in H} P(D|h)$$

This framework forms the foundation for many probabilistic learning algorithms, allowing us to make decisions under uncertainty by systematically updating our beliefs as we observe more data.

# **2. Naive Bayes Classification**

## **2.1 Theory and Assumptions**

**Naive Bayes** is a practical and effective approach to Bayesian learning. It's particularly resistant to overfitting because of its simplicity.

As a classifier, Naive Bayes estimates the probability $P(y_i|x_i, D)$ where:
- $y_i$ is the target value (class label)
- $x_i$ is the input feature vector
- $D$ is the training data

The goal is to learn an approximation of the mapping $y = f(x)$ where $f(x)$ takes values from a finite set $V$.

For a new instance with features $\langle a_1, a_2, \ldots, a_n \rangle$, we classify it using:

$$v_{MAP} = \arg\max_{v_i \in V} P(v_i|a_1, a_2, \ldots, a_n)$$

Using Bayes' theorem, this can be rewritten as:

$$v_{MAP} = \arg\max_{v_i \in V} \frac{P(a_1, a_2, \ldots, a_n|v_i)P(v_i)}{P(a_1, a_2, \ldots, a_n)} = \arg\max_{v_i \in V} P(a_1, a_2, \ldots, a_n|v_i)P(v_i)$$

The "naive" part comes from two simplifying assumptions:
1. Position does not matter (if applicable)
2. **Conditional independence assumption**: Features are conditionally independent given the target value

This second assumption leads to:

$$P(a_1, a_2, \ldots, a_n|v_i) = \prod_j P(a_j|v_i)$$

Therefore, the Naive Bayes classifier computes:

$$v_{NB} = \arg\max_{v_i \in V} P(v_i) \prod_j P(a_j|v_i)$$

When the conditional independence assumption is satisfied, $v_{NB} = v_{MAP}$. Even when this assumption is not satisfied, Naive Bayes often performs well in practice.

## **2.2 Model Implementation**

To implement a Naive Bayes classifier, we need to:

1. Estimate the prior probabilities $P(v_i)$ from training data by counting the frequency of each class
2. Estimate the conditional probabilities $P(a_j|v_i)$ by counting the proportion of instances with value $a_j$ for feature $j$ among all instances of class $v_i$

For practical implementation, we often use the log of probabilities to avoid numerical underflow with many features:

$$v_{NB} = \arg\max_{v_i \in V} \log P(v_i) + \sum_j \log P(a_j|v_i)$$

**Example: Tennis Decision Problem**

Consider a dataset about playing tennis based on weather conditions:

![image](https://github.com/user-attachments/assets/1c2553c8-649a-45dd-801a-7497134bc06c)

To classify an example $\langle \text{Outlook} = \text{Sun}, \text{Temp} = \text{Cool}, \text{Humidity} = \text{High}, \text{Wind} = \text{Strong} \rangle$, we compute:

1. Prior probabilities:
   - $P(\text{Tennis} = \text{yes}) = \frac{9}{14} = 0.64$
   - $P(\text{Tennis} = \text{no}) = \frac{5}{14} = 0.36$

2. Conditional probabilities for each feature value given the class:
   - $P(\text{Sun}|\text{yes}) = \frac{2}{9}$, $P(\text{Sun}|\text{no}) = \frac{3}{5}$
   - $P(\text{Cool}|\text{yes}) = \frac{3}{9}$, $P(\text{Cool}|\text{no}) = \frac{1}{5}$
   - $P(\text{High}|\text{yes}) = \frac{3}{9}$, $P(\text{High}|\text{no}) = \frac{4}{5}$
   - $P(\text{Strong}|\text{yes}) = \frac{3}{9}$, $P(\text{Strong}|\text{no}) = \frac{3}{5}$

3. Calculate $v_{NB}$:
   - $P(\text{yes}) \times P(\text{Sun}|\text{yes}) \times P(\text{Cool}|\text{yes}) \times P(\text{High}|\text{yes}) \times P(\text{Strong}|\text{yes}) = 0.0053$
   - $P(\text{no}) \times P(\text{Sun}|\text{no}) \times P(\text{Cool}|\text{no}) \times P(\text{High}|\text{no}) \times P(\text{Strong}|\text{no}) = 0.0206$

Since 0.0206 > 0.0053, $v_{NB} = \text{no}$, indicating we would not play tennis given these conditions.

The probability of this prediction being correct is:
$$P(v_{NB}) = \frac{0.0206}{0.0206 + 0.0053} = 0.795$$

## **2.3 Probability Estimation**

When estimating probabilities for Naive Bayes, we typically use **maximum likelihood estimation** by counting frequencies in the training data.

For prior probabilities $P(v_i)$:
- Count how many training examples belong to each class
- Divide by the total number of examples

For conditional probabilities $P(a_j|v_i)$:
- Count how many examples of class $v_i$ have feature value $a_j$
- Divide by the total number of examples in class $v_i$

In sentiment classification tasks (e.g., positive vs. negative text):
- $P(v) = \frac{N_v}{N_{doc}}$ where $N_v$ is the number of documents with class $v$
- $P(w_j|v) = \frac{count(w_j,v)}{\sum_{w \in Vocab} count(w,v)}$ where $count(w_j,v)$ is the number of times word $w_j$ appears in documents of class $v$

## **2.4 Smoothing Techniques**

Simple counting for probabilities can lead to overfitting, especially with limited training data. A particular problem occurs when $P(a_j|v_i) = 0$ because a feature value was never observed with a particular class in the training data.

To address this, various **smoothing techniques** can be applied:

**Laplace (Add-one) Smoothing**:
This adds a small count to each feature value to ensure no probability is exactly zero:

$$P(a_j|v_i) = \frac{count(a_j,v_i) + 1}{\sum_{a \in A_j} (count(a,v_i) + 1)} = \frac{count(a_j,v_i) + 1}{(\sum_{a \in A_j} count(a,v_i)) + |A_j|}$$

Where $|A_j|$ is the number of possible values for feature $j$.

**m-estimate**:
A more general smoothing approach:

$$P(a_j|v_i) = \frac{n_c + m \cdot p}{n + m}$$

Where:
- $n_c$ is the count of examples with $a_j$ and class $v_i$
- $n$ is the total count of examples with class $v_i$
- $p$ is a prior estimate (often $\frac{1}{k}$ if feature has $k$ values)
- $m$ is the equivalent sample size (weight given to the prior)

For text classification, a common smoothing technique is **Add-1 (Laplace) smoothing**:

$$P(w_j|v) = \frac{count(w_j,v) + 1}{\sum_{w \in V} (count(w,v) + 1)} = \frac{count(w_j,v) + 1}{(\sum_{w \in V} count(w,v)) + |V|}$$

This prevents zero probabilities for words that don't appear in the training data for a particular class.

## **2.5 Applications**

Naive Bayes is widely used in various applications due to its simplicity, efficiency, and surprisingly good performance:

**Text Classification**:
- Spam detection
- Sentiment analysis
- Topic categorization
- Language identification

**Example**: Document classification where features are words and classes might be topics like "sports," "politics," or "entertainment."

For text with continuous-valued features, Naive Bayes can be extended by modeling each feature as a **Gaussian distribution**:

$$P(a_1, a_2, \ldots, a_n|v_i) = \prod_{j=1}^n N(a_j|\mu_{jc}, \sigma^2_{jc})$$

Where $\mu_{jc}$ and $\sigma^2_{jc}$ are the mean and variance of feature $j$ in instances of class $c = v_i$.

**Anomaly Detection**:
Given a dataset $D$ of non-anomalous instances, we can estimate $P(x)$ and detect anomalies when $P(x_1) < threshold$ for new instances $x_1$.

**Medical Diagnosis**:
Predicting diseases based on symptoms, where each symptom is a feature.

**Advantages of Naive Bayes in Practice**:
- Fast to train and predict
- Works well with high-dimensional data
- Requires little training data
- Handles missing values well
- Often used as a baseline model for comparison with more complex models

Despite its "naive" assumption of feature independence, which is rarely true in practice, Naive Bayes often performs competitively against more sophisticated models, especially in text classification tasks.

# **3. Mixture Models**

## **3.1 Gaussian Mixture Models**

**Mixture models** are probabilistic models that represent data as being generated from a mixture of several underlying probability distributions. They're particularly useful for modeling complex data that cannot be adequately described by a single distribution.

The most common type is the **Gaussian Mixture Model (GMM)**, which combines multiple Gaussian (normal) distributions.

In a mixture model, the probability of an element $x_i$ is expressed as:

$$p(x_i) = \sum_{k=1}^K \pi_k p_k(x_i)$$

Where:
- $K$ is the number of distributions in the mixture
- $\pi_k$ are the **mixing weights** (must sum to 1)
- $p_k(x_i)$ is the probability assigned to $x_i$ by the $k$-th distribution

For a **Gaussian Mixture Model**, each component is a multivariate Gaussian with mean $\mu_k$ and covariance $\Sigma_k$:

$$p(x_i) = \sum_{k=1}^K \pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)$$

For univariate Gaussians, this simplifies to:

$$p(x_i) = \sum_{k=1}^K \pi_k \mathcal{N}(x_i|\mu_k, \sigma^2_k)$$

Gaussian mixture models are useful because:
- Many natural processes generate Gaussian-distributed values
- Combinations of these processes result in mixture models
- They can approximate almost any continuous distribution given enough components

**Example**: Human height distributions often follow a mixture of Gaussians (rather than a single Gaussian) because they combine different subpopulations (e.g., male and female height distributions).

![image](https://github.com/user-attachments/assets/a4f5a7d0-071e-4d09-9cd1-582c714573a0)

## **3.2 Parameter Estimation**

The challenge with mixture models is estimating their parameters from observed data. For a Gaussian mixture model with $K$ components, we need to estimate:
- The means $\mu_1, \mu_2, ..., \mu_K$
- The covariances $\Sigma_1, \Sigma_2, ..., \Sigma_K$ (or variances $\sigma^2_1, \sigma^2_2, ..., \sigma^2_K$ for univariate case)
- The mixing weights $\pi_1, \pi_2, ..., \pi_K$

This is challenging because we don't know which distribution generated each data point - this is a **hidden variable**. If we knew which distribution each data point came from, the solution would be straightforward (e.g., for means, we would just compute the sample mean for each group).

We can represent this hidden information as:
- For each data point $x_i$, we have hidden variables $z_{i1}, z_{i2}, ..., z_{iK}$
- $z_{ij} = 1$ if $x_i$ came from the $j$-th distribution, and 0 otherwise

For a simple case with two Gaussians of the same variance $\sigma^2$, we want to estimate:
$$h = \langle\mu_1, \mu_2\rangle$$

If we knew which Gaussian generated each point, we could simply compute:
$$\mu_j = \frac{\sum_{i=1}^m z_{ij}x_i}{\sum_{i=1}^m z_{ij}}$$

But since we don't know the $z_{ij}$ values, we need a special algorithm to estimate both the means and the hidden variables simultaneously.

## **3.3 Expectation Maximization (EM) Algorithm**

The **Expectation Maximization (EM) algorithm** is a general method for finding maximum likelihood estimates of parameters in models with hidden variables. For mixture models, it works by iteratively:
1. Estimating the hidden variables given current parameter estimates
2. Updating the parameters given the estimated hidden variables

For a Gaussian mixture model with two components, the EM algorithm proceeds as follows:

1. **Initialization**: Pick arbitrary initial values for $\mu_1$ and $\mu_2$

2. **Repeat until convergence**:
   
   a. **Expectation (E) step**: Calculate the expected value $E[z_{ij}]$ of each hidden variable $z_{ij}$, which is the probability that $x_i$ was generated by the $j$-th Gaussian:
   
   $$E[z_{ij}] = \frac{p(x = x_i|\mu = \mu_j)}{\sum_{n=1}^2 p(x = x_i|\mu = \mu_n)} = \frac{e^{-\frac{1}{2\sigma^2}(x_i-\mu_j)^2}}{\sum_{n=1}^2 e^{-\frac{1}{2\sigma^2}(x_i-\mu_n)^2}}$$
   
   b. **Maximization (M) step**: Calculate a new maximum likelihood hypothesis $h' = \langle\mu'_1, \mu'_2\rangle$ assuming each $z_{ij}$ takes its expected value $E[z_{ij}]$:
   
   $$\mu_j \leftarrow \frac{\sum_{i=1}^m E[z_{ij}]x_i}{\sum_{i=1}^m E[z_{ij}]}$$
   
   This is the weighted average of all points, where the weight for point $x_i$ is its probability of belonging to the $j$-th Gaussian.

3. **Termination**: The algorithm continues until the change in parameters becomes very small.

The EM algorithm is guaranteed to increase the likelihood $p(D|h)$ with each iteration until it reaches a stationary point (which may be a local maximum). To avoid getting stuck in a poor local maximum, random restarts or careful initialization (e.g., using k-means++) is recommended.

**More General EM Algorithm**:

EM can be applied more generally to estimate parameters $\theta$ of any distribution when we only observe a portion of the data. If $D = \{x_1, x_2, ..., x_m\}$ is our observed data and $Z = \{z_1, z_2, ..., z_m\}$ is the corresponding unobserved data, the full data is $Y = D \cup Z$.

The general EM algorithm alternates between:

1. **E-step**: Calculate $Q(h'|h) = E[\ln p(Y|h')|h, D]$ - the expected value of the log likelihood of the full data given the current hypothesis $h$

2. **M-step**: Update $h$ to $h'$ that maximizes $Q$:
   $h \leftarrow \arg\max_{h'} Q(h'|h)$

**Example: Coin Flipping Experiment**

Consider two coins A and B with unknown probabilities $\theta_A$ and $\theta_B$ of landing heads. We randomly select a coin for each experiment and perform 10 tosses, but we don't record which coin was used.

The EM algorithm would:
1. Start with random guesses for $\theta_A$ and $\theta_B$
2. E-step: Calculate the probability each experiment used coin A vs. B
3. M-step: Update the estimates of $\theta_A$ and $\theta_B$ based on these probabilities
4. Repeat until convergence

Through this process, EM can effectively separate the mixture components even when we don't know which component generated each data point.

# **4. Clustering**

## **4.1 K-means Algorithm**

**K-means clustering** is a popular unsupervised learning algorithm that partitions data into K distinct, non-overlapping clusters. It can be viewed as a special case of the EM algorithm applied to a mixture of Gaussians with equal variances.

The core idea of K-means is to identify K cluster centres such that the total distance between data points and their closest cluster centres is minimized.

**Algorithm steps**:

1. **Initialization**: Choose K random points as initial cluster centres $\mu_1, \mu_2, ..., \mu_K$

2. **Repeat until convergence**:
   
   a. **Assignment step**: Assign each data point $x_i$ to its closest cluster centre:
   
   $$z_i = \arg\min_k |x_i - \mu_k|$$
   
   Where $z_i$ indicates which cluster the point belongs to.
   
   b. **Update step**: Recalculate each cluster centre as the mean of all points assigned to that cluster:
   
   $$\mu_k = \frac{1}{|\{j: z_j = k\}|} \sum_{i \in \{j: z_j = k\}} x_i$$
   
   Where $|\{j: z_j = k\}|$ is the number of points in cluster k.

3. **Termination**: Continue until cluster assignments no longer change or a maximum number of iterations is reached.

This algorithm minimizes the **within-cluster sum of squares (WCSS)** or the sum of squared Euclidean distances between points and their assigned cluster centres.

![image](https://github.com/user-attachments/assets/a1a1facc-e6b7-43ff-bf74-5affa8120dee)

**Example of K-means Clustering**:

Consider the following dataset with 10 instances, each with two attributes:

| Instance | $x_1$ | $x_2$ |
|----------|-------|-------|
| $X_1$    | 5     | 8     |
| $X_2$    | 6     | 7     |
| $X_3$    | 6     | 4     |
| $X_4$    | 5     | 7     |
| $X_5$    | 5     | 5     |
| $X_6$    | 6     | 5     |
| $X_7$    | 1     | 7     |
| $X_8$    | 7     | 5     |
| $X_9$    | 6     | 5     |
| $X_{10}$ | 6     | 7     |

Starting with initial centres at (7,5), (9,7), and (9,1), and using Manhattan distance (sum of absolute differences), we first calculate the distance from each point to each cluster centre:

| Instance | Distance to C₁ | Distance to C₂ | Distance to C₃ | Closest |
|----------|----------------|----------------|----------------|---------|
| $X_1$    | 5              | 5              | 11             | C₁/C₂   |
| $X_2$    | 3              | 3              | 9              | C₁/C₂   |
| $X_3$    | 2              | 6              | 6              | C₁      |
| $X_4$    | 4              | 4              | 10             | C₁/C₂   |
| $X_5$    | 2              | 6              | 8              | C₁      |
| $X_6$    | 1              | 5              | 7              | C₁      |
| $X_7$    | 8              | 8              | 14             | C₁/C₂   |
| $X_8$    | 0              | 4              | 6              | C₁      |
| $X_9$    | 1              | 5              | 7              | C₁      |
| $X_{10}$ | 3              | 3              | 9              | C₁/C₂   |

After resolving ties and assigning points to clusters, we calculate new cluster centres by averaging the points in each cluster. The process repeats until convergence, when points no longer change cluster assignments.

After convergence, the final clusters might be:
- Cluster 1: $X_3$, $X_5$, $X_6$, $X_8$, $X_9$ with centre (6,5)
- Cluster 2: $X_1$, $X_2$, $X_4$, $X_7$, $X_{10}$ with centre (4.6,7.2)
- Cluster 3: (Empty or reassigned)

## **4.2 Relationship to Mixture Models**

K-means clustering and Gaussian mixture models are closely related:

**K-means as a Special Case of GMM**:
- K-means can be viewed as a simplified version of the EM algorithm for Gaussian mixture models
- It assumes all Gaussians have:
  - The same variance (spherical covariance matrices)
  - Equal mixing weights ($\pi_k = \frac{1}{K}$ for all $k$)

**Key differences**:

1. **Hard vs. Soft Clustering**:
   - **K-means** performs **hard clustering**: Each point belongs to exactly one cluster
   - **GMM** performs **soft clustering**: Each point has a probability of belonging to each cluster (represented by the expected values of $z_{ij}$)

2. **Distance Measure**:
   - **K-means** typically uses Euclidean distance (or other metrics like Manhattan distance)
   - **GMM** uses a probabilistic measure based on the likelihood of points under each Gaussian distribution

3. **Output**:
   - **K-means** outputs only cluster assignments and centroids
   - **GMM** provides a full probabilistic model of the data, with means, covariances, and mixing weights

**Mathematical Connection**:

In the limit where:
- All Gaussians have the same, spherical covariance matrix: $\Sigma_k = \sigma^2 I$ for all $k$
- The variance $\sigma^2$ approaches zero

The E-step of EM becomes equivalent to assigning each point to its nearest cluster centre (K-means assignment step), and the M-step becomes equivalent to computing the mean of assigned points (K-means update step).

**Probabilistic Interpretation of K-means**:

Each cluster in K-means can be seen as a Gaussian distribution with:
- Mean at the cluster centre
- Fixed, identical variance for all clusters
- Each point is assigned to the cluster that maximizes its probability

In this view, when we assign a point to its closest cluster centre, we're effectively choosing the distribution that gives the highest likelihood for that point, and the points associated with a given cluster are the ones most likely to have been generated by the Gaussian centred at that cluster's centroid.

This relationship helps explain why K-means works well for roughly spherical clusters of similar sizes but struggles with clusters of different shapes, sizes, or densities—conditions where a full Gaussian mixture model would be more appropriate.
