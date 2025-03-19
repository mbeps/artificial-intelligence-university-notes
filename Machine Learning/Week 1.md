# **1. Introduction to Machine Learning**

## **1.1 Definitions and Applications**

**Machine learning** is defined as "the field of study that gives computers the ability to learn from data without being explicitly programmed" (Samuel, 1959).

More formally, machine learning is "a set of methods that can automatically detect patterns in data, and then use the patterns to predict future data, or to perform other kinds of decision making" (Murphy, 2012).

### Real-world Applications

Machine learning powers many technologies we use daily:

- **Google search**: Ranking search results and understanding queries
- **Google translate**: Language translation
- **Spam filtering**: Identifying unwanted emails
- **Speech recognition**: Converting speech to text (e.g., Amazon Alexa)
- **Text prediction**: Suggesting next words when typing
- **Recommendation systems**: Suggesting products (e.g., Amazon recommendations)

## **1.2 Types of Machine Learning**

Machine learning is traditionally divided into three main categories:

### **Supervised Learning**

**Supervised learning** involves learning from labelled data to make predictions on unseen data.

**Key components** for successful supervised learning:
1. **Representative training data**: Data should be annotated with meaningful labels
2. **Sophisticated feature extraction**: Methods for identifying important properties of the data
3. **Appropriate algorithm selection**: Choosing the right ML algorithm for the task

**Training process**:
- Start with a training set $D = \{(x_i, y_i)\}_{i=1}^N$
- Where $x_i$ are input features and $y_i$ are target outputs/labels
- Learn a function $f$ such that $f(x_i) = y_i$

**Example**: Iris flower classification
- **Input features ($x_i$)**: Petal area, sepal area, etc.
- **Target labels ($y_i$)**: Flower species (setosa, versicolor, or virginica)
- **Goal**: Predict flower species based on measurements

Supervised learning is further divided into:

1. **Classification**: Predicting discrete class labels
   - **Binary classification**: Two possible classes (e.g., spam or not spam)
   - **Multiclass classification**: More than two classes (e.g., iris species)

2. **Regression**: Predicting continuous values
   - **Example**: Predicting income level, exam scores, etc.

### **Unsupervised Learning**

**Unsupervised learning** involves finding patterns in unlabelled data.

- Works with training data in the form: $D = \{x_i\}_{i=1}^N$ (no target labels)
- Aims to discover hidden structures or patterns in the data
- Often called **knowledge discovery**

**Common techniques**:
- **Clustering**: Grouping similar data points together
  - **Example**: Grouping news articles by topic
  - **Example**: Clustering gene expression data to find patterns

**Mathematical formulation**:
- Often formulated as density estimation: $p(x_i|D)$
- Unlike supervised learning's $p(y_i|x_i,D)$

### **Reinforcement Learning**

**Reinforcement learning** involves an agent learning to make decisions by taking actions in an environment to maximize rewards.

- The agent learns from trial and error
- Receives feedback in the form of rewards or penalties
- Aims to find an optimal policy (strategy) for decision-making

**Example**: A program learning to play chess by receiving positive feedback for winning moves and negative feedback for losing moves.

![image](https://github.com/user-attachments/assets/367b7e97-3b8a-44ee-b36b-9985b7c89074)

> Shows the "long tail" distribution of word frequencies in Moby Dick, illustrating why generalisation is important in machine learning

### Comparing Learning Types

| Learning Type | Data | Goal | Examples |
|---------------|------|------|----------|
| **Supervised** | Labelled (features + targets) | Predict labels for new data | Classification, regression |
| **Unsupervised** | Unlabelled (features only) | Discover patterns | Clustering, dimensionality reduction |
| **Reinforcement** | Environment feedback | Optimize decision-making | Game playing, robotics |

# **2. Supervised Learning**

## **2.1 Classification**

**Classification** is a fundamental supervised learning task that involves predicting discrete class labels for input data. It can be viewed as a **function approximation** problem, where we assume:

$y = f(x)$

Learning involves estimating $f$, creating an estimate $\hat{f}$. After learning, we predict using:

$\hat{y} = \hat{f}(x)$

### **2.1.1 Binary Classification**

**Binary classification** involves distinguishing between two classes:

$y \in \{0, 1\}$ or $y \in \{-1, 1\}$

This is the simplest form of classification, where we often conceptualize the problem as "is the example in the class or not?"

**Examples:**
- Spam detection (spam or not spam)
- Medical diagnosis (disease present or absent)
- Credit approval (approve or reject)

In binary classification, we can think of the positive class (often labeled 1) as the class of interest, and the negative class (often labeled 0) as everything else.

### **2.1.2 Multiclass Classification**

**Multiclass classification** involves distinguishing between more than two classes:

$y \in \{1, 2, ..., C\}$

where $C$ is the number of classes.

**Examples:**
- Iris flower classification (setosa, versicolor, or virginica)
- Sentiment analysis (positive, neutral, negative)
- Handwritten digit recognition (0-9)

![image](https://github.com/user-attachments/assets/6c57aa73-fee4-43e1-afa9-94c124750eda)

> Shows a visual representation of the Iris classification problem with different coloured regions representing different classes

## **2.2 Regression**

**Regression** involves predicting continuous numerical values rather than discrete classes.

**Linear regression** is a common approach where we assume the output $y$ is a linear combination of the input features $x$:

$y(x) = \sum_{j=1}^{D} w_j x_j + \epsilon$

where:
- $w_j$ are the weights forming a weight vector $w$
- $\epsilon$ is the residual error
- $D$ is the number of input features

For a one-dimensional input, we can write:
$\hat{y}(x) = w_0 + w_1 x_1$

where:
- $w_0$ is the **intercept** or **bias**
- $w_1$ is the **slope** 

**Example:**
Predicting house prices based on features like square footage, number of bedrooms, and location.

![image](https://github.com/user-attachments/assets/210d111d-c3b8-457f-b7f3-a09e53eaca0e)

> Shows a linear regression visualization with data points and a best-fit line

## **2.3 Probabilistic Prediction**

Many real-world classification problems involve uncertainty, especially at decision boundaries. **Probabilistic prediction** addresses this by returning probability distributions rather than just class labels.

For classification, we model:

$p(y_i|x_i, D)$

This represents the probability of class $y_i$ given input $x_i$ and training data $D$.

Sometimes we explicitly consider the model $M$ used:

$p(y_i|x_i, D, M)$

### Maximum a Posteriori (MAP) Estimation

To make a single class prediction, we can choose the class with the highest probability:

$\hat{y} = \arg\max_{c=1}^C p(y=c|x_i, D)$

This is called the **MAP estimate** (Maximum A Posteriori), which selects the mode of the probability distribution.

**Example:**
In a three-class prediction problem (e.g., iris classification), a probabilistic classifier might return:
- $p(y = \text{setosa}|x) = 0.15$
- $p(y = \text{versicolor}|x) = 0.75$
- $p(y = \text{virginica}|x) = 0.10$

The MAP decision would be to classify this example as "versicolor" since it has the highest probability.

### Advantages of Probabilistic Prediction

1. **Uncertainty quantification**: Provides confidence levels for predictions
2. **Decision-making flexibility**: Allows different thresholds for different use cases
3. **Improved model evaluation**: Enables more sophisticated performance metrics
4. **Informed decision-making**: Helps prioritize cases requiring human review

Probabilistic approaches are particularly valuable when classification errors have different costs or when decisions are part of a larger system.

# **3. Unsupervised Learning**

## **3.1 Clustering**

**Unsupervised learning** involves finding patterns in data without labeled examples. Unlike supervised learning, we start with only input data:

$D = \{x_i\}_{i=1}^N$

without corresponding target labels.

**Clustering** is the most common form of unsupervised learning. It aims to partition data into groups or **clusters** where:
- Data points within a cluster should be similar to each other
- Data points in different clusters should be dissimilar
- The clusters are inferred from the data itself, not predetermined

### Cluster Analysis Process

1. **Feature selection/extraction**: Deciding which attributes of the data to use
2. **Algorithm selection**: Choosing an appropriate clustering method
3. **Cluster validation**: Evaluating the quality of the clusters
4. **Results interpretation**: Drawing conclusions from the clusters

### Determining the Number of Clusters

When clustering, we need to estimate the distribution over the number of clusters $p(K|D)$, where $K$ is the number of clusters.

This is often simplified by approximating $p(K|D)$ by its mode:

$K^* = \arg\max_K p(K|D)$

Selecting the appropriate number of clusters $K$ is a critical form of **model selection**.

### Examples of Clustering Applications

- **News article clustering**: Grouping articles by topic
- **Gene expression data clustering**: Finding patterns in gene expression data
- **Customer segmentation**: Grouping customers by purchasing behavior
- **Image segmentation**: Dividing images into meaningful regions

![image](https://github.com/user-attachments/assets/b3c2dca1-4dc9-4d22-b726-c2e02e6db65b)

> Shows clustering visualization of gene expression data, with colored regions representing different clusters}}

### Evaluating Clusters

There are two main approaches to evaluating clustering results:

1. **External evaluation**: Comparing clustering results to known ground truth or measuring how well they serve a downstream task
   - Example: How well market segmentation clusters predict customer behavior

2. **Internal evaluation**: Measuring cluster coherence and separation without external references
   - Example: The **Davis-Bouldin index**:
   
   $DB = \frac{1}{N}\sum_{i=1}^n \max_{j\neq i}\left(\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}\right)$
   
   where:
   - $c_i$ is the centroid of cluster $i$
   - $\sigma_i$ is the average distance of points in cluster $i$ to the centroid
   - $d(c_i, c_j)$ is the distance between centroids
   
   A smaller DB index indicates better clustering (tightly grouped clusters that are far apart).

![image](https://github.com/user-attachments/assets/0d709e4e-b086-4269-97bd-4a371de2155e)

> Shows a visual representation of different clusters with various shapes and colors}}


## **3.2 Density Estimation**

**Density estimation** is a fundamental concept in unsupervised learning where we model the probability distribution of the data.

In this framework, unsupervised learning involves building models of the form:

$p(x_i|D)$

This differs from supervised learning in two key ways:

1. We don't predict class labels $p(y_i|x_i, D)$ because we don't have labels
2. The input $x_i$ is typically a vector of features, so we're modeling a multivariate probability distribution (which is more complex than the univariate distributions in classification)

### Importance of Density Estimation

**Density estimation** allows us to:
- Understand the underlying structure of the data
- Identify regions of high and low probability (useful for anomaly detection)
- Generate new samples similar to the observed data
- Facilitate other unsupervised learning tasks

### Connection to Clustering

Clustering can be viewed as a special case of density estimation where:
- We assume the data comes from a mixture of simpler distributions
- Each cluster corresponds to one component of the mixture
- Points are assigned to the most likely component/cluster

For example, in a Gaussian mixture model, we might model the data as coming from $K$ different Gaussian distributions, where $K$ is the number of clusters.

### Parametric vs. Non-parametric Approaches

Density estimation methods can be categorized as:

**Parametric approaches**: 
- Make assumptions about the underlying distribution (e.g., Gaussian)
- Estimate the parameters of the assumed distribution
- Computationally simpler but potentially less accurate if assumptions are wrong

**Non-parametric approaches**:
- Make fewer assumptions about the data distribution
- Let the data speak for itself
- More flexible but can be computationally intensive for large datasets

Unsupervised learning is particularly valuable because labeled data is uncommon "in the wild," making it arguably more natural than supervised learning in many real-world scenarios.

# **4. Machine Learning Models**

## **4.1 K-Nearest Neighbors (kNN)**

**K-Nearest Neighbors (kNN)** is a simple, yet effective non-parametric classification algorithm. Rather than building an explicit model during training, kNN uses the training examples directly to make predictions.

### Working Principle

The core idea of kNN is remarkably straightforward:
1. Store all training examples with their labels
2. When classifying a new example, find the k training examples that are most similar (nearest) to it
3. Assign the most common class among these k neighbors to the new example

### Training Process

Unlike many other algorithms, kNN has essentially **no training phase**. It simply stores the entire training dataset:

$D = \{(x_i, y_i)\}_{i=1}^N$

This is sometimes called **lazy learning** or **instance-based learning** since the algorithm defers all computation until classification time.

### Classification Process

To classify a new example $x$:

1. Calculate the distance between $x$ and all examples in the training set
2. Select the $k$ examples with the smallest distances to $x$ (the k nearest neighbors)
3. Assign the class based on these neighbors

The simple version assigns the majority class among the k neighbors. The more sophisticated version computes class probabilities:

$p(y = c|x, D, K) = \frac{1}{K}\sum_{i\in N_K(x,D)}I(y_i = c)$

where:
- $N_K(x,D)$ are the indices of the k nearest points to $x$ in $D$
- $I(e)$ is an indicator function: $I(e) = 1$ if $e$ is true, $0$ otherwise

### Distance Metric

The performance of kNN heavily depends on the choice of distance metric. Common metrics include:

- **Euclidean distance**: $\sqrt{\sum_{j=1}^d (x_j - z_j)^2}$
- **Manhattan distance**: $\sum_{j=1}^d |x_j - z_j|$
- **Minkowski distance**: $(\sum_{j=1}^d |x_j - z_j|^p)^{1/p}$

**Example**: Using Manhattan distance 
For two examples $i$ with features $(x^i_1, x^i_2)$ and $j$ with features $(x^j_1, x^j_2)$:

$\text{distance} = |x^i_1 - x^j_1| + |x^i_2 - x^j_2|$

### The Effect of k

The choice of $k$ significantly affects the classifier's behavior:

- **Small k** (e.g., k=1): More sensitive to noise, can lead to overfitting
- **Large k**: Smoother decision boundaries, more robust to noise, but may miss important patterns

**Example calculation**: For a point with 5 nearest neighbors having labels {1, 1, 1, 0, 0}:
- $p(y = 1|x, D, 5) = 3/5 = 0.6$
- $p(y = 0|x, D, 5) = 2/5 = 0.4$
- The point would be classified as class 1 since it has the highest probability.

### Advantages and Limitations

**Advantages**:
- Simple and intuitive
- No assumptions about data distribution
- Works well with multi-class problems
- Naturally handles complex decision boundaries

**Limitations**:
- **Scalability issues**: Computationally expensive for large datasets
- **Curse of dimensionality**: Performance degrades with high-dimensional data
- **Storage requirements**: Must store the entire training set

## **4.2 Parametric vs Non-parametric Models**

The distinction between parametric and non-parametric models is fundamental in machine learning and relates to assumptions made about the data structure.

### Parametric Models

**Parametric models** make specific assumptions about the functional form of the relationship between inputs and outputs, which can be described using a fixed number of parameters.

**Key characteristics**:
- Make strong assumptions about the data structure (e.g., "there are four Gaussian clusters")
- Model complexity is fixed regardless of the amount of data
- Once parameters are learned, original data isn't needed for prediction
- Typically faster to train and predict

**Examples**:
- Linear regression
- Logistic regression
- Naive Bayes
- Neural networks (with fixed architecture)

**Mathematical form**: $y = f(x; \theta)$ where $\theta$ represents a fixed number of parameters.

### Non-parametric Models

**Non-parametric models** make fewer assumptions about the data distribution and allow the complexity of the model to grow with the amount of training data.

**Key characteristics**:
- Make minimal assumptions about the underlying data distribution
- Model complexity increases with the amount of training data
- Often require storing part or all of the training data
- Generally more flexible but computationally intensive

**Examples**:
- K-Nearest Neighbors (kNN)
- Decision trees
- Kernel methods
- Gaussian processes

### Comparison: Pros and Cons

| Aspect | Parametric Models | Non-parametric Models |
|--------|-------------------|------------------------|
| **Assumptions** | Strong assumptions | Minimal assumptions |
| **Flexibility** | Less flexible | More flexible |
| **Data requirements** | Can work with less data | Often need more data |
| **Computational complexity** | Generally simpler | Can be more complex |
| **Overfitting risk** | Less prone (if assumptions hold) | More prone (especially with small datasets) |
| **Interpretability** | Often more interpretable | Can be less interpretable |

### Practical Considerations

When choosing between parametric and non-parametric models:

1. **Domain knowledge**: If you understand the underlying data structure well, parametric models may be appropriate
2. **Data size**: With limited data, parametric models may perform better
3. **Computational resources**: Non-parametric models can be resource-intensive for large datasets
4. **Prediction speed**: Parametric models typically offer faster predictions

![image](https://github.com/user-attachments/assets/7eb46101-82ec-4646-a6ea-d05bfa609b0a)

> Shows a visual representation of k-NN classification with different k values affecting the decision boundaries}}

### Example: 3-NN Classification

Consider a 3-nearest neighbor classifier with training examples:

| Instance | Features (x₁, x₂) | Class |
|----------|------------------|-------|
| X₁ | (1, 1) | C₁ |
| X₂ | (2, 2) | C₁ |
| X₃ | (1, 3) | C₁ |
| X₄ | (4, 2) | C₁ |
| X₅ | (1, 6) | C₂ |
| X₆ | (2, 4) | C₂ |
| X₇ | (2, 5) | C₂ |
| X₈ | (3, 4) | C₂ |
| X₉ | (5, 4) | C₂ |

To classify a new instance (3, 1) using Manhattan distance:

1. Calculate distances:
   - To X₁: |3-1| + |1-1| = 2
   - To X₂: |3-2| + |1-2| = 2
   - To X₃: |3-1| + |1-3| = 4
   - To X₄: |3-4| + |1-2| = 2
   - To X₅: |3-1| + |1-6| = 7
   - To X₆: |3-2| + |1-4| = 4
   - To X₇: |3-2| + |1-5| = 5
   - To X₈: |3-3| + |1-4| = 3
   - To X₉: |3-5| + |1-4| = 5

2. Find the 3 nearest neighbors: X₁, X₂, and X₄ (all with distance 2)

3. Compute class probabilities:
   - P(C₁) = 3/3 = 1
   - P(C₂) = 0/3 = 0

4. Assign class C₁ to the new instance.

# **5. Machine Learning in Practice**

## **5.1 Data Handling**

### **5.1.1 Training and Testing Sets**

In machine learning, properly handling data is crucial for developing models that generalise well to unseen examples.

**Training set** is the portion of data used to train the model. The model learns patterns from this data to make predictions.

**Test set** is a separate portion of data used to evaluate the model's performance on unseen examples. This provides an estimate of how well the model will perform in real-world applications.

The fundamental principle is that **the test set must remain completely separate from the training process** to provide an unbiased evaluation of model performance.

#### Why Separate Training and Testing Data?

Training models on the same data used for evaluation leads to **overfitting** - the model memorises the training data rather than learning general patterns, resulting in:
- Excellent performance on training data
- Poor performance on new, unseen data

![image](https://github.com/user-attachments/assets/11261cc8-d3e3-41b3-b13e-f1ad6501ed36)

> Shows visual representation of overfitting where a complex model fits training data perfectly but fails to generalise}}

#### Splitting Proportions

A typical split of the available data is:
- **Training set**: 80% of the data
- **Validation set** (also called development set): 10% of the data
- **Test set**: 10% of the data

The exact proportions may vary depending on the size of the dataset and specific requirements of the project.

### **5.1.2 Cross-validation**

**Cross-validation** is a technique to assess how well a model will generalise to an independent dataset, particularly useful when the amount of available data is limited.

#### K-Fold Cross-Validation

In **k-fold cross-validation**:
1. Data is divided into k equal subsets (folds)
2. For each of k iterations:
   - Train the model on k-1 folds
   - Test on the remaining fold
3. Average the performance across all k iterations

The final performance metric is the average across all k test folds, providing a more robust estimate of model performance than a single train/test split.

Common values for k are 5 and 10, which provide a good balance between bias and variance in the performance estimate.

![image](https://github.com/user-attachments/assets/28df0851-b1ae-43ea-b933-d61f989eb629)

> Shows the k-fold cross-validation process with data divided into k parts}}

#### Leave-One-Out Cross-Validation

**Leave-one-out cross-validation (LOOCV)** is an extreme case where k equals the number of data points n:
- Train on n-1 examples
- Test on the single remaining example
- Repeat for all n examples

This approach:
- Makes maximum use of the available data
- Provides nearly unbiased estimates of model performance
- Can be computationally expensive for large datasets

#### Model Selection with Cross-Validation

Cross-validation is often used for:
1. **Model selection**: Choosing between different types of models
2. **Hyperparameter tuning**: Finding optimal parameters for a given model

When using cross-validation for model selection:
1. Split data into training, validation, and test sets
2. Use cross-validation on the training set to evaluate different models/parameters
3. Select the best model based on cross-validation performance
4. Evaluate final performance on the unseen test set

![image](https://github.com/user-attachments/assets/d1cf7e58-8338-450e-8d2f-088ef8cbed50)

> Illustrates the nested cross-validation process for model selection}}

## **5.2 Model Evaluation**

### **5.2.1 Accuracy and Error Rates**

**Accuracy** measures the proportion of correctly classified examples:

$$\text{Accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}}$$

**Misclassification rate** (or **error rate**) is the complement of accuracy:

$$\text{Error Rate} = \frac{\text{number of incorrect predictions}}{\text{total number of predictions}} = 1 - \text{Accuracy}$$

For a classifier $F$ and dataset $D$, the error rate can be written as:

$$\text{err}(F, D) = \frac{1}{N}\sum_{i=1}^{N}I(F(x_i) \neq y_i)$$

where:
- $N$ is the number of examples
- $I$ is an indicator function that equals 1 when the prediction is incorrect and 0 otherwise

#### Limitations of Accuracy

While accuracy is intuitive, it has important limitations:
- **Class imbalance problem**: In datasets with uneven class distributions, a model that always predicts the majority class can achieve high accuracy without learning anything useful
- **Different error costs**: In many applications, different types of errors have different costs (e.g., false negatives in medical diagnosis)

**Example**: In a dataset with 70% negative examples and 30% positive examples, a classifier that always predicts "negative" would achieve 70% accuracy without any predictive value.

### **5.2.2 Precision and Recall**

For binary classification problems (with "positive" and "negative" classes):

**Precision** measures the proportion of positive predictions that are actually correct:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

where:
- TP (True Positives): Correctly predicted positive examples
- FP (False Positives): Negative examples incorrectly predicted as positive

Precision answers: "Of all instances predicted as positive, how many are actually positive?"

**Recall** (also called **sensitivity** or **true positive rate**) measures the proportion of actual positives that are correctly identified:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

where:
- FN (False Negatives): Positive examples incorrectly predicted as negative

Recall answers: "Of all actual positive instances, how many did we correctly identify?"

#### Precision-Recall Trade-off

There is typically a trade-off between precision and recall:
- Increasing precision often reduces recall
- Increasing recall often reduces precision

The optimal balance depends on the specific application:
- **High precision priority**: Legal document search, spam filtering
- **High recall priority**: Cancer screening, fraud detection

### **5.2.3 F-Score**

The **F-score** (or **F-measure**) combines precision and recall into a single metric. The most common version is the **F1 score**, which is the harmonic mean of precision and recall:

$$F_1 = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$$

The general form is the **Fβ score**:

$$F_\beta = (1 + \beta^2) \times \frac{\text{precision} \times \text{recall}}{\beta^2 \times \text{precision} + \text{recall}}$$

where:
- β controls the weight of precision and recall
- β = 1 gives equal weight (F1 score)
- β = 2 gives more weight to recall (F2 score)
- β = 0.5 gives more weight to precision (F0.5 score)

**Example calculation**:
If precision = 0.8 and recall = 0.6:

$$F_1 = 2 \times \frac{0.8 \times 0.6}{0.8 + 0.6} = 2 \times \frac{0.48}{1.4} = 0.69$$

### **5.2.4 Confusion Matrix**

A **confusion matrix** is a table that visualizes the performance of a classification algorithm by showing the counts of true and predicted classes.

For binary classification:

|               | Actual Positive | Actual Negative |
|---------------|-----------------|-----------------|
| **Predicted Positive** | True Positive (TP) | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) | True Negative (TN) |

From the confusion matrix, various metrics can be calculated:
- **Accuracy**: (TP + TN) / (TP + FP + FN + TN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **Specificity** (true negative rate): TN / (TN + FP)
- **F1 score**: 2 × (precision × recall) / (precision + recall)

**Example**:
Consider a classifier that produces this confusion matrix:

|               | Actual Positive | Actual Negative |
|---------------|-----------------|-----------------|
| **Predicted Positive** | 80 | 20 |
| **Predicted Negative** | 10 | 90 |

We can calculate:
- Accuracy = (80 + 90) / 200 = 0.85
- Precision = 80 / 100 = 0.8
- Recall = 80 / 90 = 0.89
- Specificity = 90 / 110 = 0.82
- F1 score = 2 × (0.8 × 0.89) / (0.8 + 0.89) = 0.84

### **5.2.5 ROC Curve**

The **Receiver Operating Characteristic (ROC) curve** plots the true positive rate (recall) against the false positive rate at various classification thresholds.

**False Positive Rate (FPR)** is defined as:

$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{Specificity}$$

Most classifiers output a probability or score that is then thresholded to make a binary decision:

$$\text{classification} = 
\begin{cases}
\text{positive,} & \text{if } p(y_i|x_i, D) > \text{threshold} \\
\text{negative,} & \text{otherwise}
\end{cases}$$

The ROC curve is created by plotting TPR vs FPR at different threshold values.

**Key characteristics of the ROC curve**:
- The diagonal line represents the performance of a random classifier
- The closer the curve follows the left and top borders, the better the classifier
- The **Area Under the Curve (AUC)** is a single number that summarizes the overall performance:
  - AUC = 0.5: No discriminative ability (random)
  - AUC = 1.0: Perfect classification
  - Typically, AUC values between 0.7 and 0.9 are considered good

![image](https://github.com/user-attachments/assets/edf59543-b4e9-4baa-80c4-39fddf385f46)

> Shows an example ROC curve with the random classifier diagonal line and a better classifier curve above it}}
