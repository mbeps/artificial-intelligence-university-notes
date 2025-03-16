# 1. Introduction to Inductive Learning

## 1.1 Core Concepts

**Inductive learning** is the process of learning from examples. It is a fundamental approach in machine learning where patterns and models are derived from specific observations or training data.

In supervised learning, inductive learning follows this general approach:
1. Gather labelled examples (training data)
2. Extract features from these examples
3. Train a model to predict the labels using these features
4. Use the trained model to predict labels for new, unseen data

![image](https://github.com/user-attachments/assets/8bf2b389-dba6-4e6b-865f-7ecd1c70a65f)

The main goal of inductive learning is to generalise from specific examples to create a model that can make accurate predictions on unseen data. This is essential for practical applications of machine learning.

Common tasks in inductive learning include:

- **Classification**: Categorising examples into discrete classes (e.g., determining whether an iris flower is setosa, virginica, or another species)
- **Regression**: Predicting continuous values (e.g., predicting income levels)
- **Optimisation**: Finding the "best" model parameters that minimise errors

## 1.2 Inductive Learning Hypothesis

The **inductive learning hypothesis** states:

> Any hypothesis found to approximate the target function well over a sufficiently large set of training examples will also approximate the target function well over other unobserved examples.

This hypothesis forms the theoretical foundation for supervised machine learning. It suggests that if we can find a model that performs well on our training data, it will likely perform well on new, unseen data drawn from the same distribution.

However, this hypothesis assumes:
1. The training examples are representative of the overall data distribution
2. The target function is consistent (not random)
3. We have sufficient training examples to capture the underlying patterns

When these assumptions are violated, issues like **overfitting** can occur. Overfitting happens when a model learns the training data too precisely, including its noise and peculiarities, resulting in poor performance on new data.

The challenge in inductive learning is to find the right balance - creating models that capture the underlying patterns in the data without memorising the training examples. This balance is often achieved through techniques like cross-validation, regularisation, and appropriate model selection.

# 2. Decision Trees

## 2.1 Basic Structure

**Decision trees** are a popular supervised learning method used for both classification and regression tasks. They represent decisions and their possible consequences in a tree-like structure.

The key components of a decision tree include:

- **Node**: A test of the value of a feature for a data point
- **Branch**: The outcome of a test, leading to another node or leaf
- **Leaf node**: Terminal node that specifies the prediction value (class label or numerical value)
- **Root**: The topmost node of the tree, representing the first decision point

Decision trees can be viewed as implementing a **disjunction of conjunctions** of constraints on feature values. In simpler terms, they represent a series of "if-then-else" rules that can be easily understood and interpreted.

![image](https://github.com/user-attachments/assets/d3403e23-4671-4dad-9332-130f5a13c2b5)

## 2.2 Building Decision Trees

The general process for building a decision tree from a set of examples (known as **Decision Tree Learning** or **DTL**) follows these steps:

```
function DTL(examples, attributes, parent_examples) returns a decision tree
    if examples is empty then return Plurality-Value(parent_examples)
    else if all examples have the same classification then return the classification
    else if attributes is empty then return Plurality-Value(examples)
    else
        best ← Most-Important-Attribute(attributes, examples)
        tree ← a new decision tree with root test best
        for each value vi of best do
            examplesi ← {elements of examples with best = vi}
            remain ← attributes - best
            subtree ← DTL(examplesi, remain, examples)
            add a branch to tree with label vi and subtree subtree
        return tree
```

When building a decision tree, several special cases can arise:

1. If all remaining examples have the same classification, we're done and can return that classification.
2. If all remaining examples are negative, we're done and can return "no."
3. If examples include both positive and negative cases, we choose the best feature to split them.
4. If there are no examples left, we return a default value based on the parent node (plurality classification).
5. If there are no features left but mixed examples, we return a plurality classification.

**Plurality classification** means making the best guess based on the parent node. This could be the majority class, a random pick weighted by the ratio of examples, or a probability.

## 2.3 Choosing Attributes

### 2.3.1 Entropy and Information Gain

**Entropy** is a measure of uncertainty or impurity in a set of examples. For a probability distribution with c classes, entropy is calculated as:

$$H(S) = \sum_{i=1}^{c} -p_i \log_2 p_i$$

Where:
- $p_i$ is the proportion of examples belonging to class $i$
- $S$ is the set of examples

For a binary classification problem with p positive and n negative examples, the entropy is:

$$H(S) = -\frac{p}{p+n} \log_2 \frac{p}{p+n} - \frac{n}{p+n} \log_2 \frac{n}{p+n}$$

**Information gain** measures how much a feature reduces entropy when used for splitting examples. It is calculated as:

$$Gain(S, A) = H(S) - \sum_{i} \frac{|S_i|}{|S|} H(S_i)$$

Where:
- $S$ is the original set of examples
- $A$ is the attribute being evaluated
- $S_i$ is the subset of examples where attribute $A$ takes the value $i$
- $|S_i|$ is the size of subset $S_i$
- $|S|$ is the size of the original set

#### Example: Information Gain Calculation

Consider this simple example from the restaurant waiting problem:

Attributes: Patrons (None, Some, Full), Type (French, Thai, Burger, Italian)
Target: WillWait (Yes/No)

For a set with 6 Yes and 6 No examples (total entropy = 1):

For Patrons:
- None: 0 Yes, 2 No
- Some: 4 Yes, 0 No
- Full: 2 Yes, 4 No

Calculating the information gain:
$$Gain(Patrons) = 1 - ((2/12 × 0) + (4/12 × 0) + (6/12 × 0.918)) = 0.541$$

For Type, the information gain would be 0 because knowing the restaurant type provides no information about whether we'll wait.

Therefore, Patrons is a better attribute choice than Type for the first split.

### 2.3.2 Gain Ratio

**Information gain** has a bias toward attributes with many values, which might lead to overfitting. For example, attributes like "RestaurantName" or precise "Time" would have high information gain but poor generalization.

**Gain ratio** addresses this by penalizing attributes with many values:

1. Calculate **split information**:
   $$SplitInformation(S, A) = -\sum_{i} \frac{|S_i|}{|S|} \log_2 \frac{|S_i|}{|S|}$$

2. Calculate **gain ratio**:
   $$GainRatio(S, A) = \frac{Gain(S, A)}{SplitInformation(S, A)}$$

The attribute with the highest gain ratio is selected for splitting.

### 2.3.3 Gini Impurity

**Gini impurity** is an alternative to entropy for measuring the quality of a split. It represents the probability of incorrectly classifying a randomly chosen element if it was randomly labeled according to the distribution of classes in the subset.

For a set with c classes, Gini impurity is calculated as:

$$G(S) = 1 - \sum_{i=1}^{c} p_i^2$$

Where $p_i$ is the proportion of examples belonging to class $i$.

For a binary classification with p positive and n negative examples:

$$G(S) = 1 - \left(\left(\frac{p}{p+n}\right)^2 + \left(\frac{n}{p+n}\right)^2\right)$$

When evaluating an attribute A that splits set S into subsets $S_i$:

$$G(S, A) = \sum_{i} \frac{|S_i|}{|S|} G(S_i)$$

The attribute with the lowest Gini impurity is selected for splitting.

#### Example: Gini Impurity Calculation

Using the same restaurant example:

For Patrons:
- None: $G(None) = 1 - ((0/2)^2 + (2/2)^2) = 0$
- Some: $G(Some) = 1 - ((4/4)^2 + (0/4)^2) = 0$
- Full: $G(Full) = 1 - ((2/6)^2 + (4/6)^2) = 0.444$

Overall Gini impurity:
$$G(Patrons) = (2/12 × 0) + (4/12 × 0) + (6/12 × 0.444) = 0.222$$

## 2.4 Overfitting and Pruning

**Overfitting** occurs when a decision tree grows to perfectly classify all training examples but fails to generalize to new data. A tree d is said to overfit if:
1. d has a smaller error than another tree d' on the training data
2. d' has a smaller error than d on all other instances

There are two main approaches to prevent overfitting:
1. Stop growing the tree early
2. Allow the tree to grow and then prune it back

The second approach (pruning) typically yields better results.

### 2.4.1 Reduced Error Pruning

**Reduced error pruning** works as follows:

1. Consider every branch in the tree as a candidate for pruning
2. Remove the branch and make its root into a leaf with the plurality classification
3. Test the modified tree on a validation set
4. Keep the branch pruned if the pruned tree performs better
5. Repeat until no further improvement is possible

This method requires dividing the data into three sets:
- Training data (to build the initial tree)
- Pruning data/validation set (to evaluate pruning)
- Test data (for final evaluation)

This approach can be challenging when data is limited.

### 2.4.2 Rule Post-Pruning

**Rule post-pruning** addresses the data limitation problem of reduced error pruning:

1. Build the complete decision tree
2. Convert the tree into a set of rules (one rule for each path from root to leaf)
3. Remove preconditions (feature tests) from rules if that improves their accuracy
4. Sort rules by accuracy and apply them in sequence when classifying examples

For example, a rule like "If (Patrons = Full) AND (Hungry = No) Then ¬Wait" might be simplified to "If (Hungry = No) Then ¬Wait" if removing the Patrons condition improves or maintains accuracy.

Advantages of rule post-pruning:
- More flexible than pruning branches
- Can remove part of a branch
- Can remove the root while keeping lower branches

## 2.5 DTL Variants

Several notable implementations of the decision tree learning algorithm exist:

**ID3** is the basic algorithm that uses entropy for attribute selection.

**C4.5** extended ID3 by adding post-processing to simplify the tree and handling continuous attributes and missing values.

**CART (Classification and Regression Trees)** is another popular implementation that can handle both classification and regression tasks. It uses Gini impurity for classification trees.

While these algorithms evolved somewhat independently, they converged on similar solutions for common challenges in tree induction.

# 3. Linear Regression

## 3.1 Univariate Linear Regression

**Linear regression** is a fundamental supervised learning technique that models the relationship between a dependent variable and one or more independent variables using a linear equation.

In **univariate linear regression**, we model the relationship between a single input feature and the output. The equation is of the form:

$$h_w(x) = w_1x + w_0$$

Where:
- $h_w(x)$ is the predicted output (hypothesis)
- $x$ is the input feature
- $w_0$ is the intercept (bias term)
- $w_1$ is the slope (coefficient/weight)
- The subscript $w$ indicates the vector $[w_0, w_1]$

The goal is to estimate values of $w_0$ and $w_1$ from training data that best fit the relationship between inputs and outputs.

![image](https://github.com/user-attachments/assets/1d5baa98-e742-438d-9c9a-bb7c33b84d85)

To find the best-fitting line, we need to minimise the **loss function** (also called the **cost function**). Traditionally, we use the **squared loss function** ($L_2$):

$$J(h_w) = Loss(h_w) = \sum_{j=1}^{N} (y_j - h_w(x_j))^2 = \sum_{j=1}^{N} (y_j - (w_1x_j + w_0))^2$$

Where:
- $N$ is the number of training examples
- $(x_j, y_j)$ are the training examples
- $y_j$ is the actual output
- $h_w(x_j)$ is the predicted output

Our optimisation objective is:

$$w^* = \arg\min_w Loss(h_w)$$

The squared loss function is used because Gauss showed that for normally distributed noise, this gives the most likely values of the weights.

## 3.2 Multivariate Linear Regression

**Multivariate linear regression** extends univariate regression to multiple input features. Instead of a single variable $x$, we now have multiple variables $x_{j,1}, x_{j,2}, ..., x_{j,n}$ with corresponding weights $w_1, w_2, ..., w_n$.

To simplify notation, we create a dummy attribute $x_{j,0} = 1$ to pair with $w_0$. The hypothesis function becomes:

$$h_w(x_j) = \sum_{i=0}^{n} w_i x_{j,i}$$

This can be expressed more compactly in vector notation as:

$$h_w(x) = w^T x$$

Where $w$ and $x$ are vectors of weights and features, respectively.

Learning in multivariate linear regression follows the same principles as univariate regression but requires adjusting multiple weights during training.

While multivariate regression is more powerful, it's also more prone to **overfitting**, especially when dealing with many features. To address this, we use **regularisation** techniques (discussed in section 3.4).

## 3.3 Gradient Descent

**Gradient descent** is an optimisation algorithm used to find the minimum of a function. In the context of linear regression, it's used to find the weights that minimise the loss function.

The basic idea is:
1. Start with some initial values for the weights
2. Iteratively adjust these weights to reduce the loss function
3. Continue until convergence (minimum is reached)

For each weight $w_i$, the update rule is:

$$w_i \leftarrow w_i - \alpha \frac{\partial}{\partial w_i} Loss(w)$$

Where:
- $\alpha$ is the **learning rate** that controls the step size
- $\frac{\partial}{\partial w_i} Loss(w)$ is the partial derivative of the loss function with respect to $w_i$

![image](https://github.com/user-attachments/assets/5b17dcd5-4dcb-4b69-884f-ce1f3bbd30cd)

### 3.3.1 Batch Gradient Descent

**Batch gradient descent** considers all training examples simultaneously when updating weights. The update rules are:

$$w_0 \leftarrow w_0 + \alpha \sum_j (y_j - h_w(x_j))$$
$$w_1 \leftarrow w_1 + \alpha \sum_j (y_j - h_w(x_j))x_j$$

For multivariate regression, the general update rule is:

$$w_i \leftarrow w_i + \alpha \sum_j (y_j - h_w(x_j))x_{j,i}$$

Advantages of batch gradient descent:
- Guaranteed to converge (with appropriate learning rate)
- More stable updates

Disadvantages:
- Can be slow for large datasets as it processes all N examples in each iteration

### 3.3.2 Stochastic Gradient Descent

**Stochastic gradient descent (SGD)** updates weights after processing each individual training example, rather than waiting to process the entire dataset:

$$w_0 \leftarrow w_0 + \alpha(y - h_w(x))$$
$$w_1 \leftarrow w_1 + \alpha(y - h_w(x))x$$

For multivariate regression:

$$w_i \leftarrow w_i + \alpha(y_j - h_w(x_j))x_{j,i}$$

Advantages of stochastic gradient descent:
- Often faster, especially for large datasets
- Can help escape local minima due to the noise in updates
- Well-suited for online learning scenarios

Disadvantages:
- May not converge with a constant learning rate
- Updates are noisier

To make SGD converge, the learning rate is typically decreased over time.

#### Example: Gradient Descent Calculation

Consider a dataset with the following points:
- $(1.5, 1)$, $(3.5, 3)$, $(3, 2)$, $(5, 3)$, $(2, 2.5)$

Starting with $w_0 = 0$, $w_1 = 0$, and $\alpha = 0.01$:

For batch gradient descent (first iteration):
- Calculate error for each point: $1, 3, 2, 3, 2.5$ (total = 11.5)
- Update $w_0 = 0 + 0.01 \times 11.5 = 0.115$
- Calculate weighted error: $1.5, 10.5, 6, 15, 5$ (total = 38)
- Update $w_1 = 0 + 0.01 \times 38 = 0.38$

For stochastic gradient descent (first example):
- Calculate error: $1 - (0 + 0 \times 1.5) = 1$
- Update $w_0 = 0 + 0.01 \times 1 = 0.01$
- Update $w_1 = 0 + 0.01 \times 1 \times 1.5 = 0.015$

## 3.4 Regularisation

**Regularisation** is a technique used to prevent overfitting in linear regression models, especially when dealing with many features. It adds a penalty for complexity to the loss function:

$$Loss'(h) = Loss(h) + \lambda \cdot Complexity(h)$$

Where:
- $\lambda$ is the **regularisation parameter** that controls the trade-off between fitting the data and keeping the model simple
- $Complexity(h)$ is a measure of model complexity

The most common form of regularisation is **L2 regularisation** (also known as Ridge regression), where the complexity is measured as the sum of squared weights:

$$Complexity(h_w) = \sum_i w_i^2$$

This results in the regularised loss function:

$$Loss'(h_w) = \sum_{j=1}^{N} (y_j - h_w(x_j))^2 + \lambda \sum_i w_i^2$$

L2 regularisation encourages smaller weights, which typically leads to simpler models that generalise better. Intuitively, if you consider a polynomial function like $w_0 + w_1x + w_2x^2 + w_3x^3$, having smaller values for $w_2$ and $w_3$ reduces the impact of higher-order terms, making the function closer to linear.

If the regularisation parameter $\lambda$ is:
- Too high: The model will be too simple and may underfit the data
- Too low: The model may still overfit the data
- Just right: The model will balance fitting the training data and generalising to new data

When using regularisation, the gradient descent update rules incorporate the regularisation term's derivative, which typically adds a term that shrinks each weight slightly at each update.

Regularisation is essential for multivariate linear regression models, especially when:
1. The number of features is large
2. Features are correlated with each other
3. The amount of training data is limited

# 5. Ensemble Methods

## 5.1 Basic Ensemble Learning

**Ensemble learning** involves combining multiple models to improve prediction performance beyond what could be achieved by any single model. The fundamental principle is that a group of "weak learners" can come together to form a "strong learner."

Every classifier has an error rate, meaning it will inevitably misclassify some examples. Ensemble methods capitalise on the idea that different models will often make different mistakes, so combining their predictions can reduce overall error.

The simplest ensemble approach works as follows:
1. Train N different classifiers
2. Use all classifiers to predict the class for a new example
3. Have the classifiers vote on the final classification
4. Take the majority prediction as the ensemble output

For example, if we have 5 binary classifiers, each with a 10% error rate, and assuming their errors are independent, the probability that a majority will make the wrong prediction drops to less than 1%.

The key to successful ensemble learning is diversity among the base models. If all models make identical errors, combining them yields no benefit. Therefore, various techniques are used to ensure diversity, including:
- Training models on different subsets of data
- Using different algorithms
- Using different feature subsets
- Introducing randomness in the training process

## 5.2 Boosting

**Boosting** is an ensemble technique that builds models sequentially, with each new model focusing on the examples that previous models misclassified.

The core idea is to use a **weighted training set**, where examples are assigned weights based on how difficult they are to classify. Higher-weighted examples are considered more important during training.

The general boosting process works as follows:

1. Start with all examples having equal weight and learn an initial classifier h₁
2. Test the classifier and increase weights for misclassified examples
3. Learn a new classifier h₂ using the updated weights
4. Repeat the process for a predetermined number of iterations
5. The final ensemble combines all classifiers, weighted by their performance on the training set

![image](https://github.com/user-attachments/assets/d9051116-9e20-4daa-aeaa-135fc3e62869)

**AdaBoost** (Adaptive Boosting) is one of the most popular boosting algorithms:
1. Given a weak learner (performing slightly better than random guessing), AdaBoost can generate an ensemble that perfectly classifies the training set
2. It automatically determines the optimal weights for combining classifiers
3. It's less prone to overfitting than many other algorithms

The mathematical framework behind AdaBoost involves minimising an exponential loss function and adjusting example weights based on classification errors.

## 5.3 Bagging

**Bagging** (Bootstrap Aggregation) is another ensemble technique that trains multiple models on different random subsets of the training data.

The bagging process works as follows:
1. From a training set D, create multiple training sets D₁, D₂, ..., Dₙ by sampling with replacement (bootstrap sampling)
2. Train a separate classifier on each of these datasets
3. Combine the models' predictions through voting (for classification) or averaging (for regression)

The key properties of bagging include:
- Training sets are sampled from the original data with replacement, creating datasets with the same size but different composition
- Each bootstrap sample typically contains about 63.2% of the original examples (some appear multiple times, others not at all)
- The remaining 36.8% of examples (called "out-of-bag" samples) can be used for validation
- Bagging reduces variance and helps avoid overfitting
- It works particularly well with "high-variance" models like decision trees

## 5.4 Random Forests

**Random forests** are a popular ensemble method that combines bagging with an additional layer of randomness.

A random forest works as follows:
1. Create multiple decision trees using bagging (different bootstrap samples)
2. When growing each tree, at each node, randomly select a subset of features to consider for splitting
3. Grow each tree to its maximum size without pruning
4. For classification, use majority voting across all trees; for regression, average the predictions

The key innovation in random forests is the **random subspace method** (feature bagging):
- At each node, only a random subset of features is considered for splitting
- This adds another layer of diversity beyond what bagging alone provides
- Typically, for a dataset with m features, about √m features are considered at each split

Random forests offer several advantages:
- They generally outperform single decision trees and even basic bagging
- They're less prone to overfitting
- They can handle high-dimensional data effectively
- They provide estimates of feature importance
- They can be parallelised easily since each tree is built independently

The combination of bagging (at the example level) and random feature selection (at the node level) creates highly diverse trees, making random forests one of the most powerful and widely used ensemble methods in practice.

