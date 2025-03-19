# **Kernel Machines**

## **1. Introduction to Kernel Machines**

### **1.1 Motivation and Background**

Kernel machines represent a powerful class of algorithms in machine learning that can efficiently learn complex, non-linear decision boundaries while maintaining computational tractability. They emerged as a solution to the limitations inherent in simpler classification methods.

**Key motivations for kernel machines:**

- The need for algorithms that can handle **non-linear classification problems**
- Desire for **efficient training** procedures that avoid local minima issues
- Focus on minimizing **generalization loss** rather than just training error
- Creating models with **better predictive performance** on unseen data

![image](https://github.com/user-attachments/assets/f48fd537-79a8-4ebe-914c-2c8f45f388f0)

> Image showing SVM separating food items

Kernel machines bridge the gap between simple linear models that are easy to train but limited in capability, and complex models like neural networks that can represent sophisticated functions but can be difficult to optimize.

### **1.2 Linear Classifiers Review**

**Linear classifier**: A model that makes classification decisions based on a linear combination of input features.

The basic form of a linear classifier is:
$$ y = \mathbf{w}^T\mathbf{x} + b $$

Where:
- $\mathbf{w}$ is the weight vector
- $\mathbf{x}$ is the input feature vector
- $b$ is the bias term (intercept)
- $y$ is the output (classification)

![image](https://github.com/user-attachments/assets/5cfee292-3b32-4be3-a6bc-77c82beb5b1f)

> Diagrams showing three possible linear classifiers

For binary classification, the decision rule is typically:
- If $\mathbf{w}^T\mathbf{x} + b > 0$, classify as positive class
- If $\mathbf{w}^T\mathbf{x} + b < 0$, classify as negative class

**Training a linear classifier** typically involves:
1. Defining a loss function that measures classification error
2. Using gradient descent or other optimization methods to find optimal weights
3. Minimizing expected empirical loss on training data

### **1.3 Limitations of Linear Classifiers**

Despite their simplicity and interpretability, linear classifiers suffer from fundamental limitations:

**1. Limited decision boundaries**: Linear classifiers can only create linear decision boundaries (straight lines in 2D, planes in 3D, hyperplanes in higher dimensions).

![image](https://github.com/user-attachments/assets/65079311-50be-49df-925f-5f3417179e63)

> Diagram showing data that cannot be separated by a linear boundary

**2. Poor generalization**: A linear classifier that perfectly separates training data might be too close to some examples, making it sensitive to slight variations and leading to poor generalization.

**3. Inability to solve XOR-like problems**: Some logical functions like XOR cannot be represented by a single linear boundary.

**4. Real-world complexity**: Most real-world data involves complex, non-linear relationships that cannot be captured by linear models.

Potential solutions to these limitations include:

- **Neural networks**: Combining many linear classifiers can learn complex boundaries, but these can be difficult to train due to many local minima/maxima in the high-dimensional weight space.

- **Feature engineering**: Manually transforming the input space to make the problem linearly separable, which requires domain expertise and can be time-consuming.

- **Kernel methods**: Using the "kernel trick" to implicitly map data to higher dimensions where it becomes linearly separable, without actually computing the transformation - this approach forms the foundation of kernel machines like Support Vector Machines.

The advantage of kernel machines over alternative approaches is that they provide a mathematically elegant way to learn non-linear decision boundaries while maintaining computational efficiency through the kernel trick.


# **Kernel Machines**

## **2. Support Vector Machines (SVMs)**

### **2.1 Maximum Margin Classifiers**

**Support Vector Machines (SVMs)** are supervised learning algorithms that find the optimal hyperplane to separate data points of different classes. What makes SVMs special is their focus on maximizing the margin.

**Maximum margin classifier**: A classifier that learns a decision boundary with the largest possible separation between classes of data, maintaining equal distance from both classes.

![image](https://github.com/user-attachments/assets/4e0863cd-9bbc-42c4-996e-23fedd7876d8)

> Diagram showing the maximum margin between two classes

The key principle of maximum margin classification:
- Among all possible hyperplanes that separate the data, choose the one with the largest margin
- This strategy aims to minimize **generalization loss** rather than just empirical loss
- A larger margin leads to better performance on unseen data

The **margin** is defined as the width of the area bounded by the closest points from each class to the decision boundary. Mathematically, it's twice the distance from the separator to the nearest data point.

**Benefits of maximum margin classification:**
- Reduces the probability of misclassification on new data
- Provides a unique solution (avoiding multiple equally valid boundaries)
- Less sensitive to small changes in training data
- Makes the model more robust to noise

### **2.2 Finding Optimal Hyperplane**

For a linear classifier with the form $y = \mathbf{w}^T\mathbf{x} + b$, changing the parameters $\mathbf{w}$ and $b$ changes the decision boundary.

![image](https://github.com/user-attachments/assets/629ac263-aeda-470c-bb06-ed32332ae335)

> Diagram showing the weight vector w and decision boundary

In SVMs, we define the decision function as:
- $\mathbf{w}^T\mathbf{x} + b > 0$ for positive class (labeled as +1)
- $\mathbf{w}^T\mathbf{x} + b < 0$ for negative class (labeled as -1)

To find the maximum margin separator, we need to find the $\mathbf{w}$ and $b$ such that:
1. The hyperplane correctly separates the two classes
2. The distance to the closest points (the margin) is maximized

For any point $\mathbf{x}_i$ in our training set with label $y_i \in \{-1, +1\}$, we want:
- $\mathbf{w}^T\mathbf{x}_i + b \geq 0$ when $y_i = +1$
- $\mathbf{w}^T\mathbf{x}_i + b < 0$ when $y_i = -1$

This can be written more compactly as:
$y_i(\mathbf{w}^T\mathbf{x}_i + b) > 0$ for all training examples.

### **2.3 The Optimisation Problem**

SVMs strengthen the constraint by requiring:
$y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ for all $i$

This creates two parallel hyperplanes:
- $\mathbf{w}^T\mathbf{x} + b = +1$ for the positive class boundary
- $\mathbf{w}^T\mathbf{x} + b = -1$ for the negative class boundary

![image](https://github.com/user-attachments/assets/6f78dca5-ecd9-4cd3-aac2-2a5ec2e72133)

> Diagram showing the two hyperplanes $w^Tx+b=1$ and $w^Tx+b=-1$

The distance between these hyperplanes (the margin) is $\frac{2}{||\mathbf{w}||}$, where $||\mathbf{w}||$ is the Euclidean norm of $\mathbf{w}$.

To maximize this margin, we need to minimize $||\mathbf{w}||$, or equivalently $\frac{1}{2}||\mathbf{w}||^2$ (the square is used for mathematical convenience).

Our optimization problem becomes:

$$\min_{\mathbf{w},b} \frac{1}{2}||\mathbf{w}||^2$$

Subject to the constraint:
$$y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, m$$

Where $m$ is the number of training examples.

### **2.4 Lagrangian Formulation**

To solve this constrained optimization problem, we use **Lagrange multipliers**. For each constraint, we introduce a Lagrange multiplier $\alpha_i \geq 0$.

The **Lagrangian** is formulated as:

$$L(\mathbf{w}, b, \alpha) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^{m} \alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]$$

Where:
- $\mathbf{w}$ and $b$ are the parameters we're optimizing
- $\alpha_i$ are the Lagrange multipliers
- $m$ is the number of training examples

To find the minimum, we take the partial derivatives of $L$ with respect to $\mathbf{w}$ and $b$ and set them to zero:

$$\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i} \alpha_i y_i \mathbf{x}_i = 0$$

$$\frac{\partial L}{\partial b} = -\sum_{i} \alpha_i y_i = 0$$

From the first equation, we get:
$$\mathbf{w} = \sum_{i} \alpha_i y_i \mathbf{x}_i$$

This shows that the optimal $\mathbf{w}$ is a linear combination of the training examples, weighted by the Lagrange multipliers.

### **2.5 Dual Representation**

Substituting the expressions for $\mathbf{w}$ into the Lagrangian and simplifying leads to the **dual representation**:

$$\arg\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2}\sum_{i,j=1}^{m} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j)$$

Subject to:
$$\alpha_i \geq 0, \quad \sum_{i=1}^{m} \alpha_i y_i = 0$$

Where:
- $\alpha_i$ are the Lagrange multipliers
- $\mathbf{x}_i \cdot \mathbf{x}_j$ is the dot product between training examples

This is a quadratic programming problem with a unique, global maximum, which can be solved using specialized software packages.

The dual formulation has a key advantage: the training data appears only in the form of dot products $\mathbf{x}_i \cdot \mathbf{x}_j$. This property is what enables the kernel trick (covered in section 3).

### **2.6 Support Vectors**

After solving the optimization problem, many of the Lagrange multipliers $\alpha_i$ will be zero. Only the examples with $\alpha_i > 0$ contribute to the decision boundary.

**Support vectors** are the training examples that lie exactly on the margin boundaries, where $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$. They are called "support vectors" because they "support" or determine the position of the decision boundary.

![image](https://github.com/user-attachments/assets/102cf9c3-607a-45d7-9026-4a01acda7b62)
> Diagram showing support vectors holding up the separator

Key properties of support vectors:
- They are the closest points to the decision boundary
- Only the support vectors matter for making predictions
- The number of support vectors is typically much smaller than the total number of training examples
- Having fewer support vectors reduces the risk of overfitting

### **2.7 Decision Function**

Once we've found the optimal values for $\alpha_i$, we can use them to make predictions for new data points.

The decision function in the dual representation is:

$$h(\mathbf{x}) = \text{sign}\left(\sum_{i} \alpha_i y_i (\mathbf{x} \cdot \mathbf{x}_i) + b\right)$$

Where:
- $\text{sign}$ returns +1 for positive values and -1 for negative values
- The sum only needs to be computed over the support vectors (where $\alpha_i > 0$)
- $b$ can be calculated using any support vector $\mathbf{x}_s$ as: $b = y_s - \sum_{i} \alpha_i y_i (\mathbf{x}_s \cdot \mathbf{x}_i)$

**Example**: 
Suppose we have trained an SVM on a dataset and found three support vectors with the following values:
- $\mathbf{x}_1 = [1, 2]$, $y_1 = +1$, $\alpha_1 = 0.5$
- $\mathbf{x}_2 = [2, 1]$, $y_2 = +1$, $\alpha_2 = 0.3$
- $\mathbf{x}_3 = [0, 0]$, $y_3 = -1$, $\alpha_3 = 0.8$

With $b = 0.2$, to classify a new point $\mathbf{x} = [1.5, 1.5]$:

$$h([1.5, 1.5]) = \text{sign}(0.5 \times 1 \times ([1.5, 1.5] \cdot [1, 2]) + 0.3 \times 1 \times ([1.5, 1.5] \cdot [2, 1]) + 0.8 \times (-1) \times ([1.5, 1.5] \cdot [0, 0]) + 0.2)$$

$$= \text{sign}(0.5 \times 1 \times (1.5 + 3) + 0.3 \times 1 \times (3 + 1.5) + 0.8 \times (-1) \times 0 + 0.2)$$

$$= \text{sign}(0.5 \times 4.5 + 0.3 \times 4.5 + 0.2)$$

$$= \text{sign}(2.25 + 1.35 + 0.2)$$

$$= \text{sign}(3.8)$$

$$= +1$$

Therefore, the new point is classified as belonging to the positive class.


# **Kernel Machines**

## **3. The Kernel Trick**

### **3.1 Feature Space Transformation**

The kernel trick is a fundamental concept that allows SVMs to handle non-linearly separable data efficiently. The core idea is to transform data into a higher-dimensional space where linear separation becomes possible.

**Feature space transformation** refers to mapping input data from its original space to a higher-dimensional feature space where a linear decision boundary can separate classes that weren't linearly separable in the original space.

Consider a simple example where data points can't be separated by a straight line:

![image](https://github.com/user-attachments/assets/c38e8070-cd7e-4e7a-921a-720bf9308367)

> Diagram showing non-linearly separable data in 2D

By transforming this 2D data into a 3D space, linear separation becomes possible:

![image](https://github.com/user-attachments/assets/910fc60d-70bf-4a74-ab48-dde596db06f4)

> Diagram showing the same data now separable in 3D space

For a simple 2D example, we might map each point $\mathbf{x} = (x_1, x_2)$ to a 3D space using:
- $f_1 = x_1^2$
- $f_2 = x_2^2$
- $f_3 = \sqrt{2} x_1 x_2$

In general, we can describe this as a mapping function $\Phi$ that transforms input vector $\mathbf{x}$ to a higher-dimensional vector $\Phi(\mathbf{x})$.

**Important property**: Data will always be linearly separable if mapped to a space with enough dimensions. In fact, N data points will always be linearly separable in N-1 dimensions.

### **3.2 Inner Products and Similarity**

The dot product (inner product) between two vectors provides a measure of their similarity, which is crucial to understanding the kernel trick.

**Inner product**: For two vectors $\mathbf{a}$ and $\mathbf{b}$, the inner product is defined as:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

Where $n$ is the dimensionality of the vectors.

For example, $[1, 2, 3] \cdot [4, 2, 3] = 1 \times 4 + 2 \times 2 + 3 \times 3 = 17$

Properties of inner products related to similarity:
- The inner product is maximum when vectors point in the same direction (θ = 0)
- The inner product is zero when vectors are orthogonal (θ = 90°)
- The inner product's magnitude increases with vector magnitudes

![image](https://github.com/user-attachments/assets/c60d4989-df1d-4075-a9f5-15b114cf7d3f)

> Diagram showing dot product as a measure of similarity

In the SVM dual formulation, the inner product $\mathbf{x}_i \cdot \mathbf{x}_j$ appears extensively. When we transform data points to a higher dimension, we would need to compute $\Phi(\mathbf{x}_i) \cdot \Phi(\mathbf{x}_j)$, which could be computationally expensive if the transformed space has many dimensions.

### **3.3 Kernel Functions**

The **kernel trick** allows us to compute the inner product in the transformed space without explicitly mapping the data to that space.

A **kernel function** $K(\mathbf{x}, \mathbf{z})$ computes the inner product between two transformed feature vectors without explicitly computing the transformation:

$$K(\mathbf{x}, \mathbf{z}) = \Phi(\mathbf{x}) \cdot \Phi(\mathbf{z})$$

Example of the kernel trick:
For a quadratic kernel in 2D space, instead of:
1. Transforming $\mathbf{x}$ to $\Phi(\mathbf{x}) = [x_1^2, x_2^2, \sqrt{2}x_1x_2]$
2. Transforming $\mathbf{z}$ to $\Phi(\mathbf{z}) = [z_1^2, z_2^2, \sqrt{2}z_1z_2]$
3. Computing the inner product $\Phi(\mathbf{x}) \cdot \Phi(\mathbf{z}) = x_1^2z_1^2 + x_2^2z_2^2 + 2x_1x_2z_1z_2$

We can directly compute:
$$K(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z})^2 = (x_1z_1 + x_2z_2)^2 = x_1^2z_1^2 + x_2^2z_2^2 + 2x_1x_2z_1z_2$$

This gives us the same result but is much more efficient, especially when the transformed space has many dimensions.

The SVM dual formulation with kernels becomes:

$$\arg\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2}\sum_{i,j=1}^{m} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

And the decision function becomes:

$$h(\mathbf{x}) = \text{sign}\left(\sum_{i} \alpha_i y_i K(\mathbf{x}, \mathbf{x}_i) + b\right)$$

### **3.4 Popular Kernel Functions**

Several kernel functions are commonly used in practice:

1. **Linear kernel**:
   $$K(\mathbf{x}, \mathbf{z}) = \mathbf{x} \cdot \mathbf{z}$$
   This is equivalent to no transformation and results in a linear SVM.

2. **Polynomial kernel**:
   $$K(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z})^d$$
   or
   $$K(\mathbf{x}, \mathbf{z}) = (1 + \mathbf{x} \cdot \mathbf{z})^d$$
   Where $d$ is the degree of the polynomial. This kernel maps to a feature space that grows exponentially with $d$.

3. **Gaussian kernel** (also known as Radial Basis Function or RBF kernel):
   $$K(\mathbf{x}, \mathbf{z}) = \exp\left(-\frac{||\mathbf{x} - \mathbf{z}||^2}{2\sigma^2}\right)$$
   Where $\sigma$ is a parameter that determines the width of the Gaussian. This kernel maps to an infinite-dimensional feature space.

4. **Sigmoid kernel**:
   $$K(\mathbf{x}, \mathbf{z}) = \tanh(\kappa(\mathbf{x} \cdot \mathbf{z}) + \Theta)$$
   Where $\kappa$ and $\Theta$ are parameters. This kernel is inspired by neural networks.

Empirically, all these kernel functions often give SVM classifiers with similar accuracies and similar sets of support vectors for many problems. The choice of kernel depends on the specific application and data characteristics.

### **3.5 Mercer's Theorem**

Not every function can be used as a valid kernel. **Mercer's theorem** (1909) provides conditions for when a function can be used as a kernel.

A kernel function $K(\mathbf{x}, \mathbf{z})$ is valid if it is **positive definite**, meaning:

$$\sum_{i,j=1}^{n} c_i c_j K(\mathbf{x}_i, \mathbf{x}_j) \geq 0$$

For any $n$, any scalars $c_i, c_j$, and any vectors $\mathbf{x}_i, \mathbf{x}_j$ in the input space.

This mathematical condition ensures that there exists some feature mapping $\Phi$ such that $K(\mathbf{x}, \mathbf{z}) = \Phi(\mathbf{x}) \cdot \Phi(\mathbf{z})$.

Mercer's theorem is important because:
- It guarantees that the optimization problem has a unique solution
- It ensures that the kernel represents a valid inner product in some feature space
- It allows us to create new kernels by combining existing ones (e.g., through addition or multiplication)

The feature spaces associated with kernels can be extremely high-dimensional or even infinite-dimensional (as with the Gaussian kernel), making the kernel trick essential for computational tractability.


# **Kernel Machines**

## **4. Soft Margin Classification**

### **4.1 Handling Non-Separable Data**

In real-world scenarios, data is often noisy and cannot be perfectly separated by a hyperplane, even after mapping to a higher-dimensional space using kernels. **Soft margin classification** extends SVMs to handle non-separable data.

![image](https://github.com/user-attachments/assets/d3dc5bc0-eaf0-43f7-a727-85d54ccfa38f)

> Showing a non-separable dataset

**Motivation for soft margin classification:**
- Perfect separation might be impossible due to noise or outliers
- Even with kernel transformations, some datasets remain non-separable
- Forcing perfect separation could lead to overfitting
- A more flexible approach is needed to handle practical datasets

As explained in the tutorial answers:
> "When there is noise in the training data, it may be impossible to find a maximum margin classifier that separates the two classes. The idea of a soft margin classifier allows us to still create an SVM classifier under these conditions, by relaxing the conditions on what counts as a good classifier."

The key insight of soft margin classification is to allow some misclassifications during training to achieve better generalization on test data.

### **4.2 Slack Variables**

**Slack variables** ($\xi_i$) are introduced to relax the constraints of the original SVM formulation, allowing some points to violate the margin boundary or even be misclassified.

In the original SVM formulation, we required:
$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$$

With slack variables, this constraint becomes:
$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i$$

Where:
- $\xi_i \geq 0$ for all $i = 1, \ldots, m$
- $\xi_i = 0$ for points that satisfy the original constraint
- $0 < \xi_i \leq 1$ for points that lie inside the margin but are correctly classified
- $\xi_i > 1$ for misclassified points

The meaning of the slack variables can be interpreted as:
- They measure the degree of misclassification
- They allow points to be on the "wrong side" of the margin
- The sum $\sum_{i=1}^{m} \xi_i$ provides an upper bound on the number of training errors

Example:
Consider a point $\mathbf{x}_i$ with label $y_i = +1$:
- If $\xi_i = 0$: The point is correctly classified and outside the margin
- If $\xi_i = 0.3$: The point is correctly classified but inside the margin
- If $\xi_i = 1$: The point is exactly on the decision boundary
- If $\xi_i = 1.5$: The point is misclassified (on the wrong side of the boundary)

### **4.3 Regularisation Parameter C**

With slack variables introduced, the SVM optimization problem becomes:

$$\min_{\mathbf{w}, \xi} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{m} \xi_i$$

Subject to:
$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i \quad \text{and} \quad \xi_i \geq 0 \quad \text{for all } i = 1, \ldots, m$$

The constant **C > 0** is called the **regularisation parameter** (or penalty parameter) and controls the trade-off between:
- Maximizing the margin (which requires more slack)
- Minimizing training errors (which requires less slack)

Effects of different C values:
- **Large C**: Places high penalty on errors; results in a narrower margin that tries to classify all training points correctly but risks overfitting
- **Small C**: Allows more errors; results in a wider margin that might misclassify more training points but may generalize better

In practice, the optimal value of C is typically determined through cross-validation, where different values are tested to find which one gives the best performance on validation data.

When we reformulate the problem in terms of the dual representation (using Lagrange multipliers), the optimization problem becomes:

$$\arg\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2}\sum_{i,j=1}^{m} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

Subject to:
$$0 \leq \alpha_i \leq C \quad \text{for all } i = 1, \ldots, m \quad \text{and} \quad \sum_{i=1}^{m} \alpha_i y_i = 0$$

The key difference from the standard SVM is the additional constraint $\alpha_i \leq C$, which limits the influence of individual training points.

### **4.4 Impact on Support Vectors**

The introduction of slack variables and the regularisation parameter C has significant effects on the selection and role of support vectors in the model.

In soft margin SVMs, we can categorize the support vectors into three types based on their Lagrange multipliers $\alpha_i$:

1. **Margin support vectors**: Points exactly on the margin ($\xi_i = 0$ and $0 < \alpha_i < C$)
2. **Bounded support vectors**: Points inside the margin or misclassified ($\xi_i > 0$ and $\alpha_i = C$)
3. **Non-support vectors**: Points correctly classified and outside the margin ($\xi_i = 0$ and $\alpha_i = 0$)

The upper bound C on the Lagrange multipliers has the following effects:

- It **limits the influence** of individual training points on the decision boundary
- It **prevents outliers** from having a disproportionate effect on the classifier
- It makes the model more **robust to noise**

Example:
Consider a dataset with an outlier far from other points of its class. In a hard-margin SVM, this outlier would significantly distort the decision boundary. In a soft-margin SVM with an appropriate C value, the outlier would be assigned $\alpha_i = C$, limiting its influence and allowing a more generalizable boundary.

![image](https://github.com/user-attachments/assets/455f45bc-d65e-433d-8148-e3d3f26e11da)

> Diagram showing how soft margin classification handles outliers

The key insight is that soft margin classification allows the SVM to ignore points that would otherwise lead to poor generalization, effectively focusing on the most informative examples while limiting the impact of noisy or outlier data points.

In summary, soft margin classification extends the applicability of SVMs to real-world, noisy datasets by introducing flexibility in the constraints and controlling the trade-off between margin size and training errors through the regularisation parameter C.


# **Kernel Machines**

## **5. Support Vector Regression**

### **5.1 From Classification to Regression**

**Support Vector Regression (SVR)** extends the SVM framework from classification to regression problems. While classification SVMs aim to find a hyperplane that best separates classes, SVR aims to find a function that best fits the data with a specified tolerance.

The key difference is in the output:
- Classification: discrete outputs $y \in \{-1, +1\}$
- Regression: continuous outputs $y \in \mathbb{R}$

SVR works by learning a function $f(\mathbf{x})$ that approximates the target values $y$ with at most $\epsilon$ deviation while maintaining the maximum margin principle.

As explained in the tutorial answers:
> "Support vector regression combines two ideas, that of the relationship between a linear classifier and linear regression, and the idea of slack variables."

**Duality between classification and regression**:
- A linear regression for n variables involves finding the best n-dimensional hyperplane that fits the data
- This hyperplane divides the space into two regions: points above the hyperplane and points below it
- Where $y = \mathbf{w} \cdot \mathbf{x}$, the two classes are $y > \mathbf{w} \cdot \mathbf{x}$ and $y < \mathbf{w} \cdot \mathbf{x}$
- This duality means that any classifier boundary is the solution to a regression problem, and any regression solution is a classifier boundary

SVR exploits this duality by using SVM methods to create a regression model, focusing on finding a function with good generalization by:
1. Defining a tolerance for errors (the $\epsilon$ tube)
2. Maximizing the margin within that tolerance
3. Using slack variables to allow some points to lie outside the tube

### **5.2 ε-Insensitive Loss Function**

The core of SVR is the **ε-insensitive loss function** developed by Vapnik, which quantifies the error of the prediction:

$$|y - f(\mathbf{x})|_{\epsilon} = \max\{0, |y - f(\mathbf{x})| - \epsilon\}$$

Where:
- $y$ is the actual target value
- $f(\mathbf{x})$ is the predicted value
- $\epsilon$ is the insensitivity parameter

![image](https://github.com/user-attachments/assets/64957160-35ed-4b5f-bf36-489dbc7e0e3b)

> Diagram showing the ε-insensitive tube fitted to data

This loss function has a crucial property: it ignores errors smaller than $\epsilon$. This means:
- Points within $\epsilon$ distance of the regression line contribute no loss
- Only points outside the $\epsilon$-tube contribute to the loss

For a linear regression function $f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$, the SVR optimization problem becomes:

$$\min_{\mathbf{w}} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{m}|y_i - f(\mathbf{x}_i)|_{\epsilon}$$

The first term maximizes the margin (as in classification SVM), while the second term minimizes the error outside the $\epsilon$-tube.

Similar to soft margin classification, we introduce **slack variables** to handle points outside the $\epsilon$-tube:
- $\xi_i$ for points above the upper bound of the tube $(f(\mathbf{x}_i) - y_i > \epsilon)$
- $\xi_i^*$ for points below the lower bound of the tube $(y_i - f(\mathbf{x}_i) > \epsilon)$

This gives us the constrained optimization problem:

$$\min_{\mathbf{w}, \xi^{(*)}} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{m}(\xi_i + \xi_i^*)$$

Subject to:
- $f(\mathbf{x}_i) - y_i \leq \epsilon + \xi_i$
- $y_i - f(\mathbf{x}_i) \leq \epsilon + \xi_i^*$
- $\xi_i, \xi_i^* \geq 0$ for all $i$

The parameter C controls the trade-off between the flatness of $f$ and the amount of deviations larger than $\epsilon$ that are tolerated.

**Example:**
Consider a simple 1D regression problem with points:
$(1,3)$, $(2,5)$, $(3,7)$, $(4,8)$, $(5,11)$

With $\epsilon = 1$ and using a linear function $f(x) = wx + b$:
- If $f(x) = 2x + 1$, then predicted values are: $3, 5, 7, 9, 11$
- For point $(4,8)$, the prediction is $f(4) = 9$, so the error is $|8-9| = 1 = \epsilon$
- This point is exactly on the boundary of the tube, contributing no loss
- For point $(5,11)$, the prediction is $f(5) = 11$, so the error is $|11-11| = 0 < \epsilon$
- This point is inside the tube, contributing no loss

### **5.3 Dual Formulation for Regression**

Similar to classification SVMs, we can derive a dual formulation for SVR using Lagrange multipliers. The dual problem is:

$$\max_{\alpha, \alpha^*} -\frac{1}{2}\sum_{i,j=1}^{m}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)K(\mathbf{x}_i, \mathbf{x}_j) - \epsilon\sum_{i=1}^{m}(\alpha_i + \alpha_i^*) + \sum_{i=1}^{m}y_i(\alpha_i - \alpha_i^*)$$

Subject to:
- $\sum_{i=1}^{m}(\alpha_i - \alpha_i^*) = 0$
- $0 \leq \alpha_i, \alpha_i^* \leq C$ for all $i$

Where:
- $\alpha_i$ and $\alpha_i^*$ are the Lagrange multipliers
- $K(\mathbf{x}_i, \mathbf{x}_j)$ is the kernel function

The resulting regression function becomes:

$$f(\mathbf{x}) = \sum_{i=1}^{m}(\alpha_i^* - \alpha_i)K(\mathbf{x}_i, \mathbf{x}) + b$$

The Lagrange multipliers $\alpha_i$ and $\alpha_i^*$ have specific interpretations:
- For points inside the $\epsilon$-tube: $\alpha_i = \alpha_i^* = 0$
- For points on the upper boundary of the tube: $\alpha_i > 0$, $\alpha_i^* = 0$
- For points on the lower boundary of the tube: $\alpha_i = 0$, $\alpha_i^* > 0$
- For points outside the tube: either $\alpha_i = C$ or $\alpha_i^* = C$

**Support vectors** in SVR are the points with non-zero Lagrange multipliers, which are precisely those on or outside the boundaries of the $\epsilon$-tube.

The dual formulation has the same advantages as in classification SVMs:
- It allows the use of kernels for non-linear regression
- Only the support vectors contribute to the final model
- The complexity of the model depends on the number of support vectors, not the dimensionality of the input space

This makes SVR particularly effective for high-dimensional regression problems and situations where outliers and noise are present in the data.


# **Kernel Machines**

## **6. Additional Kernel Methods**

### **6.1 Gaussian Process Models**

**Gaussian Process (GP) models** represent an elegant fusion of kernel methods and probabilistic approaches, offering a Bayesian perspective on regression and classification problems.

A **Gaussian Process** is a collection of random variables, any finite number of which have a joint Gaussian distribution. GPs are fully specified by a mean function and a covariance function (kernel).

The key relationship between GPs and kernel methods:
- The kernel in SVMs determines the similarity between data points
- In GPs, the kernel defines the covariance between function values at different points
- This allows GPs to encode prior beliefs about the functions being modeled

Starting from the basic Bayesian likelihood calculation:

$$p(h|D) = \frac{p(D|h)p(h)}{p(D)}$$

Where:
- $h$ is the hypothesis (model)
- $D$ is the observed data

In GP models:
- We model $p(D|h)$ as a Gaussian distribution
- This creates a probabilistic version of the linear model
- We then apply kernelization to work in higher-dimensional spaces

Advantages of Gaussian Process models:
- They provide full probabilistic predictions with uncertainty estimates
- They automatically balance model complexity against data fit
- They can learn hyperparameters directly from the data
- They handle noise in a principled way

Unlike SVMs, which provide point predictions, GPs give a distribution over possible functions, allowing for confidence intervals on predictions.

### **6.2 Non-parametric Methods**

**Non-parametric methods** are techniques that cannot be characterized by a bounded set of parameters. As more data is collected, the complexity of the model can grow indefinitely.

Key characteristics of non-parametric methods:
- They make minimal assumptions about the underlying data distribution
- The model complexity increases with the amount of training data
- They often "remember" all training examples
- They typically make predictions based on local information

The most common non-parametric methods include:

1. **k-Nearest Neighbors (k-NN)** for classification:
   - Classify a new point by finding the k nearest training examples
   - Assign the majority class among these neighbors
   - Requires a notion of distance/similarity between points

![image](https://github.com/user-attachments/assets/59dda827-20cd-4485-a56b-6d7a74f9ce72)

> Diagram showing k-NN classification

2. **Distance metrics** commonly used in non-parametric methods:
   - **Minkowski distance**:
     $$L_p(\mathbf{x}_j, \mathbf{x}_q) = \left(\sum_i |x_{ji} - x_{qi}|^p\right)^{1/p}$$
     Special cases include:
     - $p=1$: Manhattan distance
     - $p=2$: Euclidean distance
   
   - **Mahalanobis distance**: Takes into account the covariance between dimensions
   
   - **Kernel-based similarity**: Using the dot product:
     $$\mathbf{x}_j \cdot \mathbf{x}_q$$
     as a measure of similarity between vectors

3. **Non-parametric regression**:
   - **Piecewise linear regression**: Uses adjacent points to fit local linear models
   - **k-NN regression**: Averages the values of the k nearest neighbors
   - **k-NN linear regression**: Fits a linear model to the k nearest neighbors

![image](https://github.com/user-attachments/assets/fd469543-c02c-4276-b68e-151e03859e8b)

![image](https://github.com/user-attachments/assets/86e75a6e-2bc8-4e40-b88d-97beac504a8f)

> Diagrams showing piecewise linear and k-NN regression

Each of these methods has strengths and weaknesses:
- Piecewise linear regression works well for non-noisy data but poorly for noisy data
- k-NN average is simple but can be discontinuous and performs poorly at boundaries
- k-NN linear regression is more flexible but still has discontinuity issues

The choice of k is typically determined through cross-validation, balancing the bias-variance tradeoff:
- Small k: Low bias, high variance (potentially overfitting)
- Large k: Higher bias, lower variance (potentially underfitting)

### **6.3 Locally Weighted Regression**

**Locally Weighted Regression (LWR)** addresses the discontinuity issues of k-NN methods by applying weights to nearby points based on their distance, with closer points having more influence.

![image](https://github.com/user-attachments/assets/e4d1c59e-a9bc-461e-b2f4-661712292131)

> Diagram showing locally weighted regression

The key idea is to give a continuous weight to each training example based on its distance from the query point, rather than using a fixed cutoff as in k-NN methods.

The general form of locally weighted regression is:

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_j K(\text{distance}(\mathbf{x}_q, \mathbf{x}_j))(y_j - \mathbf{w} \cdot \mathbf{x}_j)^2$$

Where:
- $\mathbf{x}_q$ is the query point
- $\mathbf{x}_j$ are the training examples
- $K(\cdot)$ is a kernel function that assigns weights based on distance
- $\text{distance}(\cdot,\cdot)$ is any suitable distance metric

Once the optimal weights $\mathbf{w}^*$ are found (usually by gradient descent), the prediction is:

$$h(\mathbf{x}_q) = \mathbf{w}^* \cdot \mathbf{x}_q$$

A common kernel used in LWR is the **quadratic kernel**:

$$K(d) = \max\left(0, 1 - \left(\frac{2|d|}{k}\right)^2\right)$$

Where:
- $d$ is the distance from the test point
- $k$ is the kernel width (selected via cross-validation)

![image](https://github.com/user-attachments/assets/50b1a061-987b-494e-ab1a-4a46a2502845)

> Diagram showing different kernel widths

Key advantages of LWR:
- It prevents discontinuities by smoothly transitioning between points
- It adapts to the local density of the data
- It can model complex, non-linear relationships
- It only needs to consider points with non-zero weights (those inside the kernel)

Important implementation note:
- A separate optimization problem must be solved for each query point
- This makes prediction computationally expensive compared to parametric methods
- However, it's more flexible and can adapt to local patterns in the data

LWR represents a different type of kernel method compared to SVMs, where the kernel function measures distance rather than similarity, but the fundamental principle of using kernels to transform the problem remains the same.

In summary, these additional kernel methods demonstrate the versatility of the kernel approach beyond SVMs, offering different trade-offs between model complexity, computational efficiency, and the ability to capture complex patterns in data.
