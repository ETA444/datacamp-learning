
## Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning and predictive modeling. It relates to the tradeoff between two sources of error that affect the performance of a predictive model: bias and variance. Let's break down these concepts and their practical implications:

1. **Bias:**

   - **High Bias:** A model with high bias makes strong assumptions about the underlying data distribution. It tends to oversimplify the problem and may not capture the underlying patterns. Such models are often too simple to fit the training data well.

   - **Low Bias:** A model with low bias is flexible and can fit the training data closely. It makes fewer assumptions about the data distribution and is better at capturing complex patterns.

   - **Practical Implications:** If your model has high bias, it is likely to underfit the data. It performs poorly on both the training data and unseen data because it cannot capture the underlying patterns. Increasing model complexity (e.g., using a more sophisticated algorithm or adding more features) can reduce bias.

2. **Variance:**

   - **High Variance:** A model with high variance is overly sensitive to the noise in the training data. It captures random fluctuations in the data, which can lead to poor generalization to unseen data. These models are often complex and can fit the training data perfectly.

   - **Low Variance:** A model with low variance is stable and generalizes well to unseen data. It does not capture noise or random fluctuations in the training data and is less sensitive to small changes in the data.

   - **Practical Implications:** If your model has high variance, it is likely to overfit the training data. It performs very well on the training data but poorly on unseen data because it has learned noise rather than true patterns. Reducing model complexity (e.g., by using regularization) can reduce variance.

The Bias-Variance Tradeoff:

- The goal in machine learning is to find a balance between bias and variance that leads to a model that generalizes well to unseen data.

- Finding the right balance depends on the specific problem and dataset. It often involves adjusting hyperparameters, choosing an appropriate model complexity, and using techniques like cross-validation to evaluate performance.

- Practical strategies for managing the bias-variance tradeoff include:
  - Regularization techniques to reduce variance (e.g., Lasso, Ridge for linear models).
  - Feature engineering to select relevant features and reduce dimensionality.
  - Ensemble methods (e.g., Random Forests, Gradient Boosting) that combine multiple models to reduce variance while controlling bias.

- Visualizations, learning curves, and cross-validation are valuable tools for understanding and managing bias and variance in your models.

In summary, the bias-variance tradeoff is a central concept in machine learning. It involves balancing model complexity to achieve the best possible tradeoff between underfitting and overfitting. Understanding this tradeoff is essential for building effective and robust predictive models.

---
## `sklearn.tree.DecisionTreeClassifier()`

The `DecisionTreeClassifier` class in scikit-learn (sklearn) is used for building and training decision tree models for classification tasks. Decision trees are a type of supervised machine learning algorithm that can be used for both classification and regression tasks. In the case of `DecisionTreeClassifier`, the model is specifically designed for classification tasks.

**Class Constructor:**
```python
sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0)
```

**Parameters:**
- `criterion` (optional): The function used to measure the quality of a split. Options are 'gini' (default) for the Gini impurity or 'entropy' for information gain.
- `splitter` (optional): The strategy used to choose the split at each node. Options are 'best' (default) to choose the best split or 'random' to choose the best random split.
- `max_depth` (optional): The maximum depth of the decision tree. Default is `None`, which means nodes are expanded until all leaves are pure or contain less than `min_samples_split` samples.
- `min_samples_split` (optional): The minimum number of samples required to split an internal node. Default is 2.
- `min_samples_leaf` (optional): The minimum number of samples required to be at a leaf node. Default is 1.
- `min_weight_fraction_leaf` (optional): The minimum weighted fraction of the sum total of weights required to be at a leaf node. Default is 0.0.
- `max_features` (optional): The number of features to consider when looking for the best split. Default is `None`, which means all features are considered.
- `random_state` (optional): Seed for the random number generator used for splitting data. Default is `None`.
- `max_leaf_nodes` (optional): Grow a tree with a maximum number of leaf nodes. Default is `None`.
- `min_impurity_decrease` (optional): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
- `min_impurity_split` (optional): Deprecated and will be removed in future versions.
- `class_weight` (optional): Weights associated with classes. Default is `None`, which means all classes have equal weight.
- `ccp_alpha` (optional): Complexity parameter used for Minimal Cost-Complexity Pruning. Default is 0.0.

**Attributes:**
- `classes_`: The classes known to the classifier.
- `feature_importances_`: The importance of each feature when making predictions.
- `n_features_`: The number of features in the input data.
- `n_classes_`: The number of classes.

**Methods:**
- `fit(X, y)`: Fit the decision tree model to the training data.
- `predict(X)`: Predict the class labels for samples in `X`.
- `predict_proba(X)`: Predict class probabilities for samples in `X`.
- `score(X, y)`: Return the mean accuracy of the model on the given test data and labels.

**Example:**
```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Create a sample dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 1, 0, 1])

# Create a DecisionTreeClassifier with default parameters
clf = DecisionTreeClassifier()

# Fit the classifier to the data
clf.fit(X, y)

# Predict the class labels for new data
new_data = np.array([[2, 3], [4, 5]])
predicted_labels = clf.predict(new_data)

# Print the predicted labels
print(predicted_labels)
```

In this example, a `DecisionTreeClassifier` is created, trained on a sample dataset `X` and `y`, and used to predict the class labels for new data points. Decision trees are useful for classification tasks where the goal is to assign data points to one of several predefined classes based on their features.

---
## `sklearn.model_selection.train_test_split()`

The `train_test_split()` function is a part of scikit-learn (sklearn) and is used for splitting a dataset into two separate sets: a training set and a testing set. It is a crucial step in machine learning and model evaluation to ensure that the model is trained on one subset of the data and tested on another independent subset.

**Function Syntax:**
```python
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
```

**Parameters:**
- `*arrays`: The input data to be split. This can include one or more arrays or matrices, and they will be split into subsets of the same length along their first dimension.
- `test_size` (optional): The proportion of the dataset to include in the testing split. It can be a float (0.0 to 1.0) or an integer (the absolute number of test samples). If `None`, it defaults to 0.25.
- `train_size` (optional): The proportion of the dataset to include in the training split. If `None`, it is computed as `1 - test_size`.
- `random_state` (optional): Seed for the random number generator to ensure reproducibility. If `None`, a different random split is generated every time.
- `shuffle` (optional): Whether to shuffle the data before splitting. Default is `True`.
- `stratify` (optional): An array-like object (e.g., labels) that can be used to perform stratified sampling. It ensures that the proportion of classes in the split is similar to the proportion in the entire dataset.

**Return Value:**
- A tuple containing the split data: `(X_train, X_test, y_train, y_test)`, where `X_train` and `X_test` are the training and testing input data, and `y_train` and `y_test` are the corresponding target labels (if provided).

**Example:**
```python
from sklearn.model_selection import train_test_split
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 1, 0, 1])

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```

In this example, the `train_test_split()` function is used to split the sample data `X` and `y` into training and testing sets, with 20% of the data allocated for testing. The resulting subsets, `X_train`, `X_test`, `y_train`, and `y_test`, are ready for use in training and evaluating machine learning models.

---
## `sklearn.metrics.accuracy_score()`

The `accuracy_score()` function is a part of scikit-learn (sklearn) and is used for evaluating the accuracy of a classification model's predictions. It is commonly used to measure the proportion of correctly predicted labels in a classification problem. High accuracy indicates that the model's predictions are in agreement with the true labels.

**Function Syntax:**
```python
sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
```

**Parameters:**
- `y_true`: The true target labels or ground truth.
- `y_pred`: The predicted labels produced by a classification model.
- `normalize` (optional): A boolean flag indicating whether to return the fraction of correctly classified samples (default) or the total number of correctly classified samples (if set to `False`).
- `sample_weight` (optional): An array-like object that assigns weights to individual samples. It is used to compute a weighted accuracy score.

**Return Value:**
- The accuracy score, which is the ratio of correctly classified samples to the total number of samples if `normalize` is `True`, or the total number of correctly classified samples if `normalize` is `False`.

**Example:**
```python
from sklearn.metrics import accuracy_score

# True labels
y_true = [0, 1, 0, 1, 1, 0, 0, 1]

# Predicted labels from a classifier
y_pred = [0, 1, 0, 1, 1, 0, 1, 0]

# Compute the accuracy score
accuracy = accuracy_score(y_true, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)
```

In this example, the `accuracy_score()` function is used to compute the accuracy of predicted labels (`y_pred`) compared to true labels (`y_true`). The resulting accuracy score is a measure of how well the classifier's predictions match the actual labels. The score is a value between 0.0 and 1.0, where higher values indicate higher accuracy.

---
## `sklearn.tree.DecisionTreeRegressor()`

The `DecisionTreeRegressor` class in scikit-learn (sklearn) is used for building and training decision tree models for regression tasks. Decision trees are a type of supervised machine learning algorithm that can be used for both classification and regression tasks. In the case of `DecisionTreeRegressor`, the model is specifically designed for regression tasks, where the goal is to predict continuous numerical values.

**Class Constructor:**
```python
sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, ccp_alpha=0.0)
```

**Parameters:**
- `criterion` (optional): The function used to measure the quality of a split. Options are 'mse' (mean squared error, default) and 'mae' (mean absolute error).
- `splitter` (optional): The strategy used to choose the split at each node. Options are 'best' (default) to choose the best split or 'random' to choose the best random split.
- `max_depth` (optional): The maximum depth of the decision tree. Default is `None`, which means nodes are expanded until all leaves are pure or contain less than `min_samples_split` samples.
- `min_samples_split` (optional): The minimum number of samples required to split an internal node. Default is 2.
- `min_samples_leaf` (optional): The minimum number of samples required to be at a leaf node. Default is 1.
- `min_weight_fraction_leaf` (optional): The minimum weighted fraction of the sum total of weights required to be at a leaf node. Default is 0.0.
- `max_features` (optional): The number of features to consider when looking for the best split. Default is `None`, which means all features are considered.
- `random_state` (optional): Seed for the random number generator to ensure reproducibility. Default is `None`.
- `max_leaf_nodes` (optional): Grow a tree with a maximum number of leaf nodes. Default is `None`.
- `min_impurity_decrease` (optional): A node will be split if this split induces a decrease of the impurity greater than or equal to this value. Default is 0.0.
- `min_impurity_split` (optional): Deprecated and will be removed in future versions.
- `ccp_alpha` (optional): Complexity parameter used for Minimal Cost-Complexity Pruning. Default is 0.0.

**Attributes:**
- `n_features_`: The number of features in the input data.

**Methods:**
- `fit(X, y, sample_weight=None)`: Fit the decision tree model to the training data.
- `predict(X)`: Predict target values for samples in `X`.
- `score(X, y, sample_weight=None)`: Return the coefficient of determination R^2 of the prediction.
- `apply(X)`: Return the index of the leaf that each sample is predicted as.
- `decision_path(X)`: Return the decision path in the tree.

**Example:**
```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Create a sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Create a DecisionTreeRegressor with default parameters
regressor = DecisionTreeRegressor()

# Fit the regressor to the data
regressor.fit(X, y)

# Predict target values for new data
new_data = np.array([[3.5], [6]])
predicted_values = regressor.predict(new_data)

# Print the predicted values
print(predicted_values)
```

In this example, a `DecisionTreeRegressor` is created, trained on a sample dataset `X` and `y`, and used to predict target values for new data points. Decision trees are useful for regression tasks where the goal is to predict continuous numerical values based on input features.

---
## `sklearn.metrics.mean_squared_error()`

The `mean_squared_error()` function is a part of scikit-learn (sklearn) and is used for calculating the mean squared error (MSE) between the true target values and predicted values in a regression problem. MSE is a common metric used to evaluate the performance of regression models by measuring the average squared difference between the actual and predicted values. Lower MSE values indicate better model performance.

**Function Syntax:**
```python
sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)
```

**Parameters:**
- `y_true`: The true target values or ground truth.
- `y_pred`: The predicted values produced by a regression model.
- `sample_weight` (optional): An array-like object that assigns weights to individual samples. It is used to compute a weighted mean squared error.
- `multioutput` (optional): Defines the aggregation strategy for multioutput data. Options are 'raw_values' (returns an array of MSE values for each output), 'uniform_average' (averages the MSE values), or 'variance_weighted' (weights the MSE values by the variance of each output).
- `squared` (optional): A boolean flag indicating whether to return the squared error values (default) or the root mean squared error (RMSE) values.

**Return Value:**
- The mean squared error (MSE) or root mean squared error (RMSE) value, depending on the `squared` parameter.

**Example:**
```python
from sklearn.metrics import mean_squared_error

# True target values
y_true = [1, 2, 3, 4, 5]

# Predicted values from a regression model
y_pred = [1.1, 2.2, 2.9, 4.2, 5.1]

# Compute the mean squared error
mse = mean_squared_error(y_true, y_pred)

# Print the mean squared error
print("Mean Squared Error:", mse)
```

In this example, the `mean_squared_error()` function is used to calculate the mean squared error between the true target values (`y_true`) and predicted values (`y_pred`) from a regression model. The resulting MSE score quantifies the average squared difference between the true and predicted values, providing a measure of the model's accuracy in predicting continuous numerical outcomes.

---
## Complexity, bias and variance

The relationship between model complexity, bias, and variance is a fundamental concept in machine learning, often referred to as the bias-variance tradeoff.

- As the model's complexity increases (for example, by adding more features or using a more complex algorithm), its capacity to fit the training data also increases. This means that the model can capture more complex patterns in the data.

- When the model's complexity is high, it has lower bias. In other words, it can closely approximate the true underlying function that generated the data. Low bias means that the model is capable of fitting both the training data and the noise in the data.

- However, as the model's complexity increases, it becomes more sensitive to variations and noise in the training data. This leads to higher variance. High variance means that the model is likely to fit the training data very closely, even to the extent of capturing noise and random fluctuations.

- The tradeoff is that, as complexity increases, the model's generalization performance may suffer. It may overfit the training data, meaning it performs well on the training data but poorly on unseen data because it has learned noise instead of true patterns. This is often referred to as overfitting.

To summarize:
- High complexity → Low bias, High variance
- Low complexity → High bias, Low variance

The goal in machine learning is to strike a balance between bias and variance to build models that generalize well to unseen data. This often involves techniques like regularization, cross-validation, and choosing an appropriate level of model complexity. The art of machine learning lies in finding the right balance for a specific problem.

---
## Cross-Validation Error
- If f-hat suffers from **high variance**/**overfitting**: 
	- CV error > training error
		- Remedies:
			- decrease model complexity
				- e.g.: decrease max depth, increase min samples per leaf
			- get more data/rows
- If f-hat suffers from **high bias**/**underfitting**:
	- CV error ~= training error >> desired error
		- Remedies:
			- increase model complexity
				- e.g.: increase max depth, decrease min samples per leaf
			- gather more features/cols

---
## `sklearn.model_selection.cross_val_score()`

The `cross_val_score()` function is part of scikit-learn (sklearn) and is used for performing k-fold cross-validation on a machine learning model. Cross-validation is a widely used technique for assessing a model's performance by splitting the data into multiple subsets (folds) and iteratively training and evaluating the model on different subsets of the data. It helps estimate the model's generalization performance and reduces the risk of overfitting or underfitting.

**Function Syntax:**
```python
sklearn.model_selection.cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=nan)
```

**Parameters:**
- `estimator`: The machine learning model or estimator to be evaluated.
- `X`: The input features or data.
- `y` (optional): The target labels or values (for supervised learning tasks). If omitted, it's used for unsupervised learning or simply ignored.
- `groups` (optional): Group labels for the samples. Used for grouping data points when performing group-based cross-validation.
- `scoring` (optional): The scoring metric used to evaluate the model's performance (e.g., 'accuracy', 'mean_squared_error', 'f1', etc.). If `None`, it uses the default scorer for the estimator.
- `cv` (optional): The cross-validation strategy. You can specify the number of folds or a cross-validation object (e.g., KFold, StratifiedKFold). If `None`, it uses 5-fold cross-validation by default.
- `n_jobs` (optional): The number of CPU cores to use for parallelization. Set to `-1` to use all available cores.
- `verbose` (optional): An integer controlling the verbosity of the output during cross-validation.
- `fit_params` (optional): Additional parameters to pass to the `fit()` method of the estimator.
- `pre_dispatch` (optional): Controls the number of batches to dispatch during parallel execution.
- `error_score` (optional): The value to assign to the score if an error occurs during cross-validation.

**Return Value:**
- An array of scores obtained from cross-validation for each fold. The length of the array is equal to the number of folds.

**Example:**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Perform 5-fold cross-validation with accuracy scoring
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)
```

In this example, the `cross_val_score()` function is used to perform 5-fold cross-validation on a `DecisionTreeClassifier` model using the Iris dataset. The function returns an array of accuracy scores for each fold of cross-validation. Cross-validation is useful for estimating the model's performance on unseen data and can help identify potential issues such as overfitting or underfitting.

---
## `sklearn.ensemble.VotingClassifier()`

The `VotingClassifier` class in scikit-learn (sklearn) is an ensemble learning method that combines the predictions of multiple individual classifiers (estimators) to make a final prediction. It allows you to apply different machine learning algorithms and combine their predictions by majority voting (hard voting) or weighted voting to improve overall prediction accuracy.

**Class Constructor:**
```python
sklearn.ensemble.VotingClassifier(estimators, voting='hard', weights=None, n_jobs=None, flatten_transform=True)
```

**Parameters:**
- `estimators`: A list of tuples, where each tuple consists of a string (a name for the estimator) and an estimator object. Estimators can be classifiers or regressors.
- `voting` (optional): The type of voting to use. Options are 'hard' (default) for majority voting or 'soft' for weighted voting based on class probabilities.
- `weights` (optional): A list of float values specifying the weight assigned to each estimator when using 'soft' voting. If `None`, equal weights are assigned to all estimators.
- `n_jobs` (optional): The number of CPU cores to use for parallel execution during the `fit` and `predict` operations. Set to `-1` to use all available cores.
- `flatten_transform` (optional): A boolean flag indicating whether to flatten the results of the `transform` method for each estimator when using 'soft' voting.

**Attributes:**
- `estimators_`: The list of fitted estimator objects.
- `named_estimators_`: A dictionary containing the names and fitted estimator objects.
- `classes_`: The class labels of the target variable.

**Methods:**
- `fit(X, y, sample_weight=None)`: Fit the ensemble classifier to the training data.
- `predict(X)`: Predict the class labels using the ensemble.
- `predict_proba(X)`: Predict class probabilities using the ensemble.
- `transform(X)`: Apply transformers to the data.

**Example:**
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create individual classifiers
clf1 = DecisionTreeClassifier(max_depth=3)
clf2 = SVC(kernel='linear', probability=True)
clf3 = LogisticRegression(max_iter=1000)

# Create a VotingClassifier with 'soft' voting
ensemble_clf = VotingClassifier(estimators=[('dt', clf1), ('svc', clf2), ('lr', clf3)], voting='soft')

# Fit the ensemble classifier to the data
ensemble_clf.fit(X, y)

# Predict class probabilities for new data
new_data = [[5.1, 3.5, 1.4, 0.2]]
class_probabilities = ensemble_clf.predict_proba(new_data)

# Print the class probabilities
print("Class Probabilities:", class_probabilities)
```

In this example, a `VotingClassifier` is created with three different classifiers: a decision tree classifier (`clf1`), a support vector machine classifier (`clf2`), and a logistic regression classifier (`clf3`). The ensemble classifier combines their predictions using 'soft' voting, which takes into account class probabilities. The `predict_proba()` method is then used to obtain class probabilities for new data points. This ensemble technique is useful for improving the overall predictive performance by leveraging the strengths of multiple individual classifiers.

---