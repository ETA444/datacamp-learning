# Code Exercises from Supervised Learning with scikit-learn #

## Chapter 1

### --- Exercise 1 --- ###

# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 

y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions
print(f"Predictions: {y_pred}") 



### --- Exercise 2 --- ###

# Import the module
from sklearn.model_selection import train_test_split

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))



### --- Exercise 3 --- ###





### --- Exercise 4 --- ###

# Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
  
	# Set up a KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=neighbor)
  
	# Fit the model
	knn.fit(X_train, y_train)
  
	# Compute accuracy
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)



### --- Exercise 5 --- ###

# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(
    train_accuracies.keys(),
    train_accuracies.values(),
    label="Training Accuracy")

# Plot test accuracies
plt.plot(
    test_accuracies.keys(),
    test_accuracies.values(),
    label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()



## Chapter 2

### --- Exercise 1 --- ###





### --- Exercise 2 --- ###





### --- Exercise 3 --- ###





### --- Exercise 4 --- ###





### --- Exercise 5 --- ###





### --- Exercise 6 --- ###





### --- Exercise 7 --- ###





### --- Exercise 8 --- ###






## Chapter 3

### --- Exercise 1 --- ###





### --- Exercise 2 --- ###





### --- Exercise 3 --- ###





### --- Exercise 4 --- ###





### --- Exercise 5 --- ###





### --- Exercise 6 --- ###





### --- Exercise 7 --- ###





### --- Exercise 8 --- ###







## Chapter 4

### --- Exercise 1 --- ###





### --- Exercise 2 --- ###





### --- Exercise 3 --- ###





### --- Exercise 4 --- ###





### --- Exercise 5 --- ###





### --- Exercise 6 --- ###





### --- Exercise 7 --- ###





### --- Exercise 8 --- ###

