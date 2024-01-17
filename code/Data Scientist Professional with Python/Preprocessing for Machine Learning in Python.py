# Code Exercises from Preprocessing for Machine Learning in Python #

## Chapter 1

### --- Exercise 1 --- ###

# Drop the Latitude and Longitude columns from volunteer
volunteer_cols = volunteer.drop(
                        labels = ['Latitude', 'Longitude'],
                        axis = 1
)

# Drop rows with missing category_desc values from volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# Print out the shape of the subset
print(volunteer_subset.shape)


### --- Exercise 2 --- ###

# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer['hits'].astype(int)

# Look at the dtypes of the dataset
print(volunteer.dtypes)


### --- Exercise 3 --- ###

# Create a DataFrame with all columns except category_desc
X = volunteer.drop('category_desc', axis=1)

# Create a category_desc labels dataset
y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the y dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Print the category_desc counts from y_train
print(y_train['category_desc'].value_counts())



## Chapter 2

### --- Exercise 1 --- ###

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Fit the knn model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))


### --- Exercise 2 --- ###

# Print out the variance of the Proline column
print(wine['Proline'].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the normalized Proline column
print(wine['Proline_log'].var())


### --- Exercise 3 --- ###

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create the scaler
scaler = StandardScaler()

# Subset the DataFrame you want to scale 
wine_subset = wine[['Ash','Alcalinity of ash', 'Magnesium']]

# Apply the scaler to wine_subset
wine_subset_scaled = scaler.fit_transform(wine_subset)


### --- Exercise 4 --- ###

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Fit the k-nearest neighbors model to the training data
knn = knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))


### --- Exercise 5 --- ###

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Instantiate a StandardScaler
scaler = StandardScaler()

# Scale the training and test features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train_scaled, y_train)

# Score the model on the test data
print(knn.score(X_test_scaled, y_test))



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





## Chapter 5

### --- Exercise 1 --- ###




### --- Exercise 2 --- ###




### --- Exercise 3 --- ###




### --- Exercise 4 --- ###




### --- Exercise 5 --- ###




### --- Exercise 6 --- ###




### --- Exercise 7 --- ###




### --- Exercise 8 --- ###




