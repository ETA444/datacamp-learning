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

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible_enc', 'Accessible']].head())


### --- Exercise 2 --- ###

# Transform the category_desc column
category_enc = pd.get_dummies(volunteer['category_desc'])

# Take a look at the encoded columns
print(category_enc.head())


### --- Exercise 3 --- ###

# Use .loc to create a mean column
running_times_5k["mean"] = running_times_5k.loc[:, 'run1':'run5'].mean(axis=1)

# Take a look at the results
print(running_times_5k.head())


### --- Exercise 4 --- ###

# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(volunteer['start_date_date'])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer["start_date_converted"].dt.month

# Take a look at the converted and new month columns
print(volunteer[["start_date_month", "start_date_converted"]].head())


### --- Exercise 5 --- ###

# Write a pattern to extract numbers and decimals
def return_mileage(length):
    
    # Search the text for matches
    mile = re.search("\d+.\d+", length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking["Length"].apply(return_mileage)
print(hiking[["Length", "Length_num"]].head())


### --- Exercise 6 --- ###

# Take the title text
title_text = volunteer["title"]

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)


### --- Exercise 7 --- ###

# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify=y, random_state=42)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))



## Chapter 4

### --- Exercise 1 --- ###

# Create a list of redundant column names to drop
to_drop = ["vol_requests", "category_desc", "locality", "region", "created_date"]

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of volunteer_subset
print(volunteer_subset.head())


### --- Exercise 2 --- ###

# Print out the column correlations of the wine dataset
print(wine.corr())
print(wine.corr() > 0.75)

# Drop that column from the DataFrame
wine = wine.drop('Flavanoids', axis=1)

print(wine.head())


### --- Exercise 3 --- ###

# Add in the rest of the arguments
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    
    # Transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    
    # Sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_, text_tfidf, 8, 3))


### --- Exercise 4 --- ###

def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):
    
        # Call the return_weights function and extend filter_list
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
        
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)

# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# Filter the columns in text_tfidf to only those in filtered_words
filtered_text = text_tfidf[:, list(filtered_words)]


### --- Exercise 5 --- ###

# Split the dataset according to the class distribution of category_desc
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))


### --- Exercise 6 --- ###

# Instantiate a PCA object
pca = PCA()

# Define the features and labels from the wine dataset
X = wine.drop('Type', axis=1)
y = wine["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Apply PCA to the wine dataset X vector
pca_X_train = pca.fit_transform(X_train, y_train)
pca_X_test = pca.transform(X_test)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)


### --- Exercise 7 --- ###

# Fit knn to the training data
knn.fit(pca_X_train, y_train)

# Score knn on the test data and print it out
print(knn.score(pca_X_test, y_test))




## Chapter 5

### --- Exercise 1 --- ###

# Print the DataFrame info
print(ufo.info())

# Change the type of seconds to float
ufo["seconds"] = ufo['seconds'].astype(float)

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo["date"])

# Check the column types
print(ufo.info())


### --- Exercise 2 --- ###

# Count the missing values in the length_of_time, state, and type columns, in that order
print(ufo[["length_of_time", "state", "type"]].isna().sum())

# Drop rows where length_of_time, state, or type are missing
ufo_no_missing = ufo.dropna(subset=["length_of_time", "state", "type"])

# Print out the shape of the new dataset
print(ufo_no_missing.shape)


### --- Exercise 3 --- ###

def return_minutes(time_string):

    # Search for numbers in time_string
    num = re.search("\d+", time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(return_minutes)

# Take a look at the head of both of the columns
print(ufo[['minutes', 'length_of_time']].head())


### --- Exercise 4 --- ###

# Check the variance of the seconds and minutes columns
print(ufo[['seconds', 'minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo['seconds'])

# Print out the variance of just the seconds_log column
print(ufo["seconds_log"].var())


### --- Exercise 5 --- ###

# Use pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(
    lambda country: 1 if country == 'us' else 0
)

# Print the number of unique type values
print(len(ufo["type"].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo["type"])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)


### --- Exercise 6 --- ###

# Look at the first 5 rows of the date column
print(ufo['date'].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].dt.month

# Extract the year from the date column
ufo["year"] = ufo["date"].dt.year

# Take a look at the head of all three columns
print(ufo[['date','month','year']].head())


### --- Exercise 7 --- ###

# Take a look at the head of the desc field
print(ufo['desc'].head())

# Instantiate the tfidf vectorizer object
vec = TfidfVectorizer()

# Fit and transform desc using vec
desc_tfidf = vec.fit_transform(ufo.desc)

# Look at the number of columns and rows
print(desc_tfidf.shape)


### --- Exercise 8 --- ###

# Make a list of features to drop
to_drop = [
    'city', 'country', 'lat', 'long', 'state',
    'date', 'recorded', 'seconds', 'minutes', 
    'desc', 'length_of_time'
    ]

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)


### --- Exercise 9 --- ###

# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    random_state=42
)

# Fit knn to the training sets
knn.fit(X_train, y_train)

# Print the score of knn on the test sets
print(knn.score(X_test, y_test))


### --- Exercise 10 --- ###

# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y 
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit nb to the training sets
nb.fit(X_train, y_train)

# Print the score of nb on the test sets
print(nb.score(X_test, y_test))