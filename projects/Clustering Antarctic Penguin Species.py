# Code for Datacamp Project: "Clustering Antarctic Penguin Species"

# Import Required Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder
)

# Import explore_df
def explore_df(df, method):
    """
    Function to run describe, head, or info on df.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to explore.
    method : {'desc', 'head', 'info', 'all'}
        Specify the method to use.
        - 'desc': Display summary statistics using describe().
        - 'head': Display the first few rows using head().
        - 'info': Display concise information about the DataFrame using info().
        - 'na': Display counts of NAs per column and percentage of NAs per column.
        - 'all': Display all information from above options.

    Returns
    -------
    None
    """
    if method.lower() == "desc":
        print(df.describe())
    elif method.lower() == "head":
        pd.set_option('display.max_columns', None)
        print(df.head())
        pd.reset_option('display.max_columns')
    elif method.lower() == "info":
        print(df.info())
    elif method.lower() == "na":
        print(f"\n\n<<______NA_COUNT______>>")
        print(df.isna().sum())
        print(f"\n\n<<______NA_PERCENT______>>")
        print((df.isna().sum() / df.shape[0])*100)
    elif method.lower() == "all":
        print("<<______HEAD______>>")
        pd.set_option('display.max_columns', None)
        print(df.head())
        pd.reset_option('display.max_columns')
        print(f"\n\n<<______DESCRIBE______>>")
        print(df.describe())
        print(f"\n\n<<______INFO______>>")
        print(df.info())
        print(f"\n\n<<______NA_COUNT______>>")
        print(df.isna().sum())
        print(f"\n\n<<______NA_PERCENT______>>")
        print((df.isna().sum() / df.shape[0])*100)
    else:
        print("Methods: 'desc', 'head', 'info' or 'all'")


# lad the dataset
penguins_df = pd.read_csv("data/penguins.csv")

# explore the dataset
explore_df(penguins_df, 'all')

# inspect unique values of categorical vars
print(f"\n\n['sex']\nUnique values:\n{penguins_df['sex'].unique()}\n\nCounts of unique values:\n{penguins_df['sex'].value_counts()}")

# Findings:
# (1) There is a small number of NAs (%NA/col < 5% => drop)
# (2) There is a negative value in flipper length (not logical)
# (3) There seems to be at least 1 NA in flipper length
# (4) sex has 1 value with category . (remove it)


"""Task 1. Begin by reading in "data/penguins.csv" as a pandas
DataFrame called penguins_df, then investigate and clean the
dataset by removing the null values and outliers. Save as a
cleaned DataFrame called penguins_clean."""

# Task 1.1: Treat NAs
# %NAs:
"""<<______NA_PERCENT______>>
culmen_length_mm     0.581395
culmen_depth_mm      0.581395
flipper_length_mm    0.581395
body_mass_g          0.581395
sex                  2.616279
dtype: float64"""
# Conclusion: remove rows with NAs, since %NAs < 5%
penguins_clean = (
    penguins_df
    .dropna()
    .reset_index(drop=True)
)

# sanity check
explore_df(penguins_clean, 'na')


"""# Task 1.2: Remove negative values
# Remove negative values from 'flipper_length_mm' column
penguins_clean = penguins_clean[penguins_clean['flipper_length_mm'] > 0]

# sanity check
explore_df(penguins_clean['flipper_length_mm'], 'desc') """


# Task 1.3: Treat Outliers (remove per DataCamp instruction)
# Note: this will be used in DataSafari
# (datasafari.explore_num())

penguins_clean.boxplot()

# Calculate the interquartile range (IQR)
Q1 = penguins_clean.quantile(0.25)
Q3 = penguins_clean.quantile(0.75)
IQR = Q3 - Q1

# Determine lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify rows with outliers
outliers = penguins_clean[((penguins_clean < lower_bound) | (penguins_clean > upper_bound)).any(axis=1)]

# Display rows with outliers
print(outliers)

penguins_clean = penguins_clean[~((penguins_clean < lower_bound) | (penguins_clean > upper_bound)).any(axis=1)]

""" They don't want this method
# outlier identification (oid): z-score method

# dictionary containing values of a column that
# correspond to its outliers based on z-score > 3
oid = {}

# cols to check
cols = [
    'culmen_length_mm', 'culmen_depth_mm',
    'flipper_length_mm', 'body_mass_g'
]

treatment = 'remove'

for col in cols:
    
    # zscore column name
    z_col = col+'_zscore'
    
    # calculate z-score for col
    penguins_clean[z_col] = (
        (penguins_clean[col] - penguins_clean[col].mean())
        / penguins_clean[col].std()
    )
    
    # threshold for outlier (z-score > 3 or < -3)
    threshold = 3
    
    # identify outliers
    oid_values = (
        penguins_clean[
            (penguins_clean[z_col].abs() > threshold)
        ][col].tolist()
    )
    
    # save results to oid dictionary
    oid[col] = oid_values
    
    # Drop z-score column
    penguins_clean = penguins_clean.drop(columns=[z_col])
    
    print(f"Pre-treatment .describe() of ['{col}']:\n{penguins_clean[col].describe()}\n\n")
    
# sanity check - oid should be a dictionary of:
# key: column name in df
# value: list of values that have z-scores higher than 3
# the idea is that we don't burden the df out of the for-loop
# with new z-score columns, but extract which rows are outliers
print(oid.items())

# Findings:
# (1) we can see that flipper length has an outlier of 5000
# (2) also in the same variable there is a negative length
"""


""" They don't want this method
# sanity check pre-treatment
print(f"Pre-treatment .describe() of ['flipper_length_mm']:\n{penguins_clean['flipper_length_mm'].describe()}\n\n")

# Treat outlier
penguins_clean = (
    penguins_clean
    [~penguins_clean['flipper_length_mm'] # note: use of '~' 
     .isin(oid['flipper_length_mm']) # => opposite of is in
    ]
)

# sanity check
print(f"Post-treatment .describe() of ['flipper_length_mm']:\n{penguins_clean['flipper_length_mm'].describe()}") """

explore_df(penguins_clean, 'all')

# Task 1.4: Remove row with '.' value for sex and make into categorical
penguins_clean = penguins_clean[penguins_clean['sex'] != '.']
penguins_clean['sex'] = pd.Categorical(penguins_clean['sex'])

# sanity check
print(f"['sex']\nUnique values:\n{penguins_clean['sex'].unique()}\n\nCounts of unique values:\n{penguins_clean['sex'].value_counts()}")

"""Task 2. Pre-process the cleaned data using standard scaling
and the one-hot encoding to add dummy variables:
- Create the dummy variables and remove the original 
categorical feature from the dataset.
- Scale the data using the standard scaling method.
- Save the updated data as penguins_preprocessed.
"""

"""
# Task 2.1: One-hot encode sex
# ( datasafari code )
# one-hot encode variables that need it

var_hot_encode = ['sex']

# create an instance of OneHotEncoder
onehot_encoder = OneHotEncoder()

# encode the selected variables
for var in var_hot_encode:
    # fit and transform the data
    encoded_values = onehot_encoder.fit_transform(penguins_clean[[var]])
    
    # convert the sparse matrix to a DataFrame and append to ci
    encoded_df = pd.DataFrame(
        encoded_values.toarray(),
        columns=onehot_encoder.get_feature_names_out([var])
    )

    # Reset indices of both DataFrames
    penguins_clean.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    
    # append the new columns to df
    penguins_clean = pd.concat([penguins_clean, encoded_df], axis=1)
    
    # drop the original column
    penguins_clean.drop(columns=[var], inplace=True)

    # inform user
    print(f"One-hot encoded values of ['{var}']:\n{encoded_df.head()}\n\n")

# create penguins_preprocessed
penguins_preprocessed = penguins_clean.copy()

# sanity check
explore_df(penguins_preprocessed, 'all')
"""
# Step 3 - Perform preprocessing steps on the dataset to create dummy variables
penguins_preprocessed = pd.get_dummies(penguins_clean).drop('sex_.',axis=1)

# Task 2.2: Scale numerical features

# cols to be scaled
columns_to_scale = [
    'culmen_length_mm', 'culmen_depth_mm',
    'flipper_length_mm', 'body_mass_g'
]

# init scaler
scaler = StandardScaler()

# fit and transform directly on df
penguins_preprocessed[columns_to_scale] = scaler.fit_transform(penguins_preprocessed[columns_to_scale])

# sanity check
print(penguins_preprocessed[columns_to_scale].describe().round(3))


"""
Task 3. Perform Principal Component Analysis (PCA) on the 
penguins_preprocessed dataset to determine the desired number
of components, considering any component with an explained
variance ratio above 10% as a suitable component. 
- Save the number of components as a variable called n_components.
- Finally, execute PCA using n_components and store the result as penguins_PCA.
"""

# instantiate and fit pca
pca = PCA()
pca.fit(penguins_preprocessed)


# plot explained variance ratio
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    marker='o'
)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Number of Components')
plt.grid(True)
plt.show()

# We can see the elbow is at 3 components
n_components = 2

# execute PCA with 3 components
pca_3 = PCA(n_components=n_components)
penguins_PCA = pca.fit_transform(penguins_preprocessed)

"""Task 4. Employ k-means clustering on the penguins_PCA
dataset, setting random_state=42, to determine the number of
clusters through elbow analysis. Save the optimal number of
clusters in a variable called n_cluster."""

# Initialize an empty list to store the sum of squared distances
sum_squared_distances = []

# Specify a range of k values to try
k_values = range(1, 11)  # You can adjust this range based on your dataset and needs

# Perform k-means clustering for each value of k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(penguins_PCA)
    sum_squared_distances.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(
    k_values,
    sum_squared_distances,
    marker='o'
)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_values)
plt.show()

# elbow method shows 4
n_clusters = 4

print("Optimal number of clusters:", n_clusters)

"""
Task 5. Create and fit a new k-means cluster model, setting
n_cluster equal to your n_cluster variable, saving the model as
a variable called kmeans.
- Visualize clusters using the first two principle components.
"""
# final k means model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(penguins_PCA)

# Visualize the clusters using the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter( # plot scatter of two first components
    penguins_PCA[:, 0],
    penguins_PCA[:, 1],
    c=kmeans.labels_, # color based on cluster label
    cmap='viridis',
    s=50,
    alpha=0.8
)
plt.scatter( # plot centroids
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='x',
    s=200,
    c='red',
    label='Centroids'
)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()

"""
Task 6. Add the label column extracted from the k-means
clustering to the penguins_clean DataFrame.

Task 7. Create a statistical table by grouping penguins_clean
based on the "label" column and calculating the mean of each
numeric column. Save this table as stat_penguins.
"""
# add label col to clean data
penguins_clean['label'] = kmeans.labels_

# group based on label and calculate mean
stat_penguins = penguins_clean.groupby('label').mean()

# print the statistical table
print(stat_penguins)


# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1 - Loading and examining the dataset
penguins_df = pd.read_csv("data/penguins.csv")
penguins_df.head()
penguins_df.info()

# Step 2 - Dealing with null values and outliers
penguins_df.boxplot()  
plt.show()

penguins_clean = penguins_df.dropna()
penguins_clean[penguins_clean['flipper_length_mm']>4000]
penguins_clean[penguins_clean['flipper_length_mm']<0]
penguins_clean = penguins_clean.drop([9,14])

# Step 3 - Perform preprocessing steps on the dataset to create dummy variables
df = pd.get_dummies(penguins_clean).drop('sex_.',axis=1)

# Step 4 - Perform preprocessing steps on the dataset - scaling
scaler = StandardScaler()
X = scaler.fit_transform(df)
penguins_preprocessed = pd.DataFrame(data=X,columns=df.columns)
penguins_preprocessed.head(10)

# Step 5 - Perform PCA
pca = PCA(n_components=None)
dfx_pca = pca.fit(penguins_preprocessed)
dfx_pca.explained_variance_ratio_
n_components=sum(dfx_pca.explained_variance_ratio_>0.1)
pca = PCA(n_components=n_components)
penguins_PCA = pca.fit_transform(penguins_preprocessed)

# Step 6 - Detect the optimal number of clusters for k-means clustering
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(penguins_PCA)
    inertia.append(kmeans.inertia_)    
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
n_clusters=4

# Step 7 - Run the k-means clustering algorithm
# with the optimal number of clusters 
# and visualize the resulting clusters.
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(penguins_PCA)
plt.scatter(penguins_PCA[:, 0], penguins_PCA[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title(f'K-means Clustering (K={n_clusters})')
plt.legend()
plt.show()

# Step 8 - Create a final statistical DataFrame for each cluster.
penguins_clean['label'] = kmeans.labels_
numeric_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','label']
stat_penguins = penguins_clean[numeric_columns].groupby('label').mean()
stat_penguins
