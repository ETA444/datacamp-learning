# Code for Datacamp Project: "Predictive Modeling for Agriculture"

# All required libraries are imported here for you.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

        
# Load the dataset
crops = pd.read_csv("soil_measures.csv")


"""Task. 1: Read in soil_measures.csv as a pandas DataFrame and perform
some data checks, such as determining the number of crops, checking for 
missing values, and verifying that the data in each potential feature 
column is numeric."""

# Explore dataset
explore_df(crops, 'all')
# findings: 
# (1) no NAs whatsoever
# (2) data types look good (will check crop seperately)

# check if crop is categorical
print(f"['crop'] column before:\n{crops['crop'].unique()}\n\n")

# change crops to unordered categorical
crops['crop'] = pd.Categorical(crops['crop'])

# sanity check
print(f"['crop'] column after:\n{crops['crop'].unique()}\n\n")


"""Task 2. Split the data into training and test sets, setting
test_size equal to 20% and using a random_state of 42."""

X = crops[['N', 'P', 'K', 'ph']]
y = crops['crop']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



"""Task 3. Predict the "crop" type using each feature individually by
looping over all the features, and, for each feature, fit a Logistic 
Regression model and calculate f1_score(). When creating the model,
set max_iter to 2000 so the model can converge, and pass an appropriate
string value to the multi_class keyword argument."""

# Build single feature models
features = X.columns.tolist()

# Save single feature models performance metric
singlefeat_logr_f1scores = {}

for feature in features:
    # instantiate model: LogisticRegression()
    logr = LogisticRegression(
        max_iter=2000,
        multi_class='multinomial' # since crop has many classes
    )
    
    # fit the model on training data
    logr.fit(X_train[[feature]], y_train)
    
    # get predictions
    y_pred = logr.predict(X_test[[feature]])
    
    # calc f1 'macro' as it is multiclass
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # save results to dict
    singlefeat_logr_f1scores[feature] = np.round(f1, 4)

# sanity check dict
print(singlefeat_logr_f1scores)



"""Task 4. In order to avoid selecting two features that are highly
correlated, perform a correlation analysis for each pair of features,
enabling you to build a final model without the presence of
multicollinearity."""

# create correlation matrix
correlation_matrix = crops[features].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5
)
plt.title('Correlation Matrix of Features')
plt.show()

# Findings:
# (1) features P and K are highly correlated (0.74)
# (2) All other features are not correlated

# Option 1 - Remove P: Considering that P has a lower F1 score
# it is reasonable to remove it from the dataset.
# Option 2 - Regularization: We could use regularization 
# techniques which effectively reduce the impact of 
# multicollinearity. We could build models like Ridge, if we
# want to definitely keep all features.
# Other options - there of course are other options
# for example transforming features or PCA.

# Conclusion:
# The tasks imply that we need to build a Logreg, so option 2
# seems unsuitable. Other options seems out of scope for this
# exercise too since the final task requires f1_score derived
# from a logreg model. Therefore, I will remove P based on it
# having a lower F1 in the 1-feature model.


"""Task 5. Once you have your final features, train and test a new
Logistic Regression model called log_reg, then evaluate performance
using f1_score(), saving the metric as a variable called 
model_performance."""

# define final features
final_features = ['N', 'K', 'ph']

# resplit the data
X_train, X_test, y_train, y_test = train_test_split(
    X[final_features], y, test_size=0.2, random_state=42
)

# instantiate model: LogisticRegression()
log_reg = LogisticRegression(
    max_iter=2000,
    multi_class='multinomial'
)

# fit the model on training data
log_reg.fit(X_train[final_features], y_train)

# get predictions
y_pred = log_reg.predict(X_test[final_features])

# calc f1 'macro' as it is multiclass
model_performance = f1_score(y_test, y_pred, average='macro')

print(f"The final model's F1-score is: {model_performance}")