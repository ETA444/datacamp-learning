# Code for Datacamp Project: "Modeling Car Insurance Claim Outcomes"

# Import required modules
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression, RidgeCV
)
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder
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

# Import the data
ci = pd.read_csv("car_insurance.csv")

# Explore the df
explore_df(ci, 'all')


# Datacamp Tasks:

# Task 1. Identify the single feature of the data that is 
# the best predictor of whether a customer will put in a
# claim (the "outcome" column), excluding the "id" column.

# Task 2. Store as a DataFrame called best_feature_df, 
# containing columns named "best_feature" and "best_accuracy" 
# with the name of the feature with the highest accuracy, 
# and the respective accuracy score.

# I will demonstrate 3 approaches:
# (1) Using statsmodels but very particularly to answer task 1
# (2) Using statsmodels (suited to inference, not for task 1)
# (3) Using scikit (suited for ML applications and using model on new data, not for task 1)

# Minimal Data Pre-processing for Short Answer:
# 'credit_score' (%NAs: 9.82) 
# and 'annual_mileage' (%NAs: 9.57)
# since %NA is > 5%, dropping these rows is not ideal
# due to lack of industry knowledge we will just drop it
# so that models can run
ci.dropna(subset=['credit_score', 'annual_mileage'], inplace=True)



# Short Answer
# (1) Using statsmodels but very particularly to answer task 1

def answer(df, x):
    # Fit logistic regression model
    logit_model = logit(formula='outcome ~'+x, data=df).fit()

    # Calculate accuracy
    y_pred = (logit_model.predict(df[x]) > 0.5).astype(int)  # Convert to int
    accuracy = (y_pred == df['outcome']).mean()
    
    # best feature-best accuracy dict append
    bf_ba_dict[x] = accuracy
    
    # notify in console output
    print(f"['{x}'] had an accuracy of: {accuracy}\n\n")
    
# define list of x vars: xs
xs = (
    ci.drop(columns=['outcome', 'id']) # remove y var and id
    .columns.tolist() # make them into a list to use in for-loop
)


# init dict
bf_ba_dict = {}

# run single var model per dv
for x in xs:
    answer(ci, x)

# create answer df
best_feature_df = pd.DataFrame({
    'best_feature':bf_ba_dict.keys(),
    'best_accuracy':bf_ba_dict.values()
}).sort_values('best_accuracy', ascending=False).head(1)
print(best_feature_df)




# Long (Not) Answer (at this point just working on DataSafari pkg)

# Maximum Data Pre-processing and exploration
# pre-task data inspection (0-8)
# inspection (0): many floats in the df don't need to be, convert to int64
var_fl_to_int = [
    'vehicle_ownership', # 0 1
    'married', # 0 1
    'outcome' # 0 1
]

for var in var_fl_to_int:
    ci[var] = ci[var].astype(int)
    print(f"Converted ['{var}'] to {ci[var].dtype}.\n")


# inspection (1): here I am focused on checking out whether categorical-like variables have logical categories
# note: these should be categorical for efficiency
var_inspect1 = [
    'age', 'gender', 'driving_experience', 'education',
    'income', 'vehicle_ownership', 'vehicle_year', 'married',
    'vehicle_type', 'outcome'
]

for var in var_inspect1:
    # print unique categories:
    print(f"[{var}] There are {len(ci[var].unique())} unique categories, namely: {ci[var].unique()}\n\n")

# result: all categories are proper!


# make ordered categorical: var_order
var_order = {
    'age':[0,1,2,3],
    'driving_experience':['0-9y','10-19y','20-29y','30y+'],
    'education':['none','high school','university'],
    'income':['poverty','working class','middle class','upper class'],
    'vehicle_year':['before 2015', 'after 2015']
}
for var_name, nom_order in var_order.items():
    ci[var_name] = pd.Categorical(
        ci[var_name],
        categories = nom_order,
        ordered=True
    )
    print(f"Converted ['{var_name}'] to an ordinal categorical variable, with the following order:\n{ci[var_name].unique()}\n\n")


# ordinal encode variables that need it
# this code is for a package I'm building

# define the desired order for each variable
var_encode_order = {
    'driving_experience': ['0-9y', '10-19y', '20-29y', '30y+'],
    'education': ['none', 'high school', 'university'],
    'income': ['poverty', 'working class', 'middle class', 'upper class'],
    'vehicle_year': ['before 2015', 'after 2015']
}

# Encode the selected variables
for var, order in var_encode_order.items():
    
    # new column name
    new_var = var + '_encoded'
    
    # OrdinalEncoder /w explicit order specification
    ord_encoder = OrdinalEncoder(categories=[order])
    
    ci[new_var] = ord_encoder.fit_transform(ci[[var]])
    original_labels = ord_encoder.categories_[0].tolist()
    print(f"<Encoded values of ['{var}'] stored in ['{new_var}'].>\n\nFollowing two columns now exist:\n{ci[[var,new_var]][0:4]}\n\n")


# make categorical: var_cat
var_cat = [
    'gender', 'vehicle_ownership',
    'married', 'vehicle_type', 'outcome'
]

for var in var_cat:
    ci[var] = pd.Categorical(ci[var])
    print(f"Converted ['{var}'] to a nominal categorical variable:\n{ci[var].unique()}\n\n")



# map 'vehicle_type': 0 sedan, 1 sports car (for modeling)
# Define mapping for vehicle_type
vehicle_type_mapping = {'sedan': 0, 'sports car': 1}

# Apply mapping to create encoded column
ci['vehicle_type'] = ci['vehicle_type'].map(vehicle_type_mapping)

# sanity check
print(ci['vehicle_type'].unique())


# one-hot encode variables that need it
# this code is for a package I'm building
var_hot_encode = [
    # none
]

# create an instance of OneHotEncoder
onehot_encoder = OneHotEncoder()

# encode the selected variables
for var in var_hot_encode:
    # fit and transform the data
    encoded_values = onehot_encoder.fit_transform(ci[[var]])
    
    # convert the sparse matrix to a DataFrame and append to ci
    encoded_df = pd.DataFrame(
        encoded_values.toarray(),
        columns=onehot_encoder.get_feature_names_out([var])
    )
    
    # append the new columns to df
    ci = pd.concat([ci, encoded_df], axis=1)
    
    # drop the original column
    ci.drop(columns=[var], inplace=True)
    
    print(f"One-hot encoded values of ['{var}']:\n{encoded_df.head()}\n\n")


# inspection (2): integrity check 'credit_score' values
# inspection (3): integrity check 'children' values
# inspection (4): integrity check 'postal_code' values
# inspection (5): integrity check 'annual_mileage' values
# inspection (6): integrity check 'speeding_violations' values
# inspection (6): integrity check 'duis' values
# inspection (7): integrity check 'past_accidents' values
ic_vars = [
    'credit_score', 'children', 'postal_code',
    'annual_mileage', 'speeding_violations',
    'duis', 'past_accidents'
]

for var in ic_vars:
    print(f"['{var}']\nMean: {ci[var].mean():.2f}\nMedian: {ci[var].median():.2f}\nMin: {ci[var].min():.2f}\nMax: {ci[var].max():.2f}\n\n")
    
# all variables have values within reason, aside from:
# - there is an observation with 22 speeding violations
# - there is an observation with 6 DUIs



# inspection (8): treatment of NAs in
# 'credit_score' (%NAs: 9.82) 
# and 'annual_mileage' (%NAs: 9.57)
# since %NA is > 5%, dropping these rows is not ideal
# due to lack of industry knowledge we will just drop it
# so that models can run
ci.dropna(subset=['credit_score', 'annual_mileage'], inplace=True)



# (2) Using statsmodels (suited for inference)

# Step 1: Prepare the data
X_a1 = ci.drop(columns=[
    'outcome', 'id', # drop y, id
    'driving_experience', 'education',
    'income', 'vehicle_year' # and non-encoded
])
y_a1 = ci['outcome']  # Target variable

# Add intercept term to X
X_a1 = sm.add_constant(X_a1)

# Step 2.1: Fit a logistic regression model with all features
full_logit_results_a1 = sm.Logit(y_a1, X_a1).fit()

# Step 2.2: Print summary of the model
print(full_logit_results_a1.summary())

# coeff with p < 0.05
# gender                         0.9443
# vehicle_ownership             -1.7807
# married                       -0.3370
# postal_code                  2.15e-05
# annual_mileage              7.533e-05
# speeding_violations            0.0627
# past_accidents                -0.1641
# driving_experience_encoded    -1.7275
# vehicle_year_encoded          -1.6960

# Get the p-values from the model results
p_values = full_logit_results_a1.pvalues

# Get the coefficients from the model results
coefficients = full_logit_results_a1.params

# Filter coefficients based on p-values
significant_coefficients = coefficients[p_values < 0.05]

print(f"\nThese were the significant coefficients:\n{significant_coefficients}\n\n")

# Convert coefficients to log odds
odds_ratios = np.exp(significant_coefficients)

print(f"\nThese are their respective log odds:\n{odds_ratios}\n\n")

# Calculate the percentage change in odds
percentage_changes = (1 - odds_ratios) * 100

# Print the results
print(f"\nPercentage Change in Odds:\n{percentage_changes}\n\n")

# Conclusion:
# gender is most influential, where each unit increase in gender (from 0 to 1, or female to male), 
# the odds of a claim decrease by 157% (percent_change), 
# with a magnitute of 2.57 (log odds)


# (3) Using scikit (suited for ML applications and using model on new data)

# Step 1: Prepare the data
X = ci.drop(columns=[
    'outcome', 'id', # drop y, id
    'driving_experience', 'education',
    'income', 'vehicle_year' # and non-encoded
])
y = ci['outcome']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4242)

# Step 2.1: Fit a logistic regression model with all features
full_logit = LogisticRegression()
full_logit.fit(X_train, y_train)

# Step 2.2: Evaluate full logit model
accuracy_full = full_logit.score(X_test, y_test)
print(f"Accuracy of full model: {accuracy_full}\n\n")

# Step 3: Fit a ridge regression model to identify important features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge_model.fit(X_train_scaled, y_train)

# Step 4: Get coefficients from ridge regression model
ridge_coefs = pd.Series(ridge_model.coef_, index=X.columns)

# Step 5: Identify most important features
most_important_features = ridge_coefs.abs().sort_values(ascending=False).index[:5]
print(f"Top 4 Features According to Ridge:\n{most_important_features}\n\n")

# Step 6: Fit logistic regression model with most important features
reduced_logit = LogisticRegression()
reduced_logit.fit(X_train[most_important_features], y_train)

# Step 7: Evaluate the final model
accuracy_reduced = reduced_logit.score(X_test[most_important_features], y_test)
print(f"Accuracy of reduced model: {accuracy_reduced}\n\n")

# Step 8: Interpret results of model with higher accuracy
if accuracy_full > accuracy_reduced:
    print("Full model is more accurate")
    print("Coefficients:", full_logit.coef_)
elif accuracy_full < accuracy_reduced:
    print("Coefficients:", reduced_logit.coef_)
else:
    print("They are equally accurate.")