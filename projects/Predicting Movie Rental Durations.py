# Code for Datacamp Project: "Predicting Movie Rental Durations"

# Packages and functions

# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ( #linear models
    LinearRegression, Ridge, Lasso, LogisticRegression
)
from sklearn.tree import ( # tree-based models
    DecisionTreeClassifier, DecisionTreeRegressor
)
from sklearn.ensemble import ( # ensemble models
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)


# Define explore_df
def explore_df(df, method='all'):
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


# Read and explore the data
rental_info = pd.read_csv('rental_info.csv')
explore_df(rental_info)

# Findings/todo based on explore_df:
# rental_date & return_date not datetime
date_columns = ['rental_date', 'return_date']
rental_info[date_columns] = rental_info[date_columns].apply(pd.to_datetime)

# release year is a float, for memory optimization make int
rental_info['release_year'] = rental_info['release_year'].astype(int)

# sanity check
print(f"\n[rental_info dtypes after transformations]:\n\n{rental_info.dtypes}\n\n")



# DataCamp Tasks
"""
Task 1. Create a column named "rental_length_days" using the
columns "return_date" and "rental_date", and add it to the
pandas DataFrame.

This column should contain information on how many days a DVD
has been rented by a customer.
"""
# creating: rental_info['rental_length_day']
rental_info['rental_length_days'] = (rental_info['return_date'] - rental_info['rental_date']).dt.days

# sanity check
explore_df(rental_info['rental_length_days'])


"""
Task 2. Create two columns of dummy variables from
"special_features", which takes the value of 1 when:
- The value is "Deleted Scenes", storing as a column called "deleted_scenes".
- The value is "Behind the Scenes", storing as a column called "behind_the_scenes".
"""
# 2.1 - understand the column: special_features
print(f"{rental_info['special_features'].value_counts()}\n\n")

# 2.2 - create deleted_scenes dummy
# logic: value is 1 if Deleted Scenes is one of the features
# note: DataCamp is vague clear, it could be they want that
# Deleted Scenes is the ONLY feature, or that it is PRESENT
rental_info['deleted_scenes'] = (
    rental_info['special_features']
    .apply(lambda x: 'Deleted Scenes' in x) # return true if..
    .astype(int) # make it 1-0, instead of True-False
)

# sanity check
print(rental_info.loc[[0, 15859],['special_features', 'deleted_scenes']])


# 2.3 - create behind_the_scenes dummy
# logic: value is 1 if Behind the Scenes is one of the features
rental_info['behind_the_scenes'] = (
    rental_info['special_features']
    .apply(lambda x: 'Behind the Scenes' in x)
    .astype(int)
)

# sanity check
print(rental_info.loc[[0, 15859],['special_features', 'behind_the_scenes']])


"""
Task 3. Make a pandas DataFrame called X containing all the
appropriate features you can use to run the regression models,
avoiding columns that leak data about the target.
"""
# create X: exclude rental_date, return_date, rental_length_days, special_features
X = rental_info.drop(
    columns=[
        'rental_date', 'return_date',
        'rental_length_days', 'special_features'
    ])


"""
Task 4. Choose the "rental_length_days" as the target column
and save it as a pandas Series called y.
"""
# create y: rental_length_days
y = rental_info['rental_length_days']


"""
Task 5. Split the data into X_train, y_train, X_test, and y_test train and test sets, avoiding any features that leak
data about the target variable, and include 20% of the total
data in the test set.

Set random_state to 9 whenever you use a function/method
involving randomness, for example, when doing a test-train split.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 9
)


"""
Task 6. Recommend a model yielding a mean squared error (MSE)
less than 3 on the test set.

Save the model you would recommend as a variable named
best_model, and save its MSE on the test set as best_mse.
"""
# models to test
sklearn_models_continuousDV = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor()
}

# save model name and corresponding mse value
results = {}

# test the models
for model_name, model_instance in sklearn_models_continuousDV.items():
    model_instance.fit(X_train, y_train) # fit
    y_pred = model_instance.predict(X_test) # predict
    mse = mean_squared_error(y_test, y_pred) # evaluate (mse)
    results[model_instance] = mse # save result
    print(f"[{model_name}] MSE: {mse:.2f}\n") # inform (console)
    
# save best_model
best_model = min(results, key=lambda x: results[x])
best_mse = results[best_model]
print(f"\n\nBest Model: {best_model} [MSE = {best_mse:.2f}]")