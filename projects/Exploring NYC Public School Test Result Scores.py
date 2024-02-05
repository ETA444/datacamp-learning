# Code for Datacamp Project: "Exploring NYC Public School Test Result Scores"

# Re-run this cell 
import pandas as pd
import numpy as np

# Read in the data
schools = pd.read_csv("schools.csv")

# Fun: explore_df
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

# Get to know the data
explore_df(schools, "all")


# Question 1. What NYC schools have the best MATH results?
# Req. 1: The best MATH results are at least 80% of the *maximum possible score* for math.
# Req. 2: Save your results in a pandas DataFrame called best_math_schools, including "school_name" and "average_math" columns, sorted by "average_math" in descending order.

# create percent_X column - e.g. for math: (average_math / 800) * 100
schools["percent_math"] = (schools["average_math"] / 800) * 100
schools["percent_reading"] = (schools["average_reading"] / 800) * 100
schools["percent_writing"] = (schools["average_writing"] / 800) * 100

# create average_percent column - the mean of the 3 percent columns
schools["average_percent"] = np.mean(schools[['percent_math', 'percent_reading', 'percent_writing']], axis=1)

# sanity check
explore_df(schools, "head")

# columns of interest according to Req. 2
columns_of_interest_q1 = ['school_name', 'average_math']

# condition to include according to Req. 2 (at least 80% math score)
condition_math_80 = schools['percent_math'] >= 80

# create the new df: best_math_schools
best_math_schools = schools.loc[condition_math_80, columns_of_interest_q1]

# sort the new df according to Req. 2 (descending order)
best_math_schools = best_math_schools.sort_values(by='average_math', ascending=False)

# sanity check: check if lowest score is at least 80% (PASSED)
print((best_math_schools['average_math'].min() / 800) * 100)


# Question 2. What are the top 10 performing schools based on the combined SAT scores?
# Req. 1: Save your results as a pandas DataFrame called top_10_schools containing the "school_name" and a new column named "total_SAT", with results ordered by "total_SAT" in descending order.

# create total_SAT column: average_math + average_reading + average_writing
schools["total_SAT"] = schools[["average_math", "average_reading", "average_writing"]].sum(axis=1)

# sanity check
schools[["average_math", "average_reading", "average_writing", "total_SAT"]].head()

# columns of interest according to Req. 1
columns_of_interest_q2 = ["school_name", "total_SAT"]

# create the new df: top_10_schools
top_10_schools = (
    schools[["school_name", "total_SAT"]]
    .sort_values(by="total_SAT", ascending=False)
    .head(10)
)

print(top_10_schools)


# Question 3. What borough has the largest standard deviation in the combined SAT score?
# Req. 1: Save your results as a pandas DataFrame called largest_std_dev.
# Req. 2: The DataFrame should contain "borough" as the index, and the following columns:
  # "num_schools" for the number of schools in the borough.
  # "average_SAT" for the mean of "total_SAT".
  # "std_SAT" for the standard deviation of "total_SAT".
# Req. 3: Round all numeric values to two decimal places.

# create std_SAT column: this is the std of total_SAT per borough
schools["std_SAT"] = (
    schools
    .groupby('borough')['total_SAT']
    .transform('std') # calc std per group
)

# what is the borough with the largest std?: Manhattan
manhattan = schools[schools["std_SAT"] == schools["std_SAT"].max()][['borough', 'total_SAT', 'std_SAT']]

# Achieve Req. 1-3
largest_std_dev = pd.DataFrame({
    'borough':manhattan['borough'].unique(),
    'num_schools':manhattan.shape[0],
    'average_SAT':manhattan['total_SAT'].mean().round(2),
    'std_SAT':manhattan['std_SAT'].unique().round(2)
}).set_index("borough") # Req. 2

print(largest_std_dev)