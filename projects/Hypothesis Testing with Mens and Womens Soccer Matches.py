# Code for Datacamp Project: "Hypothesis Testing with Mens and Womens Soccer Matches"

# Import necessary packages
import pandas as pd
import numpy as np
from scipy import stats

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


# Read the data
m_scores = pd.read_csv('men_results.csv')
f_scores = pd.read_csv('women_results.csv')

# Explore the data
explore_df(m_scores, 'all')
explore_df(f_scores, 'all')


# Task: Perform an appropriate hypothesis test to 
# determine the p-value, and hence result, of whether
# to reject or fail to reject the null hypothesis that 
# the mean number of goals scored in women's international
# soccer matches is the same as men's. 
# Use a 10% significance level.

# Req. 1-2: For this analysis, you'll use Official FIFA World
# Cup matches since 2002-01-01.
# Note: You'll also assume that each match is fully 
# independent, i.e., team form is ignored.

# Req. 3: The p-value and the result of the test must
# be stored in a dictionary called 'result_dict'
# in the form:
# result_dict = {"p_val": p_val, "result": result}
# where p_val is the p-value and result is either
# the string "fail to reject" or "reject", 
# depending on the result of the test.



# Understand 'tournament' var
m_tournaments = m_scores['tournament'].unique()
f_tournaments = f_scores['tournament'].unique()
print(f"m_scores['tournament'] Unique values:\n{m_tournaments}\n\nf_scores['tournament'] Unique values:\n{f_tournaments}")


# lets see how many FIFA variations there are in each df
m_tournaments_fifa = [tournament for tournament in m_tournaments if 'FIFA' in tournament]
f_tournaments_fifa = [tournament for tournament in f_tournaments if 'FIFA' in tournament]

print(f"FIFA-related tournaments (Women):\n{f_tournaments_fifa}\n\nFIFA-related tournaments (Men):\n{m_tournaments_fifa}")

# since req.1 explicitly specifies 'FIFA World Cup' we will focus solely on that



# apply modifications to both dfs at once
# Req. 1, Req. 2 & Pre-hypothesis testing step
dfs_to_mod = {
    'm_scores': m_scores,
    'f_scores': f_scores
} 

for df_name, df in dfs_to_mod.items():
    # Req. 1: filter df so that only 'FIFA World Cup' tournaments remain
    # + remove any columns not needed for hypothesis testing/datacamp task
    df = df[df['tournament'] == 'FIFA World Cup'][['date', 'home_score', 'away_score']]
    
    # Req. 2:
    # date: make datetime (both dataframes)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    # filter df so that matches start from 2002-01-01 (Req. 2)
    df = df[df['date'] > '2002-01-01']
    
    # Pre-hypothesis testing
    # Create total_score column (used for analysis)
    df['total_score'] = df['home_score'] + df['away_score']

    # Update the original DataFrame in the dictionary
    dfs_to_mod[df_name] = df

# assign to original df objects
m_scores = dfs_to_mod['m_scores']
f_scores = dfs_to_mod['f_scores']

# sanity check
explore_df(m_scores, 'info')
explore_df(f_scores, 'info')



# Decide on parametric vs. non-parametric

# Shapiro-Wilk test for normality
stat_m, p_m = stats.shapiro(m_scores['total_score'])
stat_f, p_f = stats.shapiro(f_scores['total_score'])
print("Shapiro-Wilk test p-value for men's scores:", p_m)
print("Shapiro-Wilk test p-value for women's scores:", p_f)

if (p_m < 0.05) | (p_f < 0.05): # NON-PARAMETRIC APPROACH
    print(f"---> Result: Normality assumption is violated.\n\n")

    # perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(
        x=f_scores['total_score'],
        y=m_scores['total_score'],
        alternative="greater"
    )
    print("Mann-Whitney U test p-value:", p_value)

    # Interpret the results
    alpha = 0.10  # significance level
    if p_value < alpha:
        print("Reject null hypothesis: There is a significant difference between the means.")
        
        # create data structure for answer
        result_dict = {
            "p_val": p_value, "result": 'reject'
        }

        print(f"\n\nAnswer stored in 'result_dict':\n{result_dict}")
    else:
        print("Fail to reject null hypothesis: There is no significant difference between the means.")
        
        # create data structure for answer
        result_dict = {
            "p_val": p_value, "result": 'fail to reject'
        }

        print(f"\n\nAnswer stored in 'result_dict':\n{result_dict}")

    
else: # PARAMETRIC APPROACH
    print(f"---> Result: Normality assumption is met! A parametric test is appropriate.\n\n")
    
    # independent t-test
    t_statistic, p_value = stats.ttest_ind(
        m_scores['total_score'],
        f_scores['total_score']
    )

    # print the results
    alpha = 0.10  # significance level per task req.
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)

    # auto report results
    if p_value < alpha:
        print("Reject null hypothesis: There is a significant difference between the means.")

        # create data structure for answer
        result_dict = {
            "p_val": p_value, "result": 'reject'
        }

        print(f"\n\nAnswer stored in 'result_dict':\n{result_dict}")
    else:
        print("Fail to reject null hypothesis: There is no significant difference between the means.")

        # create data structure for answer
        result_dict = {
            "p_val": p_value, "result": 'fail to reject'
        }

        print(f"\n\nAnswer stored in 'result_dict':\n{result_dict}")