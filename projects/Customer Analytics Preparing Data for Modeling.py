# Code for Datacamp Project: "Customer Analytics: Preparing Data for Modeling"

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Import data and explore
ds_jobs = pd.read_csv('customer_train.csv')

explore_df(ds_jobs, "all")



# Task 1. Columns containing integers must be stored as 32-bit integers (int32).
task1_conv = ['student_id', 'training_hours', 'job_change']
ds_jobs_clean = ds_jobs[task1_conv].astype('int32')

# sanity check
ds_jobs_clean[task1_conv].dtypes



# Task 2. Columns containing floats must be stored as 16-bit floats (float16).
task2_conv = ['city_development_index']
ds_jobs_clean[task2_conv] = ds_jobs[task2_conv].astype('float16')

# sanity check
ds_jobs_clean[task2_conv].dtypes



# Task 3. Columns containing nominal categorical data must be stored as the category data type.
task3_conv = ['city', 'gender', 'major_discipline', 'company_type']
ds_jobs_clean[task3_conv] = ds_jobs[task3_conv].astype('category')

# sanity check
ds_jobs_clean[task3_conv].dtypes



# Task 4. Columns containing ordinal categorical data must be stored as ordered categories, and not mapped to numerical values, with an order that reflects the natural order of the column.

# convert: education_level
# explore levels
print(ds_jobs['education_level'].unique()) # there are NAs

# if NAs in row are less than 5%: drop
# explore_df(ds_jobs, "na") # 2.4%

# drop NAs in: education_level
#ds_jobs["education_level"] = ds_jobs["education_level"].dropna()

# sanity check
#print(ds_jobs['education_level'].unique())

# create order
edu_order = ['Primary School','High School','Graduate', 'Masters', 'Phd']

# update the variable: education_level
ds_jobs_clean["education_level"] = pd.Categorical(
    ds_jobs["education_level"],
    categories = edu_order, # specify order
    ordered = True # make ordinal
)

# sanity check
print(ds_jobs_clean['education_level'].unique())


# convert: experience
# explore levels
print(ds_jobs['experience'].unique()) # there are NAs

# if NAs in row are less than 5%: drop
# explore_df(ds_jobs, "na") # 2.4%

# drop NAs in: experience
#ds_jobs["experience"] = ds_jobs["experience"].dropna()

# sanity check
#print(ds_jobs['experience'].unique())

# create order
exp_order = [
    '<1','1','2', '3', '4', '5', '6', '7', '8',
    '9', '10', '11', '12', '13', '14', '15', '16',
    '17', '18', '19', '20', '>20'
]

# update the variable: education_level
ds_jobs_clean["experience"] = pd.Categorical(
    ds_jobs["experience"],
    categories = exp_order, # specify order
    ordered = True # make ordinal
)

# sanity check
print(ds_jobs_clean['experience'].unique())


# convert: company_size
# explore levels
print(ds_jobs['company_size'].unique()) # there are NAs

# if NAs in row are less than 5%: drop
# explore_df(ds_jobs, "na") # 2.4%

# drop NAs in: company_size
#ds_jobs["company_size"] = ds_jobs["company_size"].dropna()

# sanity check
#print(ds_jobs['company_size'].unique())

# create order
compsize_order = [
    '<10', '10-49', '50-99', '100-499', '500-999',
    '1000-4999', '5000-9999', '10000+'
]

# update the variable: company_size
ds_jobs_clean["company_size"] = pd.Categorical(
    ds_jobs["company_size"],
    categories = compsize_order, # specify order
    ordered = True # make ordinal
)

# sanity check
print(ds_jobs_clean['company_size'].unique())


# convert: last_new_job
# explore levels
print(ds_jobs['last_new_job'].unique()) # there are NAs

# if NAs in row are less than 5%: drop
# explore_df(ds_jobs, "na") # 2.4%

# drop NAs in: company_size
#ds_jobs["last_new_job"] = ds_jobs["last_new_job"].dropna()

# sanity check
#print(ds_jobs['last_new_job'].unique())

# create order
lnj_order = ['1', '2', '3', '4', '>4', 'never']

# update the variable: company_size
ds_jobs_clean["last_new_job"] = pd.Categorical(
    ds_jobs["last_new_job"],
    categories = lnj_order, # specify order
    ordered = True # make ordinal
)

# sanity check
print(ds_jobs_clean['last_new_job'].unique())


# convert: relevant_experience
# explore levels
print(ds_jobs['relevant_experience'].unique())

# create order
relexp_order = ['No relevant experience', 'Has relevant experience']

# update the variable: relevant_experience
ds_jobs_clean["relevant_experience"] = pd.Categorical(
    ds_jobs["relevant_experience"],
    categories = relexp_order, # specify order
    ordered = True # make ordinal
)

# sanity check
print(ds_jobs_clean['relevant_experience'].unique())


# convert: enrolled_university
# explore levels
print(ds_jobs['enrolled_university'].unique()) # there are NAs

# if NAs in row are less than 5%: drop
# explore_df(ds_jobs, "na") # 2.4%

# drop NAs in: enrolled_university
#ds_jobs["enrolled_university"] = ds_jobs["enrolled_university"].dropna()

# sanity check
#print(ds_jobs['enrolled_university'].unique())

# create order
enruni_order = [
    'no_enrollment', 'Part time course', 'Full time course'
]

# update the variable: enrolled_university
ds_jobs_clean["enrolled_university"] = pd.Categorical(
    ds_jobs["enrolled_university"],
    categories = enruni_order, # specify order
    ordered = True # make ordinal
)

# sanity check
print(ds_jobs_clean['enrolled_university'].unique())



# Task 5. The columns of ds_jobs_clean must be in the same order as the original dataset.

# sanity check: compare column order BEFORE
print(f"ORDER BEFORE REINDEX: \n")
print(ds_jobs.columns == ds_jobs_clean.columns)

# reindex ds_jobs_clean to match order in ds_jobs
ds_jobs_clean = ds_jobs_clean.reindex(columns=ds_jobs.columns)

# sanity check: compare column order AFTER
print(f"\n ORDER AFTER REINDEX: \n")
print(ds_jobs.columns == ds_jobs_clean.columns)



# Task 6. The DataFrame should be filtered to only contain students with 10 or more years of experience at companies with at least 1000 employees, as their recruiter base is suited to more experienced professionals at enterprise companies.

# sanity check: shape pre-filter
print(f"SHAPE PRE-FILTER: \n")
prefilter_shape = ds_jobs_clean.shape
print(prefilter_shape)

# filter
ds_jobs_clean = ds_jobs_clean[
    (ds_jobs_clean['experience'] >= '10') &
    (ds_jobs_clean['company_size'] >= '1000-4999')
]

# sanity check: shape post-filter
print(f"\nSHAPE POST-FILTER: \n")
postfilter_shape = ds_jobs_clean.shape
print(postfilter_shape)

# reduction %
pool_reduc = (
     (((postfilter_shape[0]) / prefilter_shape[0]) - 1) * 100
)
print(f"\nEmployee Pool Size Change: {pool_reduc:.2f}%")



# Bonus: If you call .info() or .memory_usage() methods on ds_jobs and ds_jobs_clean after you've preprocessed it, you should notice a substantial decrease in memory usage.

# check memory usage
memuse_before = ds_jobs.memory_usage()
memuse_after = ds_jobs_clean.memory_usage()
memuse_meanchange = (
    ((memuse_after.mean() / memuse_before.mean()) - 1) * 100
)
print(f"The mean memory usage before cleaning the data was {memuse_before.mean():.0f} bytes. After we cleaned it it became {memuse_after.mean():.0f} bytes. Which is a change of {memuse_meanchange:.2f}%.")