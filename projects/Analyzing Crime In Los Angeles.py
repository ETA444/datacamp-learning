# Code for Datacamp Project: "Analyizing Crime In Los Angeles"

# Import required libraries
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
crimes = pd.read_csv("crimes.csv", parse_dates=["Date Rptd", "DATE OCC"], dtype={"TIME OCC": str})

explore_df(crimes, "all")



# Question 1. Which hour has the highest frequency of crimes? Store as an integer variable called peak_crime_hour.

# Add ':' between the hour and minute in TIME OCC
crimes['TIME OCC'] = (
    crimes['TIME OCC']
    .str.slice_replace(2, 0, ':') # add : between H and M
    +':00' # add seconds at the end (useful for timedelta)
)

# sanity check
print(crimes['TIME OCC'][0])

# Make TIME OCC a timedelta object 
crimes['TIME OCC'] = pd.to_timedelta(crimes['TIME OCC']+'')

# Create new column: DATETIME_OCC
# Combination of TIME OCC and DATE OCC as one datetime
crimes['DATETIME_OCC'] = crimes['DATE OCC'] + crimes['TIME OCC']

# sanity check
print(crimes['DATETIME_OCC'].dtype, crimes['DATETIME_OCC'][0])

# crime frequency per hour
crime_freq_per_hour = crimes['DATETIME_OCC'].dt.hour.value_counts()

# Question 1. Answer
peak_crime_hour = 12



# Question 2. Which area has the largest frequency of night crimes (crimes committed between 10pm and 3:59am)? Save as a string variable called peak_night_crime_location.

# create new df: crimes_2200_to_0359
# df containing crimes only between those hours

# filter from 22:00 to 23:59
filter_2200_to_2359 = (
    (crimes['DATETIME_OCC'].dt.hour >= 22) &
    (crimes['DATETIME_OCC'].dt.hour <= 23)
)

# filter from 00:00 to 03:59
filter_0000_to_0359 = (
    (crimes['DATETIME_OCC'].dt.hour >= 0) &
    (crimes['DATETIME_OCC'].dt.hour <= 3)
)

# combined time filter
time_filter = filter_2200_to_2359 | filter_0000_to_0359

# create new df: crimes_2200_to_0359
crimes_2200_to_0359 = crimes[time_filter]

# sanity checks (1-3)
# (1) check individual filters
print("Rows between 10pm and 11:59pm:")
print(crimes[filter_2200_to_2359].head())
print("\nRows between 12am and 3:59am:")
print(crimes[filter_0000_to_0359].head())

# (2) verify combined time filter
print("\nCombined time filter:")
print(crimes[time_filter].head())

# (3) inspect resulting df
print("\nResulting DataFrame (crimes_2200_to_0359):")
print(crimes_2200_to_0359.head())

# Question 2. Answer
peak_night_crime_location = (
    crimes_2200_to_0359['AREA NAME']
    .value_counts() # count crimes/rows per area name
    .idxmax() # get id/name of highest value
)




# Question 3. Identify the number of crimes committed against victims by age group (0-17, 18-25, 26-34, 35-44, 45-54, 55-64, 65+). Save as a pandas Series called victim_ages.

# create new column: AGE_GROUP (1-2)
# (1) age group bins and labels for pd.cut
age_bins = [0,17,25,34,44,54,64,float('inf')]
age_labels = [
    '0-17', 
    '18-25', 
    '26-34', 
    '35-44', 
    '45-54', 
    '55-64', 
    '65+'
]

# (2) use pd.cut to create AGE_GROUP
crimes["AGE_GROUP"] = pd.cut(
    crimes["Vict Age"],
    bins=age_bins,
    labels=age_labels
)

# sanity check: there is a NaN further investigate
print(crimes["AGE_GROUP"].unique())

# further investigation shows
# that there are ages 0, -1 and -2 in the df
print(crimes[crimes["AGE_GROUP"].isna()]["Vict Age"].unique())

# Question 3. Answer
victim_ages = (
    crimes["AGE_GROUP"]
    .value_counts() # count crimes per age group
    .sort_index() # sort by age group/index
)

# sanity check
print(victim_ages)