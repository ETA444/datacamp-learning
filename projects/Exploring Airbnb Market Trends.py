# Code for Datacamp Project: "Exploring Airbnb Market Trends"

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


# Import datasets
# airbnb_price: listing_id, price, nbhood_full
airbnb_price = pd.read_csv('data/airbnb_price.csv')

# airbnb_last_review: listing_id, host_name, last_review
airbnb_last_review = pd.read_csv('data/airbnb_last_review.tsv', sep='\t') 

# airbnb_room_type: listing_id, description, room_type
airbnb_room_type = pd.read_excel('data/airbnb_room_type.xlsx') # excel


# Joining datasets into: airbnb

# Prior to join (1-2):
# (1) Inspect shapes: namely check if same amount of rows
shapes_prejoin = {
    'airbnb_price':airbnb_price.shape,
    'airbnb_last_review':airbnb_last_review.shape,
    'airbnb_room_type':airbnb_room_type.shape
}
print(f"Shapes Pre-Join: {shapes_prejoin}\n")

# (2) Inspect listing_id: check if id format is the same
print("listing_id format in each table: ",
    airbnb_price['listing_id'][0],
    airbnb_last_review['listing_id'][0],
    airbnb_room_type['listing_id'][0]
) # may even be in the same order

# Join - INNER JOIN: airbnb_price + airbnb_last_review
airbnb_p_lr = pd.merge(
    airbnb_price, airbnb_last_review, 
    on='listing_id', how='inner'
)

airbnb = pd.merge(
    airbnb_p_lr, airbnb_room_type,
    on = 'listing_id', how='inner'
)

# sanity check
shape_postjoin = {'airbnb':airbnb.shape}
print(f"\nShape Post-Join: {shape_postjoin}\n\n\n")

# explore our new df
explore_df(airbnb, 'all')
# observations:
# (1) price is a string with format '# dollars', e.g. 225 dollars; 
#     it will benefit from being changed to int with just the number
# (2) nbhood_full could benefit from being split into nbhood and borough
# (3) last_review is a string of date with format 'm dd yyyy' will benefit 
#     from being changed to datetime
# (4) room_type will benefit from being changed to categorical furthermore,
#     there are repeating categories but with lower/uppercase, needs to be made uniform

# Pre-Datacamp Tasks

# (1) fix airbnb.price var: make int

# sanity check before
print(f"Price variable before: {airbnb.price[0]}, {airbnb.price.dtype}")

# transform: .replace .astype
airbnb['price'] = (
    airbnb['price']
    .str.replace(' dollars', '')
    .astype(int)
)

# sanity check after
print(f"\nPrice variable after: {airbnb.price[0]}, {airbnb.price.dtype}")

# (2) nbhood_full could benefit from being split into nbhood and borough

# sanity check before, format: (borough, neighborhood)
print(f"nbhood_full before: row1: {airbnb.nbhood_full[0]}, type: {airbnb.nbhood_full.dtype}\n")

# create new seperate variables: borough and nbhood
borough = []
nbhood = []

for index, row in airbnb.iterrows():
    borough_nbhood = row['nbhood_full'].split(', ')
    borough.append(borough_nbhood[0])
    nbhood.append(borough_nbhood[1])

# sanity check inter, check lists
print(f"borough list: length is {len(borough)}; members 1-5: {borough[0:6]}\n\nnbhood list: length is {len(nbhood)}; members 1-5: {nbhood[0:6]}\n\n")

# add the columns: borough and nbhood
airbnb['borough'] = borough
airbnb['nbhood'] = nbhood

# drop the column: nbhood_full
airbnb = airbnb.drop(columns=['nbhood_full'])

# sanity check
print(f"Transformed df: airbnb \n\n")
explore_df(airbnb, 'all')

# (3) make last_review datetime: '%B %d %Y'
# sanity check before
last_review_before = airbnb['last_review'][0:6]
print(f"last_review (rows 0-6) before:\n{last_review_before} \n")

# last_review to datetime
airbnb['last_review'] = pd.to_datetime(
    airbnb['last_review'], 
    format='%B %d %Y'
)

# sanity check after
last_review_after = airbnb['last_review'][0:6]
print(f"last_review (rows 0-6) after:\n{last_review_after} \n")

# (4) room_type:
# changed to categorical
# repeating categories need to be made uniform

# inspect problematic repeating categories
categories_before = airbnb.room_type.unique().tolist()
print(f"\nroom_type categories before:\n{categories_before}")

# sanity check before
airbnb_rows1to5_before = airbnb['room_type'].iloc[0:6]
print(f"\nRows in airbnb df before:\n{airbnb_rows1to5_before}")

# make room_type uniform
airbnb['room_type'] = airbnb['room_type'].apply(
    lambda room_type: room_type.lower().capitalize()
)

# make room_type categorical
airbnb['room_type'] = airbnb['room_type'].astype('category')

# sanity check after
airbnb_rows1to5_after = airbnb['room_type'].iloc[0:6]
print(f"\nRows in airbnb df after:\n{airbnb_rows1to5_after}")



# Datacamp Tasks:

# Task 1. What are the dates of the earliest and most recent reviews? 
# Store these values as two separate variables with your preferred names.

# earliest review
oldest_review = airbnb['last_review'].min()

# most recent review
newest_review = airbnb['last_review'].max()

# print
print(f"\nThe earliest review was on: {oldest_review}\nThe most recent review was on: {newest_review}\n")



# Task 2. How many of the listings are private rooms? Save this into any variable.

# private_rooms
private_rooms = airbnb['room_type'].value_counts()[1]

# print
print(airbnb['room_type'].value_counts())



# Task 3. What is the average listing price?
# Round to the nearest penny and save into a variable.

# average_price
average_price = round(airbnb['price'].mean(),2)

# bonus: average prices per room_type
avgprice_shared, avgprice_private, avgprice_entire = [
    airbnb[airbnb['room_type'] == 'Shared room']['price'].mean(),
    airbnb[airbnb['room_type'] == 'Private room']['price'].mean(),
    airbnb[airbnb['room_type'] == 'Entire home/apt']['price'].mean()
]

# print
print(f"Average listing price for all room types: ${average_price}\nAverage listing price for Shared rooms: ${avgprice_shared:.2f}\nAverage listing price for Private rooms: ${avgprice_private:.2f}\nAverage listing price for Entire homes\apts: ${avgprice_entire:.2f}")



# Task 4. Combine the new variables into one DataFrame called review_dates with four columns in the following order: first_reviewed, last_reviewed, nb_private_rooms, and avg_price. The DataFrame should only contain one row of values.

# new dataframe: review_dates
review_dates = pd.DataFrame({
    'first_reviewed':[oldest_review],
    'last_reviewed':[newest_review],
    'nb_private_rooms':[private_rooms],
    'avg_price':[average_price],
}
)

# sanity check df
print(review_dates)