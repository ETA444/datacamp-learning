# Code for Datacamp Project: "Investigating Netflix Movies"

# Importing pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Load data, check it
netflix_df = pd.read_csv("netflix_data.csv")

# func: explore_df
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
        - 'all': Display both head(), describe(), and info().

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
    elif method.lower() == "all":
        print("<<______HEAD______>>")
        pd.set_option('display.max_columns', None)
        print(df.head())
        pd.reset_option('display.max_columns')
        print(f"\n\n<<______DESCRIBE______>>")
        print(df.describe())
        print(f"\n\n<<______INFO______>>")
        print(df.info())
    else:
        print("Methods: 'desc', 'head', 'info' or 'all'")

# test documentation
print(explore_df.__doc__)

# Get to know the data
explore_df(netflix_df, "all")

# Tasks:

# (1) Filter the data to remove TV shows and store as netflix_subset.
netflix_subset = netflix_df[~(netflix_df['type'] == "TV Show")]

# sanity check (1)
print(netflix_subset["type"].unique())


# (2) Investigate the Netflix movie data, keeping only the columns "title", "country", "genre", "release_year", "duration", and saving this into a new DataFrame called netflix_movies.
col_to_keep = ["title", "country", "genre", "release_year", "duration"]
netflix_movies = netflix_subset[col_to_keep]

# sanity check
exploreDF(netflix_movies, "info")


# (3.1) Filter netflix_movies to find the movies that are strictly shorter than 60 minutes, saving the resulting DataFrame as short_movies; 
less_than_60 = netflix_movies["duration"] < 60
short_movies = netflix_movies[less_than_60]

# sanity check 
exploreDF(short_movies["duration"], "desc")

# (3.2) Inspect the result to find possible contributing factors.

# inspect whether a specific genre pulls the mean down
short_movies.groupby("genre")["duration"].mean().sort_values()
# Dramas, Horror and Children's movies have the lowest duration

# inspect whether a specific country pulls the mean down
short_movies.groupby("country")["duration"].mean().sort_values()
# Pakistan (16), Georgia (24) and Namibia (29) have particularly low means


# (4.1) Using a for loop and if/elif statements, iterate through the rows of netflix_movies and assign colors of your choice to four genre groups ("Children", "Documentaries", "Stand-Up", and "Other" for everything else). Save the results in a colors list.
colors = []
for index, row in netflix_movies.iterrows():
    if row['genre'] == 'Children':
        colors.append('green')
    elif row['genre'] == 'Documentaries':
        colors.append('blue')
    elif row['genre'] == 'Stand-up':
        colors.append('yellow')
    else:
        colors.append('pink')
        
# sanity check
len(colors) == netflix_movies.shape[0]

# (4.2) Initialize a matplotlib figure object called fig and create a scatter plot for movie duration by release year using the colors list to color the points and using the labels "Release year" for the x-axis, "Duration (min)" for the y-axis, and the title "Movie Duration by Year of Release".
x = netflix_movies["release_year"]
y = netflix_movies["duration"]
fig = plt.figure()
plt.scatter(x, y, c=colors)
plt.xlabel("Release year")
plt.ylabel("Duration (min)")
plt.title("Movie Duration by Year of Release")
plt.show()


# (5) After inspecting the plot, answer the question "Are we certain that movies are getting shorter?" by assigning either "yes" or "no" to the variable answer.
answer = "no"


# (Bonus) Answer the question in a statistical way.
# Note: not part of Datacamp exercise but an easy concise way to answer the question in a statistical way.
import statsmodels.api as sm
x = netflix_movies["release_year"]
y = netflix_movies["duration"]

# add intercept + fit OLS linear regression
x = sm.add_constant(x)
lr = sm.OLS(y, x).fit()

# print summary
print(lr.summary())

# my answer:
# The linear regression model suggests a statistically significant negative relationship between 'release_year' and 'duration', indicating that, on average, movies may be getting shorter over the years. 
# However, the low R-squared value suggests that the model explains only a small portion of the variability in movie duration.
# Additionally, the large condition number indicates potential multicollinearity, and further investigation may be needed.