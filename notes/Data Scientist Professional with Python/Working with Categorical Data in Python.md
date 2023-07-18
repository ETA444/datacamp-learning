

* * *
## .Categorical()

The `.Categorical()` function in pandas is used to create a categorical data type for a Series or DataFrame column. Categorical data type represents variables that have a fixed number of distinct values or categories. It can provide benefits in terms of memory efficiency and enhanced functionality for working with categorical data.

**Function signature:**
```python
pandas.Categorical(values, categories=None, ordered=None, dtype=None)
```

**Parameters:**
- `values`: Specifies the data to be converted to a categorical data type. It can be a Series, array-like, or iterable.
- `categories` (optional): Specifies the categories or unique values of the categorical data. If not provided, the distinct values from the input data are used as categories.
- `ordered` (optional): Specifies whether the categories have a specific order or not. If `True`, the categories are treated as ordered. If `False`, the categories are treated as unordered. If not specified, the order is inferred based on the order of appearance of the values or the order of the provided categories.
- `dtype` (optional): Specifies the data type of the resulting categorical data. If not specified, it is inferred based on the input data.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
data = pd.Series(['apple', 'banana', 'orange', 'apple', 'banana'])

# Convert the Series to a categorical data type
cat_data = pd.Categorical(data)

# Display the categorical data
print(cat_data)
```

The resulting `cat_data` Series will be:
```
['apple', 'banana', 'orange', 'apple', 'banana']
Categories (3, object): ['apple', 'banana', 'orange']
```

In the example, the `.Categorical()` function is used to convert the `data` Series into a categorical data type. The resulting `cat_data` Series represents the original data as categorical values, with the distinct values stored as categories.

Categorical data type provides advantages in terms of memory usage and performance when working with data that has a limited number of distinct values. It enables efficient storage by mapping the values to a separate array of categories, reducing the memory footprint. Categorical data type also allows for meaningful comparisons, sorting, and ordering of the values based on the specified order or the order of appearance. It can be especially useful for working with categorical variables in statistical analyses, plotting, and machine learning tasks.

```python
# 1 # Create a dictionary:
adult_dtypes = {"Marital Status": "category"}

# 2 # Set the dtype parameter:
adult = pd.read_csv("data/adult.csv", dtype=adult_dtypes)

# 3 # Check the dtype:
adult["Marital Status"].dtype
```
![Pasted image 20230714224138](/images/Pasted%20image%2020230714224138.png)

***
## .nbytes

The `.nbytes` attribute in pandas is used to retrieve the number of bytes consumed by a DataFrame or Series. It returns the total memory usage of the object in bytes.

**Attribute syntax:**
```python
DataFrame.nbytes
Series.nbytes
```

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {'A': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data)

# Get the memory usage in bytes
memory_usage = df.nbytes

# Display the memory usage
print(memory_usage)
```

The resulting `memory_usage` value will be the total memory usage of the DataFrame `df` in bytes.

The `.nbytes` attribute is useful for assessing the memory footprint of a DataFrame or Series. It can be used to measure the memory consumption of different objects, optimize memory usage, or track memory usage changes during data processing or analysis. By comparing the memory usage before and after certain operations, you can evaluate the impact of those operations on memory efficiency and identify potential opportunities for optimization.

```python
# Why do we use categorical: memory
adult = pd.read_csv("data/adult.csv")
adult['Marital Status'].nbytes
```
![Pasted image 20230714224258](/images/Pasted%20image%2020230714224258.png)
```python
# Why do we use categorical: memory
adult["Marital Status"] = adult['Marital Status'].astype("category")
adult["Marital Status"].nbytes
```
![Pasted image 20230714224355](/images/Pasted%20image%2020230714224355.png)


***
## .Series()

The `.Series()` function in pandas is used to create a one-dimensional labeled array, known as a Series. A Series is a fundamental data structure in pandas that can hold data of various types, including numerical, string, boolean, and datetime values.

**Function signature:**
```python
pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)
```

**Parameters:**
- `data` (optional): Specifies the data for the Series. It can be a list, array-like object, dictionary, scalar value, or another Series.
- `index` (optional): Specifies the labels or indices for the Series. It can be a list, array-like object, or a pandas Index object. If not provided, a default integer index will be used.
- `dtype` (optional): Specifies the data type for the elements of the Series. It can be a NumPy dtype or a string representing a data type.
- `name` (optional): Specifies the name of the Series.
- `copy` (optional): Specifies whether to copy the data or use the input data directly. By default, `copy=False` is used, meaning that the input data is not copied.
- `fastpath` (optional): Specifies whether to use a fastpath for generating the Series. By default, `fastpath=False` is used.

**Example of use:**
```python
import pandas as pd

# Create a Series from a list
data = [10, 20, 30, 40, 50]
series = pd.Series(data)

# Display the Series
print(series)
```

The resulting `series` Series will be:
```
0    10
1    20
2    30
3    40
4    50
dtype: int64
```

In the example, the `.Series()` function is used to create a Series `series` from a list `data`. The resulting Series contains the elements of the list with default integer indices.

Series objects are commonly used in pandas for data manipulation, analysis, and visualization. They provide a labeled and indexed one-dimensional array that can be easily accessed, sliced, and operated upon. Series can be created from various data sources, such as lists, arrays, dictionaries, or by extracting columns from DataFrames. They serve as a building block for more complex data structures, like DataFrames, and provide a convenient way to work with one-dimensional data in pandas.
***
### Exercises

##### Exercise 1
```python
# Create a Series, default dtype
series1 = pd.Series(list_of_occupations)
  
# Print out the data type and number of bytes for series1
print("series1 data type:", series1.dtype)
print("series1 number of bytes:", series1.nbytes)

 # Create a Series, "category" dtype
series2 = pd.Series(list_of_occupations, dtype="category")

# Print out the data type and number of bytes for series2
print("series2 data type:", series2.dtype)
print("series2 number of bytes:", series2.nbytes)
```
![Pasted image 20230714212920](/images/Pasted%20image%2020230714212920.png)

##### Exercise 2
```python
# Create a categorical Series and specify the categories (let pandas know the order matters!)
medals = pd.Categorical(medals_won, categories=['Bronze', 'Silver', 'Gold'], ordered=True)
print(medals)
```
![Pasted image 20230714213648](/images/Pasted%20image%2020230714213648.png)

##### Exercise 3
```python
# Check the dtypes
print(adult.dtypes)
```
![Pasted image 20230714213831](/images/Pasted%20image%2020230714213831.png)

```python
# Create a dictionary with column names as keys and "category" as values
adult_dtypes = {'Workclass':'category',
                'Education':'category',
                'Relationship': 'category',
                'Above/Below 50k': 'category'
}

# Read in the CSV using the dtypes parameter
adult2 = pd.read_csv(
  "adult.csv",
  dtypes=adult_dtypes
)

print(adult2.dtypes)
```

***
## .groupby()

The `.groupby()` function in pandas is used to split a DataFrame into groups based on one or more columns. It is typically followed by an aggregation or transformation operation to compute summary statistics or perform group-wise operations on the data.

**Function signature:**
```python
DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=<no_default>, observed=False, **kwargs)
```

**Parameters:**
- `by`: Specifies the column(s) or key(s) by which the DataFrame should be grouped. It can be a column name, a list of column names, an array, or a dictionary mapping column names to group keys.
- `axis` (optional): Specifies the axis along which the grouping is performed. `axis=0` (default) groups the data vertically (by rows), while `axis=1` groups it horizontally (by columns).
- `level` (optional): Specifies the level(s) of a MultiIndex to be used for grouping.
- `as_index` (optional): Specifies whether to set the grouping columns as the index of the resulting DataFrame. If `True`, the grouping columns become the index. If `False`, they remain as regular columns.
- `sort` (optional): Specifies whether to sort the resulting groups by the group keys. If `True`, the groups are sorted. If `False`, the groups are not sorted.
- `group_keys` (optional): Specifies whether to include the group keys in the resulting DataFrame index or columns.
- `observed` (optional): Specifies whether to use only observed values for grouping. If `True`, only the observed values are used. If `False`, all possible values from the grouping columns are included, even if they are not observed in the data.
- `**kwargs`: Additional keyword arguments that can be passed to control specific aspects of the grouping operation.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [10, 15, 5, 8, 12]
}
df = pd.DataFrame(data)

# Group the DataFrame by 'Category' column
grouped_df = df.groupby('Category')

# Perform aggregation on the grouped data
mean_values = grouped_df['Value'].mean()

# Display the mean values
print(mean_values)
```

The resulting `mean_values` Series will be:
```
Category
A    9.0
B    11.5
Name: Value, dtype: float64
```

In the example, the `.groupby()` function is used to group the DataFrame `df` by the 'Category' column. The resulting `grouped_df` object represents the grouped data. Subsequently, the mean of the 'Value' column is computed for each group using the `.mean()` function.

The `.groupby()` function is a powerful tool in pandas for performing group-wise operations, such as aggregation, transformation, or filtering, on subsets of data. It allows for the analysis of data based on specific categories or groups and provides flexibility in computing summary statistics or applying custom functions to each group. The resulting grouped object can be further manipulated, such as by applying additional aggregation functions, accessing individual groups, or merging the results back into the original DataFrame.

```python
# without groupby
adult = pd.read_csv("data/adult.csv")
adult1 = adult[adult["Above/Below 50k"] == " <=50K"]
adult2 = adult[adult["Above/Below 50k"] == " >50K"]

# with groupby
groupby_object = adult.groupby(by=["Above/Below 50k"])

groupby_object.mean()
```
![Pasted image 20230715200952](/images/Pasted%20image%2020230715200952.png)
```python
adult.groupby(by=["Above/Below 50k"]).mean()
```
![Pasted image 20230715200952](/images/Pasted%20image%2020230715200952.png)

```python
adult.groupby(by=["Above/Below 50k"])['Age', 'Education Num'].sum()
```
![Pasted image 20230715201336](/images/Pasted%20image%2020230715201336.png)

```python
adult.groupby(by=["Above/Below50k", "Marital Status"]).size
```
![Pasted image 20230715201534](/images/Pasted%20image%2020230715201534.png)

***
## .size

In pandas, the `.size` attribute is used to retrieve the number of elements in a DataFrame or Series. It returns the total number of values in the object.

**Attribute syntax:**
```python
DataFrame.size
Series.size
```

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Get the size of the DataFrame
df_size = df.size

# Display the size
print(df_size)
```

The resulting `df_size` value will be the total number of elements in the DataFrame.

In the example, the `.size` attribute is used to retrieve the size of the DataFrame `df`. The resulting value represents the total number of elements in the DataFrame, which is calculated as the product of the number of rows and the number of columns.

The `.size` attribute is useful for determining the overall size or number of elements in a DataFrame or Series. It can be used to check the dimensions of a DataFrame, calculate the memory usage, or perform size-related operations or comparisons. Note that the `.size` attribute returns the total number of elements, including missing or NaN (Not a Number) values.

### Exercises

##### Exercise 1
```python
# Group the adult dataset by "Sex" and "Above/Below 50k"
gb = adult.groupby(by=["Sex", "Above/Below 50k"])

# Print out how many rows are in each created group
print(gb.size)
  
# Print out the mean of each group for all columns
print(gb.mean())
```
![Pasted image 20230716135027](/images/Pasted%20image%2020230716135027.png)

##### Exercise 2
```python
# Create a list of user-selected variables
user_list = ['Education', 'Above/Below 50k']

# Create a GroupBy object using this list
gb = adult.groupby(by=user_list)

# Find the mean for the variable "Hours/Week" for each group - Be efficient!
print(gb["Hours/Week"].mean())
```
![Pasted image 20230716135657](/images/Pasted%20image%2020230716135657.png)

***
## .cat

In pandas, the `.cat` accessor is used to access categorical data and perform operations specific to categorical data types. It provides access to a set of methods and properties that are useful for working with categorical columns in a DataFrame.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {'Category': ['A', 'B', 'A', 'B', 'A']}
df = pd.DataFrame(data)

# Convert the 'Category' column to a categorical data type
df['Category'] = df['Category'].astype('category')

# Access categorical methods using .cat accessor
df['Category'].cat.categories = ['Category 1', 'Category 2']  # Rename categories
df['Category'] = df['Category'].cat.add_categories('Category 3')  # Add a new category
df['Category'] = df['Category'].cat.remove_unused_categories()  # Remove unused categories

# Display the modified DataFrame
print(df)
```

The resulting output will be:
```
     Category
0  Category 1
1  Category 2
2  Category 1
3  Category 2
4  Category 1
```

In the example, the `.cat` accessor is used to perform operations on the 'Category' column. Firstly, the column is converted to a categorical data type using `.astype('category')`. Then, several categorical methods are applied using the `.cat` accessor. The categories are renamed using `.cat.categories`, a new category is added using `.cat.add_categories()`, and unused categories are removed using `.cat.remove_unused_categories()`.

The `.cat` accessor provides a range of methods and properties for categorical data manipulation, such as renaming categories, adding new categories, removing unused categories, reordering categories, and more. It allows for efficient handling and analysis of categorical variables in pandas DataFrames, enabling operations and computations specific to categorical data types.

```python
# .cat.set_categories(new_categories=[...])
dogs['coat'] = dogs['coat'].cat.set_categories(
   new_categories=['short','medium','long']
)

dogs['coat'].value_counts(dropna=False)
```
![Pasted image 20230716140536](/images/Pasted%20image%2020230716140536.png)
```python
# .cat.set_categories(new_categories=[...]) - continued
dogs['coat'] = dogs['coat'].cat.set_categories(
   new_categories=['short', 'medium', 'long']
   ordered=True
)
dogs['coat'].head(3)
```
![Pasted image 20230716141051](/images/Pasted%20image%2020230716141051.png)

##### Missing Categories
```python
dogs['likes_people'].value_counts(dropna=False)
```
![Pasted image 20230716141229](/images/Pasted%20image%2020230716141229.png)
```python
# add 2 new categories
dogs['likes_people'] = dogs['likes_people'].cat.add_categories(
   new_categories = ['did not check', 'could not tell']
)

# check 
dogs['likes_people'].cat.categories
```
![Pasted image 20230716141454](/images/Pasted%20image%2020230716141454.png)
```python
# setting up the new categories will be covered in a different lesson
dogs['likes_people'].value_counts(dropna=False)
```
![Pasted image 20230716141650](/images/Pasted%20image%2020230716141650.png)

##### Removing categories
```python
# .cat.remove_categories(removals=[...])
dogs['coat'] = dogs['coat'].astype('category')
dogs['coat'] = dogs['coat'].cat.remove_categories(removals=['wirehaired'])

dogs['coat'] = dogs['coat'].cat.categories
```
![Pasted image 20230716141955](/images/Pasted%20image%2020230716141955.png)

### Exercises

##### Exercise 1

```python
# Check frequency counts while also printing the NaN count
print(dogs['keep_in'].value_counts(dropna=False))
```
![Pasted image 20230716142408](/images/Pasted%20image%2020230716142408.png)
```python
# Switch to a categorical variable
dogs["keep_in"] = dogs["keep_in"].astype("category")

# Add new categories
new_categories = ["Unknown History", "Open Yard (Countryside)"]
dogs["keep_in"] = dogs["keep_in"].cat.add_categories(new_categories)

# Check frequency counts one more time
print(dogs["keep_in"].value_counts(dropna=False))
```
![Pasted image 20230716142827](/images/Pasted%20image%2020230716142827.png)

##### Exercise 2
```python
# Set "maybe" to be "no"
dogs.loc[dogs["likes_children"] == "maybe", "likes_children"] = "no"


# Print out categories
print(dogs["likes_children"].cat.categories)


# Print the frequency table
print(dogs["likes_children"].value_counts())
  

# Remove the `"maybe" category
dogs["likes_children"] = dogs["likes_children"].cat.remove_categories(["maybe"])
print(dogs["likes_children"].value_counts())

  
# Print the categories one more time
print(dogs['likes_children'].cat.categories)
```
![Pasted image 20230716143439](/images/Pasted%20image%2020230716143439.png)


### Updating categories

```python
dogs['breed'] = dogs['breed'].astype('category')
dogs['breed'].value_counts()
```
![Pasted image 20230716143643](/images/Pasted%20image%2020230716143643.png)

##### Renaming categories
```python
# .rename_categories
my_changes = {"Unknown Mix":"Unknown"}
dogs['breed'] = dogs['breed'].cat.rename_categories(my_changes)
dogs['breed'].value_counts()
```
![Pasted image 20230716144013](/images/Pasted%20image%2020230716144013.png)

##### Combining with Lambda functions
```python
# Title Case all current categories
dogs['sex'] = dogs['sex'].cat.rename_categories(lambda c: c.title())
dogs.cat.categories
```
![Pasted image 20230716144313](/images/Pasted%20image%2020230716144313.png)


##### Collapsing categories
```python
dogs['colors'] = dogs['colors'].astype('category')
print(dogs['colors'].cat.categories)
```
![Pasted image 20230716145235](/images/Pasted%20image%2020230716145235.png)
```python
# all colors to be collapsed
update_colors = {
				 "black and brown":"black",
				 "black and tan":"black",
				 "black and white":"black"
}

# create a new column "main_color" 
dogs["main_color"] = dogs["color"].replace(update_colors)

# ! this method changes the dtype
dogs["main_color"].dtype
```
![Pasted image 20230716145902](/images/Pasted%20image%2020230716145902.png)
```python
# reassign dtype
dogs["main_color"] = dogs["main_color"].astype("category")

# check
dog["main_color"].cat.categories
```
![Pasted image 20230716150006](/images/Pasted%20image%2020230716150006.png)

### Exercises

##### Exercise 1
```python
# Create the my_changes dictionary
my_changes = {"Maybe?":"Maybe"}

# Rename the categories listed in the my_changes dictionary
dogs["likes_children"] = dogs['likes_children'].cat.rename_categories(my_changes) 

# Use a lambda function to convert all categories to uppercase using upper()
dogs["likes_children"] =  dogs["likes_children"].cat.rename_categories(lambda c: c.upper())

# Print the list of categories
print(dogs['likes_children'].cat.categories)
```
![Pasted image 20230716150655](/images/Pasted%20image%2020230716150655.png)

##### Exercise 2
```python
# Create the update_coats dictionary
update_coats = {
    'wirehaired':'medium',
    'medium-long':'medium'
}
  
# Create a new column, coat_collapsed
dogs["coat_collapsed"] = dogs['coat'].replace(update_coats)

# Convert the column to categorical
dogs["coat_collapsed"] = dogs["coat_collapsed"].astype('category')

# Print the frequency table
print(dogs["coat_collapsed"].value_counts())
```
![Pasted image 20230716151105](/images/Pasted%20image%2020230716151105.png)

### Reordering categories

```python
# using .reorder_categories
dogs['coat'] = dogs['coat'].cat.reorder_categories(
   new_categories = ['short', 'medium', 'wirehaired', 'long'],
   ordered = True
)

# now using inplace (shorter)
dogs['coat'].cat.reorder_categories(
   new_categories = ['short', 'medium', 'wirehaired', 'long'],
   ordered = True,
   inplace = True
)

# visualizations, printouts etc. will use this order, e.g.:
dogs.groupby(by=['coat'])['age'].mean

```
![Pasted image 20230716151938](/images/Pasted%20image%2020230716151938.png)


## Exercises

##### Exercise 1
```python
# Print out the current categories of the size variable
print(dogs["size"].cat.categories)

# Reorder the categories, specifying the Series is ordinal, and overwriting the original series
dogs["size"].cat.reorder_categories(
  new_categories=["small", "medium", "large"],
  ordered=True,
  inplace=True
)
```

##### Exercise 2
```python
# Previous code
dogs["size"].cat.reorder_categories(
  new_categories=["small", "medium", "large"],
  ordered=True,
  inplace=True
)

# How many Male/Female dogs are available of each size?
print(dogs.groupby(by='size')['sex'].value_counts())
  
# Do larger dogs need more room to roam?
print(dogs.groupby(by='size')['keep_in'].value_counts())
```
![Pasted image 20230716153456](/images/Pasted%20image%2020230716153456.png)

***
## .str.strip()

The `.str.strip()` function in Python is a string method that is used to remove leading and trailing whitespace characters from a string. It returns a new string with the leading and trailing whitespace removed.

**Function syntax:**
```python
string.strip()
```

**Parameters:**
This function does not take any additional parameters.

**Example of use:**
```python
# Remove leading and trailing whitespace
string = "   Hello, World!   "
stripped_string = string.strip()
print(stripped_string)
```

The resulting output will be:
```
Hello, World!
```

In the example, the `.strip()` function is used to remove the leading and trailing whitespace from the string `string`. The resulting `stripped_string` contains the original string without the leading and trailing spaces.

The `.strip()` function is commonly used when dealing with user inputs or reading data from external sources, where leading or trailing whitespace may be present. It helps in cleaning and normalizing the string data by removing unnecessary whitespace characters. Note that `.strip()` only removes leading and trailing whitespace and does not affect any whitespace characters within the string.

```python
# Identifying the issue
dogs['get_along_cats'].value_counts()
```
![Pasted image 20230716154545](/images/Pasted%20image%2020230716154545.png)
```python
# Removing trailing white space (strip)
dogs['get_along_cats'] = dogs['get_along_cats'].str.strip()

dogs['get_along_cats'].value_counts()
```
![Pasted image 20230716154516](/images/Pasted%20image%2020230716154516.png)
***
## .str.title()

The `.str.title()` function in pandas is a string method that is used to convert the first character of each word in a string to uppercase and convert the remaining characters to lowercase. It returns a new string with the title-cased format.

**Function syntax:**
```python
Series.str.title()
```

**Parameters:**
This function does not take any additional parameters.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
data = {'Name': ['john doe', 'jane smith', 'alice brown']}
series = pd.Series(data['Name'])

# Convert the strings to title case
title_case_series = series.str.title()
print(title_case_series)
```

The resulting output will be:
```
0      John Doe
1    Jane Smith
2    Alice Brown
dtype: object
```

In the example, the `.str.title()` function is used to convert the strings in the `series` to title case. Each word in the string has its first character converted to uppercase, while the remaining characters are converted to lowercase.

The `.str.title()` function is useful for formatting strings, especially when dealing with names or titles that require capitalization. It helps in standardizing the capitalization of strings and making them visually appealing. It can be applied to individual strings within a Series or DataFrame column using the `.str` accessor. Note that the original Series is not modified, and a new Series with the title-cased strings is returned.


```python
# Capitalization: .title() / .upper() / .lower()
dogs['get_along_cats'] = dogs['get_along_cats'].str.title()

dogs['get_along_cats'].value_counts()
```
![Pasted image 20230716154837](/images/Pasted%20image%2020230716154837.png)

```python
# Fixing misspellings
replace_map = {'Noo':'No'}
dogs['get_along_cats'].replace(replace_map, inplace=True)

dogs['get_along_cats'].value_counts()
```
![Pasted image 20230716155058](/images/Pasted%20image%2020230716155058.png)
```python
# Always remember to make it categorical again
dogs['get_along_cats'] = dogs['get_along_cats'].astype('category')
```

***
## .str.contains()

The `.str.contains()` function in pandas is used to check whether each element in a string Series contains a specified pattern or substring. It returns a boolean Series indicating whether the pattern or substring is present in each element of the original Series.

**Function syntax:**
```python
Series.str.contains(pat, case=True, na=False, regex=True)
```

**Parameters:**
- `pat`: Specifies the pattern or substring to search for within each element of the Series.
- `case` (optional): Specifies whether the search should be case-sensitive. If `True` (default), the search is case-sensitive. If `False`, the search is case-insensitive.
- `na` (optional): Specifies whether missing values should be treated as `False`. If `True`, missing values are treated as `False`. If `False` (default), missing values are treated as `NA` (not available).
- `regex` (optional): Specifies whether the pattern is a regular expression. If `True` (default), the pattern is treated as a regular expression. If `False`, the pattern is treated as a literal string.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
data = {'Text': ['apple', 'banana', 'orange']}
series = pd.Series(data['Text'])

# Check for the presence of a pattern
contains_series = series.str.contains('na')
print(contains_series)
```

The resulting output will be:
```
0    False
1     True
2    False
dtype: bool
```

In the example, the `.str.contains()` function is used to check whether the substring `'na'` is present in each element of the `series`. The resulting `contains_series` is a boolean Series where `True` indicates that the pattern is present and `False` indicates its absence.

The `.str.contains()` function is useful for performing pattern matching or substring searches within string data. It allows you to identify elements that contain specific patterns or substrings, enabling filtering, conditional operations, or creating new columns based on the search results. Note that `.str.contains()` supports both literal strings and regular expressions, giving you flexibility in defining the patterns to search for.

```python
dogs['breed'].str.contains("Shepherd", regex=False)
```
![Pasted image 20230716155500](/images/Pasted%20image%2020230716155500.png)

***
## .loc\[]

The `.loc[]` function in pandas is used for label-based indexing and slicing of rows and columns in a DataFrame. It allows you to access specific rows and columns by their labels or a boolean condition based on labels.

**Function syntax:**
```python
DataFrame.loc[row_indexer, column_indexer]
```

**Parameters:**
- `row_indexer`: Specifies the labels or boolean condition to select specific rows from the DataFrame. It can be a single label, a list of labels, a slice object, or a boolean condition.
- `column_indexer`: Specifies the labels or boolean condition to select specific columns from the DataFrame. It can be a single label, a list of labels, a slice object, or a boolean condition.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {'Name': ['John', 'Jane', 'Alice'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)

# Access specific rows and columns using .loc[]
subset = df.loc[[0, 2], ['Name', 'City']]
print(subset)
```

The resulting output will be:
```
   Name      City
0  John  New York
2  Alice     Paris
```

In the example, the `.loc[]` function is used to access a subset of rows and columns from the DataFrame `df`. The `row_indexer` `[0, 2]` selects the first and third rows, and the `column_indexer` `['Name', 'City']` selects the 'Name' and 'City' columns. The resulting `subset` DataFrame contains the selected rows and columns.

The `.loc[]` function is powerful for label-based indexing and allows you to perform various operations, such as selecting specific rows and columns, modifying values, conditional filtering, or creating new columns based on label-based conditions. It provides a flexible and intuitive way to access and manipulate data in a DataFrame using labels.

```python
dogs.loc[dogs['get_along_cats'] == "Yes", "size"].value_counts(sort=False)
```
![Pasted image 20230716155733](/images/Pasted%20image%2020230716155733.png)

### Exercises

##### Exercise 1
```python
# Fix the misspelled word
replace_map = {"Malez": "male"}

# Update the sex column using the created map
dogs["sex"] = dogs["sex"].replace(replace_map)
  
# Strip away leading whitespace
dogs["sex"] = dogs["sex"].str.strip() 

# Make all responses lowercase
dogs["sex"] = dogs["sex"].str.lower()

# Convert to a categorical Series
dogs["sex"] = dogs["sex"].astype("category")


print(dogs["sex"].value_counts())
```
![Pasted image 20230716160135](/images/Pasted%20image%2020230716160135.png)

##### Exercise 2
```python
# Print the category of the coat for ID 23807
print(dogs.loc[dogs.index == 23807, 'coat'])
```
![Pasted image 20230716160802](/images/Pasted%20image%2020230716160802.png)
```python
# Find the count of male and female dogs who have a "long" coat
print(dogs.loc[dogs['coat'] == 'long', 'sex'].value_counts())
```
![Pasted image 20230716160736](/images/Pasted%20image%2020230716160736.png)
```python
# Print the mean age of dogs with a breed of "English Cocker Spaniel"
print(dogs.loc[dogs['breed'] == 'English Cocker Spaniel', 'age'].mean())
```
![Pasted image 20230716160902](/images/Pasted%20image%2020230716160902.png)
```python
# Count the number of dogs that have "English" in their breed name
print(dogs[dogs["breed"].str.contains("English", regex=False)].shape[0])
```
![Pasted image 20230716161136](/images/Pasted%20image%2020230716161136.png)

***
## sns.catplot()

The `sns.catplot()` function is a categorical plot function provided by the Seaborn library. It is used to create categorical plots that can display the relationship between one categorical variable and one or more numerical or categorical variables.

**Function syntax:**
```python
sns.catplot(x=None, y=None, hue=None, data=None, kind='strip', ...)
```

**Parameters:**
- `x`, `y`, `hue` (optional): Variables that specify the data for the x-axis, y-axis, and hue grouping, respectively. These parameters accept column names or arrays from the `data` parameter.
- `data`: Specifies the DataFrame or long-form data object that contains the data to be plotted.
- `kind` (optional): Specifies the type of categorical plot to be created. It can be 'strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', or 'count', with 'strip' being the default.
- Additional parameters: `sns.catplot()` supports many additional parameters such as `order`, `col`, `row`, `height`, `aspect`, `orient`, `dodge`, `palette`, and more, which allow you to further customize the categorical plot.

**Example of use:**
```python
import seaborn as sns

# Create a sample DataFrame
data = {'Category': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 15, 8, 12, 9],
        'Group': ['X', 'Y', 'X', 'Y', 'X']}
df = pd.DataFrame(data)

# Create a categorical plot using sns.catplot()
sns.catplot(x='Category', y='Value', hue='Group', data=df, kind='bar')
```

This example creates a bar plot using `sns.catplot()` from the Seaborn library. It specifies the 'Category' column for the x-axis, the 'Value' column for the y-axis, and the 'Group' column for color grouping. The data is provided from the DataFrame `df`, and the plot type is set to 'bar' using the `kind` parameter.

The `sns.catplot()` function allows you to create various types of categorical plots, such as strip plots, swarm plots, box plots, violin plots, bar plots, and more. It provides a convenient way to visualize categorical data and relationships between categorical and numerical variables.


### sns.catplot(kind='box')

```python
reviews['Score'].value_counts()
```
![Pasted image 20230716174943](/images/Pasted%20image%2020230716174943.png)
```python
# Setting font size and plot background
sns.set(font_scale=1.4)
sns.set_style("whitegrid")

sns.catplot(
	x='Pool',
	y='Score',
	data=reviews,
	kind='box'
)
plt.show()
```
![Pasted image 20230716175209](/images/Pasted%20image%2020230716175209.png)

### Exercises

##### Exercise 1
```python
# Set the font size to 1.25
sns.set(font_scale=1.25)

# Set the background to "darkgrid"
sns.set_style("darkgrid")

# Create a boxplot
sns.catplot(x= "Traveler type", y="Helpful votes", data=reviews, kind='box')

plt.show()
```
![Pasted image 20230716175444](/images/Pasted%20image%2020230716175444.png)

### sns.catplot(kind='bar')

```python
# Traditional bar chart
reviews["Traveler type"].value_counts().plot.bar()
```
![Pasted image 20230716175927](/images/Pasted%20image%2020230716175927.png)
```python
# Seaborn bar chart
sns.set(font_scale=1.3)
sns.set_style("darkgrid")
sns.catplot(x="Traveler type", y="Score", data=reviews, kind="bar")
plt.show()
```
![Pasted image 20230716180105](/images/Pasted%20image%2020230716180105.png)

##### Ordering the categories
```python
reviews["Traveler type"] = reviews["Traveler type"].astype("category")
reviews["Traveler type"].cat.categories
```
![Pasted image 20230716180259](/images/Pasted%20image%2020230716180259.png)
```python
sns.catplot(x="Traveler type", y="Score", data=reviews, kind="bar")
plt.show()
```
![Pasted image 20230716180358](/images/Pasted%20image%2020230716180358.png)

##### Hue parameter
```python
sns.set(font_scale=1.2)
sns.set_style("darkgrid")
sns.catplot(
	x="Traveler type", 
	y="Score", 
	data=reviews, 
	kind="bar", 
	hue="Tennis court"
)
plt.show()
```
![Pasted image 20230716180603](/images/Pasted%20image%2020230716180603.png)

### Exercises

##### Exercise 1
```python
# Print the frequency counts of "Period of stay"
print(reviews["Period of stay"].value_counts())
```
![Pasted image 20230716180848](/images/Pasted%20image%2020230716180848.png)
```python
sns.set(font_scale=1.4)
sns.set_style("whitegrid")

# Create a bar plot of "Helpful votes" by "Period of stay"
sns.catplot(x='Period of stay', y='Helpful votes', data=reviews, kind='bar')
plt.show()
```
![Pasted image 20230716180936](/images/Pasted%20image%2020230716180936.png)

##### Exercise 2
```python
# Create a bar chart
sns.set(font_scale=.9)
sns.set_style("whitegrid")
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar")
plt.show()
```
![Pasted image 20230716181116](/images/Pasted%20image%2020230716181116.png)
```python
# Print the frequency counts for "User continent"
print(reviews['User continent'].value_counts())
```
![Pasted image 20230716181202](/images/Pasted%20image%2020230716181202.png)
```python
# Set style
sns.set(font_scale=.9)
sns.set_style("whitegrid")
  
# Print the frequency counts for "User continent"
print(reviews["User continent"].value_counts())
  
# Convert "User continent" to a categorical variable
reviews["User continent"] = reviews['User continent'].astype('category')
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar")
plt.show()
```
![Pasted image 20230716181432](/images/Pasted%20image%2020230716181432.png)
```python
# Set style
sns.set(font_scale=.9)
sns.set_style("whitegrid")
  
# Print the frequency counts for "User continent"
print(reviews["User continent"].value_counts())

# Convert "User continent" to a categorical variable
reviews["User continent"] = reviews["User continent"].astype("category")

# Reorder "User continent" using continent_categories and rerun the graphic
continent_categories = list(reviews["User continent"].value_counts().index)
reviews["User continent"] = reviews["User continent"].cat.reorder_categories(new_categories=continent_categories)

sns.catplot(x="User continent", y="Score", data=reviews, kind="bar")
plt.show()
```
![Pasted image 20230716181753](/images/Pasted%20image%2020230716181753.png)

##### Exercise 3
```python
# Add a second category to split the data on: "Free internet"
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x="Casino", y="Score", data=reviews, kind="bar", hue="Free internet")
plt.show()
```
![Pasted image 20230716182349](/images/Pasted%20image%2020230716182349.png)
```python
# Switch the x and hue categories
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x="Free internet", y="Score", data=reviews, kind="bar", hue="Casino")
plt.show()
```
![Pasted image 20230716182447](/images/Pasted%20image%2020230716182447.png)
```python
# Update x to be "User continent"
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar", hue="Casino")
plt.show()
```
![Pasted image 20230716182631](/images/Pasted%20image%2020230716182631.png)
```python
# Lower the font size so that all text fits on the screen.
sns.set(font_scale=1.0)
sns.set_style("darkgrid")
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar", hue="Casino")
plt.show()
```
![Pasted image 20230716182720](/images/Pasted%20image%2020230716182720.png)

### sns.catplot(kind='point')

```python
sns.catplot(x='Pool', y='Score', data=reviews, kind='point')
plt.show()
```
![Pasted image 20230716182938](/images/Pasted%20image%2020230716182938.png)

##### Point plot with hue & dodge
```python
sns.catplot(
			x="Spa", 
			y="Score", 
			data=reviews, 
			kind="point", 
			hue="Tennis court", 
			dodge=True
)
plt.show()
```
![Pasted image 20230716183200](/images/Pasted%20image%2020230716183200.png)

##### Using the join parameter
```python
sns.catplot(
			x="Score",
			y="Review weekday",
			data=reviews,
			kind="point",
			join=False
)
plt.show()
```
![Pasted image 20230716183339](/images/Pasted%20image%2020230716183339.png)

### sns.catplot(kind='count')

```python
sns.catplot(
			x='Tennis court',
			data=reviews,
			kind='count',
			hue='Spa'
)
plt.show()
```
![Pasted image 20230716183517](/images/Pasted%20image%2020230716183517.png)

### Exercises

##### Exercise 1
```python
# Create a point plot with catplot using "Hotel stars" and "Nr. reviews"
sns.catplot(
  # Split the data across Hotel stars and summarize Nr. reviews
  x='Hotel stars',
  y='Nr. reviews',
  data=reviews,
  # Specify a point plot
  kind='point',
  hue="Pool",
  # Make sure the lines and points don't overlap
  dodge=True
)

plt.show()
```
![Pasted image 20230716184017](/images/Pasted%20image%2020230716184017.png)

##### Exercise 2
```python
sns.set(font_scale=1.4)
sns.set_style("darkgrid")

# Create a catplot that will count the frequency of "Score" across "Traveler type"
sns.catplot(
  x='Score',
  data=reviews,
  hue='Traveler type',
  kind='count'
)

plt.show()
```
![Pasted image 20230718131938](/images/Pasted%20image%2020230718131938.png)


### Additional .catplot options

##### Using the catplot() facetgrid

```python
sns.catplot(
			x="Traveler type",
			data=reviews,
			kind="count",
			col="User continent",
			col_wrap=3,
			palette=sns.color_palette("Set1")
)
```
![Pasted image 20230718132617](/images/Pasted%20image%2020230718132617.png)

```python
ax = sns.catplot(
			x="Traveler type",
			data=reviews,
			kind="count",
			col="User continent",
			col_wrap=3,
			palette=sns.color_palette("Set1")
)
ax.fig.suptitle("Hotel Score by Traveler Type & User Continent")
ax.set_axis_labels("Traveler Type", "Number of Reviews")
plt.subplots_adjust(top=.9)
plt.show()
```
![Pasted image 20230718133121](/images/Pasted%20image%2020230718133121.png)

### Exercises

##### Exercise 1
```python
# Create a catplot for each "Period of stay" broken down by "Review weekday"
ax = sns.catplot(
  # Make sure Review weekday is along the x-axis
  x='Review weekday',
  # Specify Period of stay as the column to create individual graphics for
  col='Period of stay',
  # Specify that a count plot should be created
  kind='count',
  # Wrap the plots after every 2nd graphic.
  col_wrap=2,
  data=reviews
)

plt.show()
```
![Pasted image 20230718133726](/images/Pasted%20image%2020230718133726.png)

##### Exercise 2
```python
# Adjust the color
ax = sns.catplot(
  x="Free internet", y="Score",
  hue="Traveler type", kind="bar",
  data=reviews,
  palette=sns.color_palette("Set2")
)

  

# Add a title
ax.fig.suptitle("Hotel Score by Traveler Type and Free Internet Access")

# Update the axis labels
ax.set_axis_labels("Free Internet", "Average Review Rating")

  

# Adjust the starting height of the graphic
plt.subplots_adjust(top=0.93)
plt.show()
```
![Pasted image 20230718133948](/images/Pasted%20image%2020230718133948.png)

##### Exercise 3
```python
# Print the frequency table of body_type and include NaN values
print(used_cars["body_type"].value_counts(dropna=False))

  

# Update NaN values
used_cars.loc[used_cars["body_type"].isna(), "body_type"] = "other"

  

# Convert body_type to title case
used_cars["body_type"] = used_cars["body_type"].str.title()

  

# Check the dtype
print(used_cars["body_type"].dtype)
```

##### Exercise 3
```python
# Print the frequency table of Sale Rating
print(used_cars["Sale Rating"].value_counts())
```
![Pasted image 20230718135904](/images/Pasted%20image%2020230718135904.png)
```python
# Find the average score
average_score = used_cars["Sale Rating"].astype("int").mean()

# Print the average
print(average_score)
```
![Pasted image 20230718140226](/images/Pasted%20image%2020230718140226.png)

***
## .cat.codes

The `.cat.codes` attribute in pandas is used to obtain the category codes of a categorical column in a DataFrame or Series. It returns a Series of integers representing the category codes corresponding to the categorical values.

**Attribute syntax:**
```python
Series.cat.codes
```

**Example of use:**
```python
import pandas as pd

# Create a sample Series with categorical data
data = pd.Series(['A', 'B', 'C', 'A', 'B'])
categories = ['A', 'B', 'C']
categorical_series = pd.Categorical(data, categories=categories)

# Get the category codes
category_codes = categorical_series.codes
print(category_codes)
```

The resulting output will be:
```
[0 1 2 0 1]
```

In the example, a Series `categorical_series` is created with categorical data using `pd.Categorical()`. The `category_codes` attribute is then used to obtain the category codes for each value in the categorical series. The resulting `category_codes` Series contains the integer codes corresponding to the categories ['A', 'B', 'C'], where 'A' is represented by 0, 'B' by 1, and 'C' by 2.

The `.cat.codes` attribute is helpful when you need to convert categorical data into numerical codes for various purposes, such as encoding categorical variables for machine learning algorithms or performing numerical computations on categorical data. It provides a convenient way to access the underlying integer codes of categorical values in a DataFrame or Series.


##### Label Encoding

```python
# Codes each category as an integer from 0 through n
# -1 = NA

# Example
used_cars['manufacturer_name'] = used_cars['manufacturer_name'].astype('category')

# Using .cat.codes we create a new column for the coded categorical variable
used_cars['manufacturer_code'] = used_cars['manufacturer_name'].cat.codes

# Let's see what happened
print(used_cars[['manufacturer_name', 'manufacturer_code']])
```
![Pasted image 20230718140807](/images/Pasted%20image%2020230718140807.png)

##### Creating a Codebook
```python
codes = used_cars['manufacturer_name'].cat.codes
categories = used_cars['manufacturer_name']

name_map = dict(zip(codes, categories))
print(name_map)
```
![Pasted image 20230718141030](/images/Pasted%20image%2020230718141030.png)

```python
# To revert previous values we can use .map
used_cars['manufacturer_code'].map(name_map)
```
![Pasted image 20230718141312](/images/Pasted%20image%2020230718141312.png)
***
## zip()

The `zip()` function in Python is used to combine or iterate over multiple iterables simultaneously. It takes multiple iterables as input and returns an iterator that produces tuples containing elements from each iterable.

**Function syntax:**
```python
zip(iterable1, iterable2, ...)
```

**Parameters:**
The `zip()` function takes one or more iterables as input. These can be sequences like lists, tuples, or strings, or any other iterable object.

**Example of use:**
```python
# Using zip() to combine two lists
numbers = [1, 2, 3]
letters = ['A', 'B', 'C']
combined = zip(numbers, letters)

# Iterate over the combined elements
for num, letter in combined:
    print(num, letter)
```

The resulting output will be:
```
1 A
2 B
3 C
```

In the example, `zip(numbers, letters)` combines the `numbers` and `letters` lists into an iterator of tuples, where each tuple contains the corresponding elements from both lists. The `for` loop then iterates over the combined elements, unpacking them into `num` and `letter`, and prints each pair.

The `zip()` function is commonly used when you need to iterate over multiple iterables in parallel or combine their corresponding elements. It is useful for tasks like pairwise comparisons, merging data from multiple sources, or creating dictionaries or data structures based on multiple sequences.

***
## np.where()

The `np.where()` function in NumPy is used to return elements from one of two arrays based on a specified condition. It acts as a vectorized version of the conditional expression in Python.

**Function syntax:**
```python
np.where(condition, x, y)
```

**Parameters:**
- `condition`: Specifies the condition to evaluate. It can be a boolean array or a logical expression.
- `x`: Specifies the array or value to select when the condition is True.
- `y`: Specifies the array or value to select when the condition is False.

**Example of use:**
```python
import numpy as np

# Create a sample NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Use np.where() to apply a condition and return elements from different arrays
result = np.where(arr < 3, arr, arr*10)
print(result)
```

The resulting output will be:
```
[ 1  2 30 40 50]
```

In the example, `np.where(arr < 3, arr, arr*10)` applies the condition `arr < 3` to the array `arr`. For elements where the condition is True, it selects the corresponding elements from `arr`. For elements where the condition is False, it selects the corresponding elements from `arr*10`. The resulting array `result` contains the selected elements based on the condition.

The `np.where()` function is useful for performing element-wise conditional operations on NumPy arrays. It allows you to select elements from different arrays or assign values based on specified conditions in a concise and efficient manner. It is commonly used in data manipulation, data cleaning, and numerical computations where conditional logic is required.

##### Boolean coding
```python
# Finding all body types that have van in them and create a new column which indicates if a vehicle is a van or not (1 or 0)

used_cars['van_code'] = np.where(
	 used_cars['body_type'].str.contains('van', regex=False), 1, 0
)

# Lets inspect
used_cars['van_code'].value_counts()
```
![Pasted image 20230718141716](/images/Pasted%20image%2020230718141716.png)

## Exercises

##### Exercise 1

```python
# Convert to categorical and print the frequency table
used_cars["color"] = used_cars["color"].astype("category")
print(used_cars["color"].value_counts())
```
![Pasted image 20230718142122](/images/Pasted%20image%2020230718142122.png)
```python
# Create a label encoding
used_cars["color_code"] = used_cars["color"].cat.codes

# Create codes and categories objects
codes = used_cars["color"].cat.codes
categories = used_cars["color"]
color_map = dict(zip(codes, categories))

# Print the map
print(color_map)
```
![Pasted image 20230718142303](/images/Pasted%20image%2020230718142303.png)

##### Exercise 2

```python
# Update the color column using the color_map
used_cars_updated["color"] = used_cars_updated['color'].map(color_map)
# Update the engine fuel column using the fuel_map
used_cars_updated["engine_fuel"] = used_cars_updated['engine_fuel'].map(fuel_map)
# Update the transmission column using the transmission_map
used_cars_updated["transmission"] = used_cars_updated['transmission'].map(transmission_map)

# Print the info statement
print(used_cars_updated.info())
```
![Pasted image 20230718142527](/images/Pasted%20image%2020230718142527.png)

##### Exercise 3

```python
# Print the manufacturer name frequency table
print(used_cars['manufacturer_name'].value_counts())
```
![Pasted image 20230718142640](/images/Pasted%20image%2020230718142640.png)
```python
# Create a Boolean column for the most common manufacturer name
used_cars["is_volkswagen"] = np.where(
  used_cars["manufacturer_name"].str.contains("Volkswagen", regex=False), 1, 0
)
  
# Check the final frequency table
print(used_cars["is_volkswagen"].value_counts())
```
![Pasted image 20230718142933](/images/Pasted%20image%2020230718142933.png)

### One-hot Encoding aka. Dummifying variables

## .get_dummies()

The `get_dummies()` function in pandas is used to convert categorical variables into dummy or indicator variables. It creates a new DataFrame with binary columns representing the presence or absence of each category from the original variable.

**Function syntax:**
```python
pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, drop_first=False)
```

**Parameters:**
- `data`: Specifies the DataFrame or Series containing the categorical variables to be converted.
- `prefix` (optional): Specifies the prefix to add to the column names of the dummy variables.
- `prefix_sep` (optional): Specifies the separator to use between the prefix and the original column name.
- `dummy_na` (optional): Specifies whether to include a column for missing values (NaN). If `True`, a column will be added to represent missing values. Default is `False`.
- `columns` (optional): Specifies the columns in the DataFrame to be converted. If `None`, all categorical columns are converted.
- `drop_first` (optional): Specifies whether to drop the first category for each variable to avoid multicollinearity. Default is `False`.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame with a categorical variable
data = {'Category': ['A', 'B', 'A', 'C']}
df = pd.DataFrame(data)

# Convert the categorical variable into dummy variables
dummy_df = pd.get_dummies(df, prefix='Category', prefix_sep='_', drop_first=True)
print(dummy_df)
```

The resulting output will be:
```
   Category_B  Category_C
0           0           0
1           1           0
2           0           0
3           0           1
```

In the example, `pd.get_dummies(df, prefix='Category', prefix_sep='_', drop_first=True)` converts the 'Category' column in the DataFrame `df` into dummy variables. The resulting `dummy_df` DataFrame contains binary columns for each category, with 'Category' used as the prefix and '_' as the prefix separator. The first category ('A') is dropped to avoid multicollinearity using the `drop_first` parameter.

The `get_dummies()` function is useful for converting categorical variables into a numerical format suitable for machine learning algorithms or statistical analysis. It allows you to encode categorical variables as binary columns, making them compatible with various modeling techniques. Additionally, it provides flexibility through optional parameters for customizing the column names, including missing values, and handling multicollinearity.

##### One-hot Encoding on a DataFrame
```python
# Creating our new dummified dataframe
used_cars_onehot = pd.get_dummies(used_cars[['odometer_value', 'color']])
used_cars_onehot.head()
```
![Pasted image 20230718143423](/images/Pasted%20image%2020230718143423.png)
```python
print(used_cars_onehot.shape)
```
![Pasted image 20230718143445](/images/Pasted%20image%2020230718143445.png)

##### Specifying columns to dummify
```python
# Dummifying only the "color" column and giving the new column names no prefix
used_cars_onehot = pd.get_dummies(used_cars, columns=['color'], prefix='')
used_cars_onehot.head()
```
![Pasted image 20230718143620](/images/Pasted%20image%2020230718143620.png)

### Exercises

##### Exercise 1
```python
# Create one-hot encoding for just two columns
used_cars_simple = pd.get_dummies(
  used_cars,
  # Specify the columns from the instructions
  columns=['manufacturer_name', 'transmission'],
  # Set the prefix
  prefix='dummy'
)

# Print the shape of the new dataset
print(used_cars_simple.shape)
```
![Pasted image 20230718143901](/images/Pasted%20image%2020230718143901.png)