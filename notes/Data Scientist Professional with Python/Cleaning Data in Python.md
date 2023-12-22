
***
![[Pasted image 20230720164903.png]]

## str.strip()

The `.strip()` method is used to remove leading and trailing whitespace characters from a string in Python.

**Method syntax:**
```python
str.strip([chars])
```

**Parameters:**
- `chars` (optional): Specifies the characters to be removed from the string. If not provided, it removes leading and trailing whitespace characters.

**Example of use:**
```python
text = "   Hello, World!   "

# Remove leading and trailing whitespace
stripped_text = text.strip()

print(stripped_text)  # Output: "Hello, World!"
```

In this example, `text.strip()` removes the leading and trailing whitespace characters from the string `text`. The resulting `stripped_text` variable contains the modified string without any leading or trailing whitespace.

The `.strip()` method is commonly used to clean up user input, remove unnecessary whitespace, or normalize strings before further processing. It is particularly useful when dealing with strings obtained from user inputs or reading text from files, ensuring consistent and properly formatted string values.

***
## assert

The `assert` statement is used in Python for debugging and testing purposes to check if a condition is true. If the condition is false, an `AssertionError` is raised.

**Syntax:**
```python
assert condition, message
```

**Parameters:**
- `condition`: Specifies the expression or condition to be checked. It should evaluate to either `True` or `False`.
- `message` (optional): Specifies an optional error message to be displayed when the condition is false. It can be a string or any object that can be converted to a string.

**Example of use:**
```python
x = 10

# Check if x is greater than 0
assert x > 0, "x must be positive"

print("After the assertion")
```

In this example, the `assert x > 0, "x must be positive"` statement checks if the value of `x` is greater than 0. If the condition is `False`, an `AssertionError` is raised, and the specified error message "x must be positive" is displayed. If the condition is `True`, the program continues execution without any interruption.

The `assert` statement is often used during development and testing to validate assumptions or check the correctness of code logic. It helps identify and report potential errors early in the development process. Note that assertions are typically used for debugging and testing purposes and can be disabled in a production environment for performance reasons.

```python
# this will pass
assert 1 + 1 == 2

# this won't pass
assert 1 + 1 == 3
```
![[Pasted image 20230720171258.png]]

***
## dt.date.today()

The `dt.date.today()` function is used to retrieve the current local date as a `date` object from the `datetime` module in Python.

**Function syntax:**
```python
dt.date.today()
```

**Example of use:**
```python
import datetime as dt

# Get the current local date
today = dt.date.today()

# Display the current date
print(today)
```

In this example, `dt.date.today()` retrieves the current local date and assigns it to the `today` variable. The resulting `today` variable represents the current date.

By using `dt.date.today()`, you can obtain the current local date in a standardized format. This can be useful for various applications, such as timestamping, date calculations, or data filtering based on the current date.

```python
import datetime as dt
today_date = dt.date.today()

# logic check 
user_signups[user_signups['subscription_date'] > dt.date.today()]
```
![[Pasted image 20230720172557.png]]

***
## .drop()

The `.drop()` method in pandas is used to remove specified rows or columns from a DataFrame.

**Method syntax to drop rows:**
```python
DataFrame.drop(labels, axis=0, inplace=False)
```

**Method syntax to drop columns:**
```python
DataFrame.drop(labels, axis=1, inplace=False)
```

**Parameters:**
- `labels`: Specifies the index labels or column names to be dropped. It can be a single label or a list of labels.
- `axis`: Specifies the axis along which to drop the labels. Use `axis=0` to drop rows (default) and `axis=1` to drop columns.
- `inplace`: Specifies whether to modify the DataFrame in-place (if `True`) or return a new DataFrame with the specified rows or columns dropped (if `False`, default).

**Example of dropping rows:**
```python
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Drop rows by index label
df.drop([0, 2], inplace=True)

# Display the updated DataFrame
print(df)
```

In this example, `df.drop([0, 2], inplace=True)` drops the rows with index labels 0 and 2 from the DataFrame `df`. The `inplace=True` parameter modifies the DataFrame in-place, removing the specified rows. The resulting DataFrame contains the remaining rows.

**Example of dropping columns:**
```python
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Drop columns by column name
df.drop(['Age'], axis=1, inplace=True)

# Display the updated DataFrame
print(df)
```

In this example, `df.drop(['Age'], axis=1, inplace=True)` drops the column with the name 'Age' from the DataFrame `df`. The `axis=1` parameter indicates that the drop operation should be performed along the columns. The `inplace=True` parameter modifies the DataFrame in-place, removing the specified column. The resulting DataFrame contains the remaining columns.

The `.drop()` method is a versatile way to remove rows or columns from a DataFrame in pandas. It allows for the selective removal of specific indices or column names, providing flexibility in data manipulation and cleaning tasks.

```python
import pandas as pd
# Output Movies with rating > 5
movies[movies['avg_rating'] > 5]
```
![[Pasted image 20230720172955.png]]
```python
# Drop values using filterin
gmovies = movies[movies['avg_rating'] <= 5]
# Drop values using .drop()
movies.drop(movies[movies['avg_rating'] > 5].index, inplace = True)
# Assert results
assert movies['avg_rating'].max() <= 5
```


***
## .duplicated()

The `.duplicated()` method in pandas is used to identify duplicate rows in a DataFrame.

**Method syntax:**
```python
DataFrame.duplicated(subset=None, keep='first')
```

**Parameters:**
- `subset` (optional): Specifies the column(s) or label(s) to consider for identifying duplicates. By default, it checks all columns for duplication.
- `keep` (optional): Specifies which occurrence(s) of a duplicate to mark as `True`. It can take the following values:
  - `'first'` (default): Mark all duplicates as `True`, except for the first occurrence.
  - `'last'`: Mark all duplicates as `True`, except for the last occurrence.
  - `False`: Mark all duplicates as `True`.

**Example:**
```python
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'Alice'],
        'Age': [25, 30, 35, 25]}
df = pd.DataFrame(data)

# Identify duplicate rows
duplicates = df.duplicated()

# Display the DataFrame with a new column indicating duplicates
df['IsDuplicate'] = duplicates

# Display the updated DataFrame
print(df)
```

In this example, `df.duplicated()` identifies duplicate rows in the DataFrame `df`. The resulting `duplicates` variable contains a boolean Series with `True` for duplicate rows and `False` for non-duplicates. We then assign this Series to a new column named `'IsDuplicate'` in the DataFrame using `df['IsDuplicate'] = duplicates`. The updated DataFrame contains an additional column indicating whether each row is a duplicate or not.

The `.duplicated()` method is useful for detecting and handling duplicate rows in a DataFrame. You can use it to filter out duplicates, perform further analysis, or clean up data by removing redundant rows.

```python
# Column names to check for duplication
column_names = ['first_name','last_name','address']
duplicates = height_weight.duplicated(subset = column_names, keep = False)
```

---
## .info()

The `.info()` method in pandas is used to display concise information about a DataFrame, including the data types, non-null values, and memory usage. It provides a quick summary of the DataFrame's structure and contents.

**Method syntax:**
```python
DataFrame.info(verbose=True, memory_usage=True, null_counts=True)
```

**Parameters:**
- `verbose` (optional): Controls the amount of information displayed. If `True`, all column names and the count of non-null values are shown. If `False`, a concise summary is displayed. By default, `verbose=None` displays the summary based on the DataFrame's size.
- `memory_usage` (optional): Specifies whether to include memory usage information in the summary. By default, it's set to `True`, and memory usage is included if the DataFrame has numeric columns.
- `null_counts` (optional): Specifies whether to include the count of null values in the summary. By default, it's set to `True`, and the count of null values is included if the DataFrame has null values.

**Example of use:**
```python
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, None, 35]}
df = pd.DataFrame(data)

# Display information about the DataFrame
df.info()
```

In this example, `df.info()` displays information about the DataFrame `df`. The output includes data type information, the count of non-null values, memory usage, and the presence of null values in each column.

The `.info()` method is a useful tool for quickly assessing the structure and data quality of a DataFrame. It helps you identify missing values, understand the data types, and get an overview of the DataFrame's memory usage.

---
## .drop()

The `.drop()` method in pandas is used to remove specified rows or columns from a DataFrame.

**Method syntax to drop rows:**
```python
DataFrame.drop(labels, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
```

**Method syntax to drop columns:**
```python
DataFrame.drop(labels, axis=1, index=None, columns=None, level=None, inplace=False, errors='raise')
```

**Parameters:**
- `labels`: Specifies the index labels or column names to be dropped. It can be a single label or a list of labels.
- `axis` (optional): Specifies the axis along which to drop the labels. Use `axis=0` to drop rows (default) and `axis=1` to drop columns.
- `index` (optional): An alternative to specifying `axis=0`, used to specify the index labels to drop (only applicable if `axis=0`).
- `columns` (optional): An alternative to specifying `axis=1`, used to specify the column names to drop (only applicable if `axis=1`).
- `level` (optional): For DataFrames with multi-level indexes, specifies the level from which to drop labels.
- `inplace` (optional): Specifies whether to modify the DataFrame in-place (if `True`) or return a new DataFrame with the specified rows or columns dropped (if `False`, default).
- `errors` (optional): Specifies how to handle errors if the specified labels are not found. It can take the following values:
  - `'raise'` (default): Raise an error if any of the specified labels are not found.
  - `'ignore'`: Ignore errors and do not raise an error if labels are not found.

**Examples of use:**
```python
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Drop rows by index label
df.drop([0, 2], inplace=True)

# Drop columns by column name
df.drop(['Age'], axis=1, inplace=True)
```

In these examples:
- `df.drop([0, 2], inplace=True)` drops the rows with index labels 0 and 2 from the DataFrame `df`.
- `df.drop(['Age'], axis=1, inplace=True)` drops the column with the name 'Age' from the DataFrame `df`.

The `.drop()` method is a versatile way to remove rows or columns from a DataFrame in pandas. It allows for the selective removal of specific indices or column names, providing flexibility in data manipulation and cleaning tasks.

---
## pd.to_datetime()

The `pd.to_datetime()` function in pandas is used to convert a series of date-like objects (e.g., strings, numbers) into a pandas DateTime object. This is particularly useful when you have date or time data in your DataFrame, and you want to ensure it's treated as datetime data for various operations and analyses.

**Function syntax:**
```python
pd.to_datetime(arg, errors='raise', format=None, infer_datetime_format=False, exact=True, unit=None, utc=None)
```

**Parameters:**
- `arg`: This can be a single datetime-like object or an iterable (e.g., list, Series, array) of datetime-like objects to be converted.
- `errors` (optional): Specifies how to handle parsing errors. It can take the following values:
  - `'raise'` (default): Raise an error if any parsing error occurs.
  - `'coerce'`: Set any parsing errors to NaT (Not a Timestamp) and continue.
  - `'ignore'`: Ignore parsing errors and return the original input.
- `format` (optional): A string specifying the expected format of the input data, using format codes (e.g., '%Y-%m-%d' for 'YYYY-MM-DD'). If not provided, pandas will attempt to infer the format.
- `infer_datetime_format` (optional): If set to `True`, pandas will attempt to infer the datetime format from the input data.
- `exact` (optional): If `True`, require an exact format match when parsing. If `False`, allow for inexact parsing.
- `unit` (optional): Specifies the unit of the input data, e.g., 's' for seconds or 'ms' for milliseconds.
- `utc` (optional): If `True`, convert the datetime to UTC (Coordinated Universal Time).

**Examples:**
```python
import pandas as pd

# Convert a string to a datetime object
date_str = '2023-01-15'
date = pd.to_datetime(date_str)

# Convert a list of strings to datetime objects
date_list = ['2023-01-15', '2023-02-20', '2023-03-25']
date_series = pd.to_datetime(date_list)

# Convert Unix timestamps (seconds since epoch) to datetime
timestamps = [1610668800, 1613827200, 1616635200]
date_from_timestamps = pd.to_datetime(timestamps, unit='s')
```

In these examples:
- `pd.to_datetime(date_str)` converts the string '2023-01-15' into a pandas DateTime object.
- `pd.to_datetime(date_list)` converts a list of date strings into a pandas DateTime Series.
- `pd.to_datetime(timestamps, unit='s')` converts a list of Unix timestamps (given in seconds since the epoch) into pandas DateTime objects.

The `pd.to_datetime()` function is a versatile tool for handling datetime data in pandas. It can handle a wide range of input formats and provides options for handling errors, specifying formats, and more.

---
## .dt

In pandas, the `.dt` accessor is used to access various datetime-related attributes and methods for a Series containing datetime values. You can use it to perform operations on datetime data efficiently. Here are some common attributes and methods that you can use with `.dt`:

- `.dt.year`: Extracts the year component of datetime values.
- `.dt.month`: Extracts the month component (1-12) of datetime values.
- `.dt.day`: Extracts the day component (1-31) of datetime values.
- `.dt.hour`: Extracts the hour component (0-23) of datetime values.
- `.dt.minute`: Extracts the minute component (0-59) of datetime values.
- `.dt.second`: Extracts the second component (0-59) of datetime values.
- `.dt.microsecond`: Extracts the microsecond component of datetime values.
- `.dt.weekday`: Returns the day of the week as an integer (0=Monday, 6=Sunday).
- `.dt.weekday_name` (or `.dt.day_name()`): Returns the name of the day of the week.
- `.dt.quarter`: Extracts the quarter (1-4) from datetime values.
- `.dt.is_leap_year`: Returns a boolean indicating if the year is a leap year.
- `.dt.strftime()`: Formats datetime values as strings using strftime format codes.
- `.dt.date`: Extracts the date part (year, month, and day) from datetime values.
- `.dt.time`: Extracts the time part (hour, minute, second, and microsecond) from datetime values.
- `.dt.timedelta()`: Performs arithmetic operations with datetime values.

Here's an example of how to use the `.dt` accessor with some of these attributes:

```python
import pandas as pd

# Create a DataFrame with a datetime column
data = {'Datetime': ['2023-01-15 08:00:00', '2023-02-20 12:30:00', '2023-03-25 18:45:00']}
df = pd.DataFrame(data)

# Convert the 'Datetime' column to datetime objects
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Extract various datetime components using .dt
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Weekday'] = df['Datetime'].dt.weekday
df['Quarter'] = df['Datetime'].dt.quarter
df['IsLeapYear'] = df['Datetime'].dt.is_leap_year

# Display the updated DataFrame
print(df)
```

In this example, we first convert a column of datetime strings to datetime objects using `pd.to_datetime()`. Then, we use various `.dt` attributes to extract and create new columns with different datetime components.

These datetime-related operations with `.dt` are quite handy when you need to work with datetime data in pandas and perform operations based on date and time components.

---
## .drop_duplicates()

The `.drop_duplicates()` method in pandas is used to remove duplicate rows from a DataFrame based on the values in one or more columns. It allows you to keep only the first occurrence of each unique row or remove all duplicates, depending on the parameters you provide.

**Method syntax:**
```python
DataFrame.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
```

**Parameters:**
- `subset` (optional): Specifies the column(s) or label(s) to consider for identifying duplicates. It can be a single column name or a list of column names. By default, it checks all columns for duplication.
- `keep` (optional): Specifies which occurrence(s) of a duplicate to keep. It can take the following values:
  - `'first'` (default): Keep the first occurrence and remove the subsequent duplicates.
  - `'last'`: Keep the last occurrence and remove the previous duplicates.
  - `False`: Remove all duplicates.
- `inplace` (optional): Specifies whether to modify the DataFrame in-place (if `True`) or return a new DataFrame with duplicates removed (if `False`, default).
- `ignore_index` (optional): If set to `True`, the resulting DataFrame will have a new index. If `False` (default), the index will be retained.

**Example:**
```python
import pandas as pd

# Create a DataFrame with duplicate rows
data = {'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'Age': [25, 30, 25, 35, 30]}
df = pd.DataFrame(data)

# Remove duplicate rows based on 'Name' column, keeping the first occurrence
df_no_duplicates = df.drop_duplicates(subset='Name')

# Remove duplicate rows based on both 'Name' and 'Age' columns, keeping the last occurrence
df_no_duplicates_last = df.drop_duplicates(subset=['Name', 'Age'], keep='last')
```

In this example:
- `df.drop_duplicates(subset='Name')` removes duplicate rows based on the 'Name' column, keeping the first occurrence and creating a new DataFrame with duplicates removed.
- `df.drop_duplicates(subset=['Name', 'Age'], keep='last')` removes duplicate rows based on both the 'Name' and 'Age' columns, keeping the last occurrence of each unique combination.

The `.drop_duplicates()` method is a valuable tool for cleaning and preparing data by removing redundant rows from a DataFrame. Depending on your data analysis needs, you can choose to keep the first, last, or remove all occurrences of duplicate rows.

---
## .set()

In Python, a `set` is an unordered collection of unique elements. Sets are defined by enclosing a comma-separated sequence of elements in curly braces `{}` or by using the `set()` constructor.

Here are some common operations you can perform with sets:

**1. Creating a Set:**
```python
my_set = {1, 2, 3}
```

**2. Adding Elements:**
```python
my_set.add(4)  # Adds the element 4 to the set
```

**3. Removing Elements:**
```python
my_set.remove(2)  # Removes the element 2 from the set
```

**4. Checking for Membership:**
```python
if 3 in my_set:
    print("3 is in the set")
```

**5. Set Operations (Union, Intersection, Difference):**
```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

union_set = set1.union(set2)         # Union of two sets
intersection_set = set1.intersection(set2)  # Intersection of two sets
difference_set = set1.difference(set2)    # Set difference
```

**6. Iterating Over a Set:**
```python
for item in my_set:
    print(item)
```

**7. Length of a Set:**
```python
length = len(my_set)  # Returns the number of elements in the set
```

**8. Creating an Empty Set:**
```python
empty_set = set()  # Creates an empty set
```

Sets are commonly used to store unique values or perform mathematical set operations like union, intersection, and difference. They are mutable, which means you can add or remove elements after creating them. However, sets themselves are not hashable and cannot be elements of another set.

---
## .difference()

The `.difference()` method in Python is used to find the set difference between two sets. It returns a new set containing elements that are in the first set but not in the second set. In other words, it calculates the elements that are unique to the first set.

**Method syntax:**
```python
set1.difference(set2)
```

- `set1`: The set from which you want to find the difference.
- `set2`: The set you want to compare with `set1` to find the difference.

**Example:**
```python
set1 = {1, 2, 3, 4, 5}
set2 = {3, 4, 5, 6, 7}

difference_set = set1.difference(set2)
```

In this example, `difference_set` will contain elements {1, 2}, which are in `set1` but not in `set2`.

You can also use the `-` operator to find the difference between sets:

```python
difference_set = set1 - set2
```

Both `.difference()` and the `-` operator perform the same set difference operation. Sets are useful for comparing and manipulating collections of unique elements.

---
## .isin()

The `.isin()` method in pandas is used to filter data frames. It is often used to filter rows or select a subset of rows from a DataFrame where a specified column's values match any of the values in a given list, series, or another iterable.

**Method syntax:**
```python
DataFrame[DataFrame['column_name'].isin(values)]
```

- `DataFrame`: The DataFrame you want to filter.
- `'column_name'`: The name of the column you want to filter based on.
- `values`: A list, series, or iterable containing the values you want to use for filtering.

**Example:**
```python
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 25]}
df = pd.DataFrame(data)

# Filter rows where 'Name' is in a specified list
selected_names = ['Bob', 'David']
filtered_df = df[df['Name'].isin(selected_names)]
```

In this example, `filtered_df` will contain the rows with names 'Bob' and 'David' because we used the `.isin()` method to filter based on the 'Name' column.

The `.isin()` method is a powerful tool for conditional filtering in pandas, allowing you to select rows where a specific column's values match a predefined list or set of values.

---
## pd.qcut()

The `pd.qcut()` function in pandas is used to perform quantile-based discretization of continuous data. It divides a series of continuous data into intervals or bins in such a way that each bin contains roughly the same number of data points. This can be useful for converting continuous data into categorical data for analysis or visualization.

**Function syntax:**
```python
pd.qcut(x, q, labels=False, retbins=False, precision=3, duplicates='raise')
```

**Parameters:**
- `x`: The input data to be discretized, typically a Series or array.
- `q`: Either an integer specifying the number of quantiles to create or an array of quantiles (values between 0 and 1) that define the bin edges.
- `labels` (optional): If `True`, it assigns labels to the discrete bins. If `False` (default), it returns integer bin labels.
- `retbins` (optional): If `True`, it returns the bin edges along with the discretized data.
- `precision` (optional): The number of decimal places to round the quantile values to.
- `duplicates` (optional): How to handle duplicates in the quantile values. It can take the following values:
  - `'raise'` (default): Raises an error if there are duplicate quantiles.
  - `'drop'`: Drops duplicates and uses the remaining values.

**Return values:**
- If `labels` is `True`, it returns a Series with labels for each data point.
- If `labels` is `False`, it returns a Series with integer bin labels.

**Examples:**
```python
import pandas as pd

# Create a Series of continuous data
data = [10, 15, 20, 25, 30, 35, 40, 45, 50]

# Perform quantile-based discretization into 3 equal bins
bins = pd.qcut(data, q=3, labels=False)

# Perform quantile-based discretization into custom quantiles
custom_quantiles = [0, 0.25, 0.5, 0.75, 1.0]
bins_custom = pd.qcut(data, q=custom_quantiles, labels=False)

# Get bin edges and labels
bins_with_labels = pd.qcut(data, q=3, labels=True, retbins=True)
```

In these examples:
- `pd.qcut(data, q=3, labels=False)` discretizes the `data` into 3 equal-width bins and returns integer bin labels.
- `pd.qcut(data, q=custom_quantiles, labels=False)` discretizes the `data` into custom quantiles specified by `custom_quantiles`.
- `pd.qcut(data, q=3, labels=True, retbins=True)` returns both bin labels and the bin edges.

`pd.qcut()` is useful when you want to convert continuous data into discrete bins with equal frequencies or custom quantiles for analysis or visualization.

---
## pd.cut()

The `pd.cut()` function in pandas is used for binning or discretizing continuous data into discrete intervals or bins. This can be useful when you want to convert a continuous variable into a categorical variable for analysis or visualization.

**Function syntax:**
```python
pd.cut(x, bins, labels=None, right=True, include_lowest=False, duplicates='raise')
```

**Parameters:**
- `x`: The input data to be discretized, typically a Series or array.
- `bins`: Specifies the bin edges. It can take different forms:
  - An integer, indicating the number of equal-width bins to create.
  - A list or array specifying the bin edges (e.g., `[0, 10, 20, 30]`).
  - A single number specifying the bin width, and the bins are generated between the minimum and maximum values of `x`.
- `labels` (optional): An array or list that assigns labels to each bin. It should have the same length as the number of bins.
- `right` (optional): Indicates whether the bins should be closed on the right (default) or left. If `right=True`, the intervals are right-closed, meaning the right bin edge is included in the interval.
- `include_lowest` (optional): If `True`, the first bin includes the minimum value of `x`.
- `duplicates` (optional): How to handle duplicate bin edges. It can take the following values:
  - `'raise'` (default): Raises an error if there are duplicate bin edges.
  - `'drop'`: Drops duplicates and uses the remaining values.

**Return value:**
- A Categorical object that represents the bin labels for each element in the input data.

**Examples:**
```python
import pandas as pd

# Create a Series of continuous data
data = [5, 12, 18, 25, 32, 40]

# Bin the data into equal-width bins
equal_width_bins = pd.cut(data, bins=3)

# Bin the data into custom bins with labels
custom_bins = pd.cut(data, bins=[0, 10, 20, 30, 50], labels=['Low', 'Mid', 'High', 'Very High'])

# Include the minimum value in the first bin
include_lowest = pd.cut(data, bins=3, include_lowest=True)

# Specify left-closed bins and handle duplicate bin edges
left_closed_bins = pd.cut(data, bins=[0, 10, 20, 30, 40], right=False, duplicates='drop')
```

In these examples:
- `pd.cut(data, bins=3)` divides the `data` into three equal-width bins.
- `pd.cut(data, bins=[0, 10, 20, 30, 50], labels=['Low', 'Mid', 'High', 'Very High'])` creates custom bins and assigns labels to each bin.
- `pd.cut(data, bins=3, include_lowest=True)` includes the minimum value of `data` in the first bin.
- `pd.cut(data, bins=[0, 10, 20, 30, 40], right=False, duplicates='drop')` creates left-closed bins and handles duplicate bin edges by dropping duplicates.

`pd.cut()` is a useful function for transforming continuous data into categorical data, making it easier to analyze and visualize.

---
## .replace()

The `.replace()` method in pandas is used to replace values in a DataFrame or Series with other values. It's a versatile function that allows you to perform various replacement operations based on specified rules.

**Method syntax for Series:**
```python
Series.replace(to_replace, value, inplace=False, limit=None, regex=False, method='pad')
```

**Method syntax for DataFrame:**
```python
DataFrame.replace(to_replace, value, inplace=False, limit=None, regex=False, method='pad')
```

**Parameters:**
- `to_replace`: Specifies the value(s) you want to replace. It can be:
  - a single value (e.g., `10`) to replace occurrences of that value.
  - a list, dictionary, or mapping that defines replacement rules.
- `value`: Specifies the value(s) to use for replacement.
- `inplace` (optional): If `True`, the DataFrame or Series is modified in place, and `None` is returned. If `False` (default), a new DataFrame or Series with replacements is returned.
- `limit` (optional): The maximum number of replacements to make. If `None`, replace all occurrences (default).
- `regex` (optional): If `True`, treat the `to_replace` as a regular expression. If `False` (default), treat it as a literal value.
- `method` (optional): Specifies how to handle replacements when using a DataFrame. It can take the following values:
  - `'pad'` (default): Replace using the next valid observation forward.
  - `'ffill'`: Same as 'pad'.
  - `'bfill'`: Replace using the next valid observation backward.
  - `'backfill'`: Same as 'bfill'.

**Examples:**
```python
import pandas as pd

# Replace a single value with another
data = {'A': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data)
df['A'].replace(2, 10, inplace=True)

# Replace multiple values using a dictionary
data = {'B': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)
df['B'].replace({20: 25, 30: 35}, inplace=True)

# Replace values using regular expressions
data = {'Text': ['apple', 'banana', 'pear', 'cherry']}
df = pd.DataFrame(data)
df['Text'].replace(to_replace=r'p.*', value='fruit', regex=True, inplace=True)
```

In these examples:
- `.replace(2, 10, inplace=True)` replaces the value `2` with `10` in the 'A' column.
- `.replace({20: 25, 30: 35}, inplace=True)` replaces multiple values in the 'B' column using a dictionary.
- `.replace(to_replace=r'p.*', value='fruit', regex=True, inplace=True)` replaces values in the 'Text' column using a regular expression to match words starting with 'p'.

The `.replace()` method is useful for data cleaning and transformation tasks where you need to replace specific values in your DataFrame or Series with other values or apply complex replacement rules.

---
## .any()

In pandas, the `.any()` method is used to determine whether any elements in a DataFrame or Series evaluate to `True`. It's often used to check if there are any non-zero or non-empty elements in a DataFrame or Series.

**Method syntax for Series:**
```python
Series.any(axis=0, bool_only=None, skipna=True, level=None, **kwargs)
```

**Method syntax for DataFrame:**
```python
DataFrame.any(axis=0, bool_only=None, skipna=True, level=None, **kwargs)
```

**Parameters:**
- `axis` (optional): Specifies the axis along which the operation is performed. By default (`axis=0`), it checks if any elements in each column evaluate to `True`. If `axis=1`, it checks if any elements in each row evaluate to `True`.
- `bool_only` (optional): If `True`, it only considers boolean data types for the check.
- `skipna` (optional): If `True` (default), it skips `NaN` values when performing the check.
- `level` (optional): For DataFrames with multi-level index, specifies the level to perform the operation on.

**Return value:**
- For Series, it returns a boolean value (`True` if any elements evaluate to `True`, otherwise `False`).
- For DataFrames, it returns a Series containing boolean values for each column (or row if `axis=1`).

**Examples:**
```python
import pandas as pd

# Create a Series
s = pd.Series([False, False, True, False])

# Check if any element is True in the Series
result = s.any()
# Result: True

# Create a DataFrame
data = {'A': [True, False, False], 'B': [False, False, False]}

df = pd.DataFrame(data)

# Check if any element is True in each column of the DataFrame
result = df.any()
# Result: A     True
#         B    False
#         dtype: bool

# Check if any element is True in each row of the DataFrame
result = df.any(axis=1)
# Result: 0     True
#         1    False
#         2    False
#         dtype: bool
```

In these examples:
- `s.any()` checks if any element in the Series `s` is `True`, and it returns `True` because there is at least one `True` element.
- `df.any()` checks if any element in each column of the DataFrame `df` is `True`. It returns a Series of boolean values indicating whether any `True` values are present in each column.
- `df.any(axis=1)` checks if any element in each row of the DataFrame `df` is `True`. It returns a Series of boolean values indicating whether any `True` values are present in each row.

The `.any()` method is helpful for quickly assessing whether any conditions are met in your data, especially when dealing with boolean data or data that can be evaluated as boolean.

---
## .dt.strftime()

In pandas, the `.strftime()` method is used to format dates and times in a DataFrame or Series into a string format based on specified format codes. It's a powerful tool for customizing how date and time information is displayed.

**Method syntax:**
```python
Series.dt.strftime(format)
```

**Parameters:**
- `format`: A string containing format codes that define how the date and time should be displayed. Format codes start with a percentage symbol `%` and are replaced with the corresponding date or time components when formatting.

**Return value:**
- A Series of strings with the date and time values formatted according to the specified format.

**Examples:**
```python
import pandas as pd

# Create a DataFrame with a date column
data = {'Date': ['2023-07-01', '2023-07-15']}
df = pd.DataFrame(data)

# Convert the 'Date' column to a datetime dtype
df['Date'] = pd.to_datetime(df['Date'])

# Format the date column
df['Formatted Date'] = df['Date'].dt.strftime('%Y-%m-%d')
```

In this example:
- We first create a DataFrame with a 'Date' column containing date strings.
- We convert the 'Date' column to a datetime data type using `pd.to_datetime()`.
- Then, we use `dt.strftime('%Y-%m-%d')` to format the date values in the 'Date' column into the 'YYYY-MM-DD' format and store the formatted values in a new column called 'Formatted Date'.

The resulting DataFrame will look like this:

```
        Date Formatted Date
0 2023-07-01      2023-07-01
1 2023-07-15      2023-07-15
```

You can customize the format string to include various date and time components such as year, month, day, hour, minute, and more using format codes. The `strftime()` method is especially useful for displaying date and time information in the desired format for reporting and visualization purposes.

---
## .fillna()
In pandas, the `.fillna()` method is used to fill missing or NaN (Not a Number) values in a DataFrame or Series with specified values or using specific filling strategies.

**Method syntax for Series:**
```python
Series.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
```

**Method syntax for DataFrame:**
```python
DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
```

**Parameters:**
- `value` (optional): The value to use for filling missing values. It can be a scalar, a dictionary, a Series, or a DataFrame.
- `method` (optional): A method for filling missing values. Options include 'ffill' (forward fill), 'bfill' (backward fill), or None (default, which fills with the specified `value`).
- `axis` (optional): Specifies whether to fill along rows (axis=0) or columns (axis=1). Default is None, which fills values in both directions.
- `inplace` (optional): If True, the DataFrame or Series is modified in place, and `None` is returned. If False (default), a new object with missing values filled is returned.
- `limit` (optional): Maximum number of consecutive NaN values to be filled in a forward or backward fill operation.
- `downcast` (optional): If specified, downcast the resulting data type (e.g., from float64 to int64) if possible.
- `**kwargs` (optional): Additional keyword arguments that may be specific to certain fill methods.

**Return value:**
- If `inplace=True`, it returns `None`.
- If `inplace=False`, it returns a new DataFrame or Series with missing values filled.

**Examples:**
```python
import pandas as pd

# Create a DataFrame with missing values
data = {'A': [1, 2, None, 4, 5], 'B': [None, 2, 3, None, 5]}
df = pd.DataFrame(data)

# Fill missing values with a specified value
df_filled = df.fillna(value=0)

# Fill missing values using forward fill (ffill)
df_ffill = df.fillna(method='ffill')

# Fill missing values using backward fill (bfill)
df_bfill = df.fillna(method='bfill')

# Fill missing values in a specific column with a custom value
df['A'].fillna(value=10, inplace=True)
```

In these examples:
- `df.fillna(value=0)` fills missing values in the DataFrame `df` with the specified value (0) and creates a new DataFrame `df_filled`.
- `df.fillna(method='ffill')` uses forward fill to fill missing values in the DataFrame `df`, creating a new DataFrame `df_ffill`.
- `df.fillna(method='bfill')` uses backward fill to fill missing values in the DataFrame `df`, creating a new DataFrame `df_bfill`.
- `df['A'].fillna(value=10, inplace=True)` fills missing values in the 'A' column of the DataFrame `df` with the value 10 in place, modifying the original DataFrame.

---
## fuzz.WRatio()

In the thefuzz library, the `fuzz.WRatio()` method is used to calculate a similarity ratio between two strings. It is based on the Wagner-Fisher algorithm, which measures the similarity between two strings by comparing the sequences of characters and their positions.

**Method syntax:**
```python
fuzz.WRatio(str1, str2)
```

**Parameters:**
- `str1`: The first string to be compared.
- `str2`: The second string to be compared.

**Return value:**
- Returns an integer similarity ratio between 0 and 100, where higher values indicate greater similarity.

**Example:**
```python
from thefuzz import fuzz

# Compare the similarity between two strings
similarity_ratio = fuzz.WRatio("apple", "apples")

# Result: 94
```

In this example, `fuzz.WRatio()` is used to compare the similarity between the strings "apple" and "apples." The method returns a similarity ratio of 94, indicating a high degree of similarity between the two strings.

The `fuzz.WRatio()` method is helpful for tasks like string matching, deduplication, and record linkage where you want to quantify how similar two strings are in a fuzzy manner.

---
## process.extract()

In the thefuzz library, the `process.extract()` function is used to find the best matches for a given query string within a list of strings or choices. It uses fuzzy string matching to identify the most similar strings from the list.

**Function syntax:**
```python
process.extract(query, choices, scorer=fuzz.WRatio, limit=5, processor=None)
```

**Parameters:**
- `query`: The query string for which you want to find matches within the list of choices.
- `choices`: A list of strings among which you want to find matches for the query.
- `scorer` (optional): The scoring function used to compare the similarity between the query and each choice. It defaults to `fuzz.WRatio`, but you can specify other scoring functions provided by thefuzz.
- `limit` (optional): The maximum number of matches to return. It defaults to 5.
- `processor` (optional): A function to preprocess the strings before comparing them.

**Return value:**
- Returns a list of tuples, each containing a choice string and its similarity score to the query string, sorted in descending order of similarity.

**Example:**
```python
from thefuzz import process

# List of choices
choices = ["apple", "banana", "cherry", "date", "grape"]

# Find the best matches for the query "apples"
matches = process.extract("apples", choices)

# Result:
# [('apple', 90), ('cherry', 45), ('date', 45), ('banana', 36), ('grape', 9)]
```

In this example, `process.extract()` is used to find the best matches for the query "apples" within the list of choices. It returns a list of tuples, where each tuple contains a choice and its similarity score to the query.

The `process.extract()` function is useful in applications where you need to find approximate matches for a given query string, such as record linkage, fuzzy search, or spell correction.

---
## `recordlinkage.Index()`

In the "recordlinkage" library, the `Index()` class is used to create and manage indexing structures for efficient record linkage tasks. It is a fundamental component of the record linkage process, enabling the comparison of records between different datasets.

**Class instantiation:**
```python
indexer = recordlinkage.Index()
```

**Methods:**
- `.full()` method: Create a full index, where all pairs of records are compared.
- `.block()` method: Create a block index, specifying specific blocks or groups of records to compare.
- `.sortedneighbourhood()` method: Create a sorted neighborhood index for efficient comparison of nearby records.
- `.pairs()` method: Retrieve the pairs of record indices for comparison based on the index type.

**Example:**
```python
import recordlinkage

# Create an indexer
indexer = recordlinkage.Index()

# Create a full index for comparing all pairs of records
indexer.full()

# Generate pairs of record indices for comparison
pairs = indexer.pairs(df1, df2)

# Result: A MultiIndex containing pairs of record indices
```

In this example:
- We instantiate the `Index()` class using `recordlinkage.Index()`.
- We create a full index using the `.full()` method, which compares all pairs of records.
- We then generate pairs of record indices for comparison between two DataFrames `df1` and `df2` using the `.pairs()` method.

The `recordlinkage.Index()` class is a crucial step in the record linkage process, allowing you to efficiently identify pairs of records that need to be compared for matching or deduplication tasks.

---
## `recordlinkage.Index().block()`

In the "recordlinkage" library, the `.block()` method is used in conjunction with the `Index()` class to create a block index, which specifies specific blocks or groups of records to be compared during the record linkage process. This can be useful when you have prior knowledge or domain-specific information about which records are likely to match.

**Method syntax:**
```python
Index.block(left_on=None, right_on=None, lfilter=None, rfilter=None)
```

**Parameters:**
- `left_on` (optional): The column name or list of column names from the left DataFrame (usually the first dataset) used for blocking.
- `right_on` (optional): The column name or list of column names from the right DataFrame (usually the second dataset) used for blocking.
- `lfilter` (optional): A custom filtering function for the left DataFrame, used to create the block index.
- `rfilter` (optional): A custom filtering function for the right DataFrame, used to create the block index.

**Return value:**
- Returns the current `Index` object with the block index applied.

**Example:**
```python
import recordlinkage

# Create an indexer
indexer = recordlinkage.Index()

# Create a block index based on a specific column
indexer.block(left_on='zipcode')

# Generate pairs of record indices for comparison
pairs = indexer.pairs(df1, df2)

# Result: A MultiIndex containing pairs of record indices, filtered by the 'zipcode' column
```

In this example:
- We instantiate the `Index()` class using `recordlinkage.Index()`.
- We create a block index using the `.block()` method, specifying the 'zipcode' column as the blocking key. This means that only records with the same 'zipcode' value will be compared during the record linkage process.
- We then generate pairs of record indices for comparison between two DataFrames `df1` and `df2` using the `.pairs()` method. The pairs are filtered based on the block index, so only records with matching 'zipcode' values are considered.

The `.block()` method is a powerful tool in record linkage when you have prior knowledge or criteria for selecting which records to compare, potentially improving the efficiency and accuracy of the linkage process.

---
## `recordlinkage.Compare()`

The `Compare()` class in the "recordlinkage" library is used to create comparison objects for specifying the rules and methods to compare records from different datasets or sources.

**Class instantiation:**
```python
comparison = recordlinkage.Compare()
```

**Methods and attributes:**
- `.exact(column1, column2)` method: Specify that two columns should be compared for exact equality.
- `.string(column1, column2, method='jarowinkler', threshold=0.85)` method: Specify that two string columns should be compared using a fuzzy string matching method like Jaro-Winkler.
- `.numeric(column1, column2, method='step', offset=0.1)` method: Specify that two numeric columns should be compared using a numeric comparison method.
- `.geo(left_coordinates, right_coordinates)` method: Specify that two sets of geographical coordinates should be compared.
- `.date_of_birth(column1, column2)` method: Specify that two date of birth columns should be compared.
- `.compute(df, method='simple')` method: Compute the comparison results for a DataFrame using the specified method.

**Example:**
```python
import recordlinkage

# Create a comparison object
comparison = recordlinkage.Compare()

# Specify comparison rules
comparison.exact('first_name', 'first_name')
comparison.string('last_name', 'last_name', method='jarowinkler', threshold=0.85)
comparison.numeric('income', 'income', method='step', offset=0.1)

# Compute comparison results for a DataFrame
result = comparison.compute(df)
```

In this example:
- We create a `Compare()` object using `recordlinkage.Compare()`.
- We specify comparison rules using methods like `.exact()`, `.string()`, and `.numeric()`. These rules define how specific columns in a DataFrame should be compared.
- We use the `.compute()` method to apply the comparison rules to a DataFrame `df` and compute the comparison results.

The `Compare()` class is a fundamental component of the "recordlinkage" library, allowing you to define the comparison logic tailored to your record linkage task. It helps you determine which fields should be compared and how similar they need to be to consider two records as potential matches.

---
## `recordlinkage.Compare().compute()`

In the "recordlinkage" library, the `compute()` method is used in conjunction with the `Compare()` class to perform the actual comparison of records based on the specified comparison rules. It computes the comparison results for a DataFrame, indicating the degree of similarity between records.

**Method syntax:**
```python
comparison.compute(df, method='simple')
```

**Parameters:**
- `df`: The DataFrame containing the records you want to compare.
- `method` (optional): The method used for computing comparison results. Options include 'simple' (default), 'step', 'symmetric', and 'dedupe'. The method defines how the results are calculated.

**Return value:**
- Returns a DataFrame containing the comparison results. Each row corresponds to a pair of records, and each column represents a comparison rule, with values indicating the similarity or match score.

**Example:**
```python
import recordlinkage

# Create a comparison object
comparison = recordlinkage.Compare()

# Specify comparison rules
comparison.exact('first_name', 'first_name')
comparison.string('last_name', 'last_name', method='jarowinkler', threshold=0.85)
comparison.numeric('income', 'income', method='step', offset=0.1)

# Compute comparison results for a DataFrame
result = comparison.compute(df)
```

In this example:
- We create a `Compare()` object and specify comparison rules using methods like `.exact()`, `.string()`, and `.numeric()`.
- We use the `.compute()` method to apply these comparison rules to a DataFrame `df`.
- The resulting DataFrame `result` contains the comparison results, with each row representing a pair of records and each column representing a comparison rule. The values in the DataFrame indicate the degree of similarity between the corresponding records for each rule.

The `compute()` method is a crucial step in record linkage, as it quantifies how similar records are based on the defined comparison rules. These similarity scores can then be used to identify potential matches or duplicates in your data. The choice of the `method` parameter affects how the similarity scores are computed, allowing you to fine-tune the comparison process based on your specific requirements.

---
## `.get_level_values(level)`

In pandas, the `.get_level_values()` method is used to extract the values at a specific level of a multi-level index DataFrame.

**Method syntax:**
```python
DataFrame.get_level_values(level)
```

**Parameters:**
- `level`: An integer or label representing the level of the multi-level index from which you want to retrieve values.

**Return value:**
- Returns a pandas Series containing the values from the specified level of the index.

**Example:**
```python
import pandas as pd

# Create a multi-level index DataFrame
data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
}

index = pd.MultiIndex.from_tuples([('x', 'i'), ('x', 'ii'), ('y', 'i'), ('y', 'ii')], names=['first', 'second'])

df = pd.DataFrame(data, index=index)

# Get values from the 'first' level of the index
first_level_values = df.index.get_level_values('first')

# Result: Index(['x', 'x', 'y', 'y'], dtype='object', name='first')
```

In this example:
- We create a multi-level index DataFrame `df` with two levels: 'first' and 'second'.
- We use `.index.get_level_values('first')` to retrieve the values from the 'first' level of the index. This returns an Index object containing the values 'x', 'x', 'y', and 'y'.

The `.get_level_values()` method is helpful when working with multi-level index DataFrames, allowing you to access and manipulate data at specific levels of the index. You can use these extracted values for various data analysis and filtering tasks.