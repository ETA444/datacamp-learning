
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
## 