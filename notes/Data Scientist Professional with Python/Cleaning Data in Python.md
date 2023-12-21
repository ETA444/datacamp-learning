
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