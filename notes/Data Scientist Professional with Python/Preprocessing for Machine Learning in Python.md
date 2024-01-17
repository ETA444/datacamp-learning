
## `pd.read_json()`

In the pandas library, the `pd.read_json()` function is used to read and parse JSON (JavaScript Object Notation) data and create a DataFrame from it. JSON is a common data interchange format, and `pd.read_json()` allows you to import JSON data into a pandas DataFrame for further analysis and manipulation.

**Function Syntax:**
```python
pd.read_json(path_or_buf=None, orient=None, typ='frame', dtype=True, convert_axes=True, convert_dates=True, keep_default_dates=True, numpy=False, precise_float=False, date_unit=None, encoding=None, lines=False, chunksize=None, compression='infer')
```

**Parameters:**
- `path_or_buf`: The path to a JSON file or a JSON string to be read. It can be a file path, URL, or a JSON string.
- `orient` (optional): The format of the JSON data. Common values include `'split'`, `'records'`, `'index'`, `'columns'`, and `'values'`. The default is `'columns'`.
- `typ` (optional): The type of object to return. By default, it returns a DataFrame.
- `dtype` (optional): Data type to force. By default, it infers the data types.
- `convert_axes` (optional): Whether to convert the axes' data types. Default is `True`.
- `convert_dates` (optional): Whether to convert date-like columns. Default is `True`.
- `keep_default_dates` (optional): Whether to keep the default dates when parsing dates. Default is `True`.
- `numpy` (optional): Whether to use numpy.datetime64 for date parsing. Default is `False`.
- `precise_float` (optional): Whether to use higher precision (strtod) when parsing floats. Default is `False`.
- `date_unit` (optional): The time unit to use when parsing dates. Default is `None`.
- `encoding` (optional): The character encoding to use when reading the file. Default is `None`.
- `lines` (optional): Read the file as a JSON object per line. Default is `False`.
- `chunksize` (optional): Read the file in chunks of this size. Default is `None`.
- `compression` (optional): Specify the compression method if the file is compressed. Default is `'infer'`.

**Return Value:**
- A pandas DataFrame or other specified data structure containing the parsed JSON data.

**Example (Reading JSON from a File):**
```python
import pandas as pd

# Read JSON data from a file
df = pd.read_json('data.json')

# Display the DataFrame
print(df)
```

**Example (Reading JSON from a String):**
```python
import pandas as pd

# JSON data as a string
json_data = '{"Name": "Alice", "Age": 25, "City": "New York"}'

# Read JSON data from a string
df = pd.read_json(json_data)

# Display the DataFrame
print(df)
```

In the examples above, we demonstrate how to use `pd.read_json()` to read JSON data from both a file and a string, creating a pandas DataFrame from the JSON data.

---
## `.info()`

In pandas, the `.info()` method is used to obtain a concise summary of a DataFrame. It provides essential information about the DataFrame, including its data types, non-null values, memory usage, and more. This method is particularly useful for understanding the structure of your data and identifying potential issues.

**Method Syntax:**
```python
DataFrame.info(verbose=True, memory_usage=True, null_counts=True)
```

**Parameters:**
- `verbose` (optional): Controls the amount of information displayed. By default, it's set to `True`, which provides a detailed summary. Set to `False` for a more concise summary.
- `memory_usage` (optional): Determines whether memory usage information is displayed. Default is `True`.
- `null_counts` (optional): Indicates whether the count of non-null values is displayed. Default is `True`.

**Example:**
```python
import pandas as pd

# Create a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, None],
        'City': ['New York', 'Los Angeles', 'Chicago']}

df = pd.DataFrame(data)

# Display a concise summary of the DataFrame using .info()
df.info()
```

In this example, we create a sample DataFrame `df`, and then we use the `.info()` method to obtain a concise summary of its structure. The summary includes information about the data types of columns, the number of non-null values, and memory usage.

The `.info()` method is valuable for data exploration and quality assessment, helping you understand the characteristics of your DataFrame.

---
## `.describe()`

In pandas, the `.describe()` method is used to generate descriptive statistics for a DataFrame or a specific DataFrame column. It provides summary statistics such as count, mean, standard deviation, minimum, and maximum values for numeric columns, allowing you to quickly understand the distribution of data.

**Method Syntax (DataFrame):**
```python
DataFrame.describe(percentiles=None, include=None, exclude=None)
```

**Method Syntax (Series/Column):**
```python
Series.describe(percentiles=None)
```

**Parameters (DataFrame):**
- `percentiles` (optional): Specifies which percentiles to include in the summary statistics. By default, it includes the 25th, 50th (median), and 75th percentiles.

**Parameters (Series/Column):**
- `percentiles` (optional): Specifies which percentiles to include in the summary statistics for the column. By default, it includes the 25th, 50th (median), and 75th percentiles.

**Return Value:**
- For a DataFrame, it returns a new DataFrame with summary statistics for numeric columns.
- For a Series/Column, it returns a Series with summary statistics.

**Example (DataFrame):**
```python
import pandas as pd

# Create a sample DataFrame
data = {'Age': [25, 30, 35, 28, 22],
        'Salary': [50000, 60000, 75000, 52000, 48000]}

df = pd.DataFrame(data)

# Generate summary statistics for numeric columns using .describe()
summary = df.describe()

# Display the summary statistics
print(summary)
```

**Example (Series/Column):**
```python
import pandas as pd

# Create a sample Series
data = [25, 30, 35, 28, 22]

series = pd.Series(data)

# Generate summary statistics for the Series using .describe()
summary = series.describe()

# Display the summary statistics
print(summary)
```

In these examples, we use `.describe()` to generate summary statistics for a DataFrame and a Series. The summary includes count, mean, standard deviation, minimum, and maximum values, as well as quartiles by default.

The `.describe()` method is a quick and convenient way to gain insights into the distribution of data within a DataFrame or a specific column.

---
## `.drop()`

In pandas, the `.drop()` method is used to remove specified rows or columns from a DataFrame. This method allows you to eliminate unwanted data from your DataFrame, effectively reducing its size or altering its structure.

**Method Syntax (Drop Rows):**
```python
DataFrame.drop(labels, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
```

**Method Syntax (Drop Columns):**
```python
DataFrame.drop(labels, axis=1, index=None, columns=None, level=None, inplace=False, errors='raise')
```

**Parameters:**
- `labels`: The labels to drop, which can be a single label or a list of labels.
- `axis` (optional): Specifies whether to drop rows (`axis=0`, the default) or columns (`axis=1`).
- `index` (optional): An alternative to specifying rows to drop by label. Use `index` to specify rows by their index.
- `columns` (optional): An alternative to specifying columns to drop by label. Use `columns` to specify columns by their label.
- `level` (optional): For DataFrames with multi-level index, specify the level from which to drop labels.
- `inplace` (optional): If set to `True`, it modifies the DataFrame in place and returns `None`. If set to `False` (the default), it returns a new DataFrame with the specified rows or columns dropped.
- `errors` (optional): Determines how to handle errors if any of the specified labels are not found. Options include `'raise'` (default), `'ignore'`, and `'coerce'`.

**Example (Drop Rows by Label):**
```python
import pandas as pd

# Create a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}

df = pd.DataFrame(data)

# Drop rows with labels 0 and 2 using .drop()
df_dropped = df.drop([0, 2])

# Display the DataFrame with rows dropped
print(df_dropped)
```

**Example (Drop Columns by Label):**
```python
import pandas as pd

# Create a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}

df = pd.DataFrame(data)

# Drop the 'Name' column using .drop() with axis=1
df_dropped = df.drop('Name', axis=1)

# Display the DataFrame with the column dropped
print(df_dropped)
```

In these examples, we use `.drop()` to remove rows and columns from a DataFrame based on labels. The `axis` parameter specifies whether to operate along rows (`axis=0`) or columns (`axis=1`). You can specify labels by providing a single label or a list of labels.

The `.drop()` method is a powerful tool for data manipulation in pandas, allowing you to tailor your DataFrame to your specific needs.

---
## `.dropna()`

In pandas, the `.dropna()` method is used to remove rows or columns from a DataFrame that contain missing or NaN (Not a Number) values. This method allows you to filter out incomplete data, which can be useful when working with datasets that may have missing information.

**Method Syntax (Drop Rows with Missing Values):**
```python
DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
```

**Method Syntax (Drop Columns with Missing Values):**
```python
DataFrame.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)
```

**Parameters:**
- `axis` (optional): Specifies whether to drop rows (`axis=0`, the default) or columns (`axis=1`) with missing values.
- `how` (optional): Determines the criteria for dropping. Options include `'any'` (default) to drop if any NaN is present, and `'all'` to drop only if all values are NaN.
- `thresh` (optional): Specifies the minimum number of non-NaN values required to keep a row or column.
- `subset` (optional): Allows you to specify a subset of columns or rows for checking missing values.
- `inplace` (optional): If set to `True`, it modifies the DataFrame in place and returns `None`. If set to `False` (the default), it returns a new DataFrame with the missing values removed.

**Example (Drop Rows with Missing Values):**
```python
import pandas as pd

# Create a sample DataFrame with missing values
data = {'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 30, None]}

df = pd.DataFrame(data)

# Drop rows with missing values using .dropna()
df_cleaned = df.dropna()

# Display the cleaned DataFrame
print(df_cleaned)
```

**Example (Drop Columns with Missing Values):**
```python
import pandas as pd

# Create a sample DataFrame with missing values
data = {'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 30, None]}

df = pd.DataFrame(data)

# Drop columns with missing values using .dropna() with axis=1
df_cleaned = df.dropna(axis=1)

# Display the cleaned DataFrame
print(df_cleaned)
```

In these examples, we use `.dropna()` to remove rows with missing values (NaN) from a DataFrame and to remove columns with missing values. The `axis` parameter specifies whether to operate along rows (`axis=0`) or columns (`axis=1`). The `how` parameter determines the criteria for dropping, with `'any'` being the default to drop rows or columns containing any NaN values.

The `.dropna()` method is useful for data preprocessing when you want to eliminate missing or incomplete data from your DataFrame.

---
## `.astype()`

In pandas, the `.astype()` method is used to change the data type of a pandas Series or DataFrame. This method allows you to explicitly convert the data type of elements in a Series or the entire DataFrame, which can be helpful when you need to ensure data consistency or perform specific operations with the desired data type.

**Method Syntax (Series):**
```python
Series.astype(dtype, copy=True, errors='raise')
```

**Method Syntax (DataFrame):**
```python
DataFrame.astype(dtype, copy=True, errors='raise')
```

**Parameters:**
- `dtype`: The data type to which you want to convert the Series or DataFrame.
- `copy` (optional): If set to `True` (the default), it returns a new Series or DataFrame with the specified data type. If set to `False`, it modifies the original Series or DataFrame in place and returns `None`.
- `errors` (optional): Determines how to handle errors if any conversion is not possible due to data incompatibility. Options include `'raise'` (default), `'coerce'`, and `'ignore'`.

**Example (Convert Series Data Type):**
```python
import pandas as pd

# Create a sample Series with integers
data = pd.Series([1, 2, 3, 4, 5])

# Convert the data type of the Series to float using .astype()
float_series = data.astype(float)

# Display the new Series with the updated data type
print(float_series)
```

**Example (Convert DataFrame Column Data Type):**
```python
import pandas as pd

# Create a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': ['25', '30', '35']}

df = pd.DataFrame(data)

# Convert the 'Age' column data type to integer using .astype()
df['Age'] = df['Age'].astype(int)

# Display the updated DataFrame with the column data type converted
print(df)
```

In these examples, we use `.astype()` to change the data type of a Series and a DataFrame column, respectively. The `dtype` parameter specifies the target data type to which the elements are converted.

The `.astype()` method is useful when you need to ensure that your data has the correct data types for performing calculations or analysis.

Here's a bulleted list of common data types that you can use as arguments for `.astype()`:

- `int`: Integer data type.
- `float`: Floating-point data type.
- `str` or `object`: String data type.
- `bool`: Boolean data type (True or False).
- `datetime64`: Datetime data type for date and time.
- `category`: Categorical data type for a finite set of unique values.
- `timedelta64`: Timedelta data type for time intervals.
- `complex`: Complex number data type.
- `int32` or `int64`: Specific integer data types with 32 or 64 bits.
- `float32` or `float64`: Specific floating-point data types with 32 or 64 bits.

---
## `sklearn.model_selection.train_test_split()`

In scikit-learn (sklearn), the `train_test_split()` function is used for splitting a dataset into training and testing sets. This function is commonly employed in machine learning workflows to create separate datasets for model training and evaluation.

**Function Syntax:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True, stratify=None)
```

**Parameters:**
- `X`: The feature data or independent variables.
- `y`: The target data or dependent variable.
- `test_size` (optional): The proportion of the dataset to include in the test split. It can be a float (e.g., 0.2 for 20%) or an integer specifying the number of samples in the test split.
- `random_state` (optional): Controls the randomness of the split. Setting a specific value ensures reproducibility. If `None`, the split will be random.
- `shuffle` (optional): Determines whether the data is shuffled before splitting. Default is `True`.
- `stratify` (optional): Ensures that the class distribution in the target variable is maintained in both the training and testing sets. This is especially useful for imbalanced datasets.

**Return Value:**
- `X_train`: The training set of feature data.
- `X_test`: The testing set of feature data.
- `y_train`: The training set of target data.
- `y_test`: The testing set of target data.

**Example:**
```python
from sklearn.model_selection import train_test_split
import numpy as np

# Create sample data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the resulting sets
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)
print("y_train:")
print(y_train)
print("y_test:")
print(y_test)
```

In this example, we use `train_test_split()` to split a dataset represented by `X` and `y` into training and testing sets. The `test_size` parameter specifies that 30% of the data should be allocated for testing, and the `random_state` parameter ensures that the split is reproducible.

This function is a fundamental tool for machine learning tasks, allowing you to assess your model's performance on an independent dataset separate from the one used for training.

---
## `.value_counts()`

In pandas, the `.value_counts()` method is used to count the unique values in a Series and return the counts as a new Series or DataFrame. This method is particularly useful for exploring the frequency distribution of values within a categorical column.

**Method Syntax (Series):**
```python
Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
```

**Parameters:**
- `normalize` (optional): If set to `True`, it returns the relative frequencies (proportions) instead of counts. Default is `False`.
- `sort` (optional): Determines whether to sort the result by counts. Default is `True`.
- `ascending` (optional): Specifies the sorting order. Set to `False` to sort in descending order (default is `True` for ascending).
- `bins` (optional): Used for binning numeric data into discrete intervals and counting the values in each bin.
- `dropna` (optional): Determines whether to exclude missing values (NaN) from the count. Default is `True`.

**Return Value:**
- A Series containing the unique values as indices and their corresponding counts (or relative frequencies) as values.

**Example (Count Unique Values in a Categorical Column):**
```python
import pandas as pd

# Create a sample Series with categorical data
data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'A'])

# Count the unique values in the Series using .value_counts()
value_counts = data.value_counts()

# Display the frequency distribution
print(value_counts)
```

**Example (Count Unique Values with Relative Frequencies):**
```python
import pandas as pd

# Create a sample Series with categorical data
data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'A'])

# Count the unique values with relative frequencies using .value_counts(normalize=True)
value_counts_normalized = data.value_counts(normalize=True)

# Display the relative frequencies
print(value_counts_normalized)
```

In these examples, we use `.value_counts()` to count the unique values in a Series containing categorical data. The method provides counts (or relative frequencies) of each unique value, allowing you to understand the distribution of values within the column.

`.value_counts()` is a valuable tool for exploratory data analysis and can help you gain insights into the characteristics of your data.

---
