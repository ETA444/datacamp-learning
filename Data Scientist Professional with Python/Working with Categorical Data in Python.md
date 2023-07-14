

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
![[Pasted image 20230714224138.png]]

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
![[Pasted image 20230714224258.png]]
```python
# Why do we use categorical: memory
adult["Marital Status"] = adult['Marital Status'].astype("category")
adult["Marital Status"].nbytes
```
![[Pasted image 20230714224355.png]]


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
![[Pasted image 20230714212920.png]]

##### Exercise 2
```python
# Create a categorical Series and specify the categories (let pandas know the order matters!)
medals = pd.Categorical(medals_won, categories=['Bronze', 'Silver', 'Gold'], ordered=True)
print(medals)
```
![[Pasted image 20230714213648.png]]

##### Exercise 3
```python
# Check the dtypes
print(adult.dtypes)
```
![[Pasted image 20230714213831.png]]

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
## 