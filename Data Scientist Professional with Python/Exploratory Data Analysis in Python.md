

* * *
## .select_dtypes()

The `.select_dtypes()` function in pandas is used to select columns from a DataFrame based on their data types.

**Function signature:**
```python
DataFrame.select_dtypes(include=None, exclude=None)
```

**Parameters:**
- `include` (optional): Specifies the data types to include in the selection. It can be a single data type or a list of data types.
- `exclude` (optional): Specifies the data types to exclude from the selection. It can be a single data type or a list of data types.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['John', 'Alice', 'Bob'],
    'Age': [25, 32, 41],
    'City': ['New York', 'Paris', 'London'],
    'Salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)

# Select columns with data type 'object' (string)
object_cols = df.select_dtypes(include='object')

# Select columns with data type 'int64' or 'float64'
numeric_cols = df.select_dtypes(include=['int64', 'float64'])

# Select columns excluding data type 'object'
non_object_cols = df.select_dtypes(exclude='object')
```

In the example, we have a DataFrame `df` with columns of different data types. The `.select_dtypes()` function is used to perform the following selections:
- `object_cols` will contain the columns 'Name' and 'City' since they have the data type 'object' (string).
- `numeric_cols` will contain the columns 'Age' and 'Salary' since they have the data types 'int64' and 'float64'.
- `non_object_cols` will contain the columns 'Age' and 'Salary' since they are not of data type 'object'.

This function is helpful when you want to focus on specific data types in your analysis or when you need to separate columns based on their data types for further processing.

***Datacamp Example***
![Pasted image 20230710164636](/images/Pasted%20image%2020230710164636.png)

### All dtypes
![Pasted image 20230710164146](/images/Pasted%20image%2020230710164146.png)

---
## .value_counts()

The `.value_counts()` function in pandas is used to obtain the frequency count of unique values in a pandas Series.

**Function signature:**
```python
Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)
```

**Parameters:**
- `normalize` (optional): If `normalize=True`, the resulting counts will be expressed as proportions (relative frequencies) instead of raw counts. The default is `False`.
- `sort` (optional): If `sort=True`, the counts will be sorted in descending order. If `sort=False`, the counts will appear in the order of occurrence. The default is `True`.
- `ascending` (optional): If `ascending=True`, the counts will be sorted in ascending order. If `ascending=False`, the counts will be sorted in descending order. This parameter is deprecated and will be removed in future versions of pandas. Use `sort` instead.
- `bins` (optional): This parameter is applicable only to numeric data. It specifies the number of equal-width bins to use in the histogram calculation.
- `dropna` (optional): If `dropna=True`, missing values (NaN) will be excluded from the counts. If `dropna=False`, missing values will be included in the counts. The default is `True`.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
s = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

# Count the frequency of unique values
value_counts = s.value_counts()
```

The resulting `value_counts` will be:
```
4    4
3    3
2    2
1    1
dtype: int64
```

In the example, the `.value_counts()` function is used to count the frequency of unique values in the Series `s`. The resulting Series `value_counts` displays each unique value in the Series along with its count.

This function is particularly useful for obtaining insights into the distribution of values in a Series and identifying the most common or least common values present.

***Datacamp Example***
![Pasted image 20230710164544](/images/Pasted%20image%2020230710164544.png)


* * *
## .nunique()

The `.nunique()` function in pandas is used to count the number of unique values in a pandas Series.

**Function signature:**
```python
Series.nunique(dropna=True)
```

**Parameters:**
- `dropna` (optional): If `dropna=True`, missing values (NaN) will be excluded from the count. If `dropna=False`, missing values will be included in the count. The default is `True`.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
s = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

# Count the number of unique values
unique_count = s.nunique()
```

The resulting `unique_count` will be:
```
4
```

In the example, the `.nunique()` function is used to count the number of unique values in the Series `s`. The resulting count is a single value representing the number of distinct values in the Series.

This function is useful for getting a quick understanding of the diversity or distinctiveness of values in a Series. It can be helpful for tasks such as identifying categorical variables, assessing cardinality, or measuring the level of uniqueness in a dataset.

* * *
## .str.contains()

The `.str.contains()` function in pandas is used to check whether each element of a Series contains a specified substring or pattern.

**Function signature:**
```python
Series.str.contains(pat, case=True, flags=0, na=None, regex=True)
```

**Parameters:**
- `pat`: The substring or pattern to search for within each element of the Series.
- `case` (optional): If `case=True`, the search is case-sensitive. If `case=False`, it is case-insensitive. The default is `True`.
- `flags` (optional): Flags controlling the behavior of the regex matching. Common flags include `re.IGNORECASE`, `re.MULTILINE`, `re.DOTALL`, etc.
- `na` (optional): The value to use for missing or non-string elements. If set to `None`, missing or non-string elements will raise an exception. The default is `None`.
- `regex` (optional): If `True`, treats the `pat` parameter as a regular expression. If `False`, treats it as a literal string. The default is `True`.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
s = pd.Series(['apple', 'banana', 'cherry'])

# Check if each element contains 'a'
contains_a = s.str.contains('a')
```

The resulting `contains_a` will be:
```
0    True
1    True
2    False
dtype: bool
```

In the example, the `.str.contains()` function is used to check if each element of the Series `s` contains the letter 'a'. The resulting Series `contains_a` has boolean values, where `True` indicates that the corresponding element contains 'a', and `False` indicates that it does not.

This function is useful for performing string matching operations and filtering data based on specific patterns or substrings within a Series. It allows you to identify elements that meet certain criteria or create boolean masks for conditional data selection.

*You can use **conditional** markers such as | (or)*
```python
salaries["Designation"].str.contains("Scientist|AI")
```
*Note: if you add spaces it means you'd like there to be a space in that given string (e.g. "Scientist " instead of just "Scientist" as it is now)

* * *
## np.select(cond, val, default=val)

The `np.select()` function in NumPy is used to apply multiple conditions to an array-like object and return an array with corresponding values based on the conditions.

**Function signature:**
```python
numpy.select(condlist, choicelist, default=0)
```

**Parameters:**
- `condlist`: A list of conditions. Each condition is a boolean array-like object or a boolean mask. The conditions are evaluated in order, and the corresponding `choicelist` value is selected when a condition is met.
- `choicelist`: A list of values. Each value corresponds to a condition in `condlist`. The value is selected when the corresponding condition is met.
- `default` (optional): The default value to use when none of the conditions are met. The default is `0`.

**Example of use:**
```python
import numpy as np

# Create a sample array
arr = np.array([1, 2, 3, 4, 5])

# Apply conditions and select values
result = np.select([arr < 3, arr > 4], ['A', 'B'], default='C')
```

The resulting `result` will be:
```
['A' 'A' 'C' 'C' 'B']
```

In the example, the `np.select()` function is used to apply conditions to the elements of the array `arr`. The conditions `[arr < 3, arr > 4]` are evaluated in order. If an element satisfies the first condition (`arr < 3`), the corresponding value `'A'` is selected. If an element satisfies the second condition (`arr > 4`), the corresponding value `'B'` is selected. If none of the conditions are met, the default value `'C'` is selected.

This function is useful for applying complex conditional logic to arrays and assigning different values based on specified conditions. It allows for efficient and vectorized operations on large arrays.

**Datacamp example**
![Pasted image 20230710171000](/images/Pasted%20image%2020230710171000.png)
![Pasted image 20230710171042](/images/Pasted%20image%2020230710171042.png)
![Pasted image 20230710171059](/images/Pasted%20image%2020230710171059.png)
![Pasted image 20230710171114](/images/Pasted%20image%2020230710171114.png)
![Pasted image 20230710171125](/images/Pasted%20image%2020230710171125.png)
![Pasted image 20230710171151](/images/Pasted%20image%2020230710171151.png)

***
## .countplot()

The `.countplot()` function in the seaborn library is used to display a count plot, which shows the count of observations in each category of a categorical variable. It is a useful tool for visualizing the distribution and frequency of different categories within a single variable.

**Function signature:**
```python
sns.countplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None)
```

**Parameters:**
- `x`, `y` (optional): The categorical variables for the x and y axes, respectively. One of these parameters must be specified.
- `hue` (optional): An additional categorical variable used for grouping the data and producing separate count bars with different colors.
- `data` (optional): The DataFrame or array-like object that contains the data.
- `order`, `hue_order` (optional): The order of the categories in the plot. By default, the categories are ordered based on their appearance in the dataset.
- `orient` (optional): The orientation of the plot, which can be either 'v' (vertical) or 'h' (horizontal).
- `color` (optional): The color for all the bars in the plot. If not specified, the default color will be used.
- `palette` (optional): The color palette to use for the different categories in the plot.
- `saturation` (optional): The intensity of the colors. A value of 1 corresponds to full saturation.
- `dodge` (optional): If set to True, the bars will be plotted side by side when using the `hue` parameter.
- `ax` (optional): The matplotlib Axes object to draw the plot onto. If not specified, a new figure and axes will be created.

**Example of use:**
```python
import seaborn as sns

# Create a sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'A', 'B', 'C', 'C', 'C'],
    'Value': [1, 2, 1, 1, 3, 2, 3, 2]
}
df = pd.DataFrame(data)

# Create a count plot
sns.countplot(x='Category', data=df)
```


***
### Exercises

First we select only non_numeric columns and save it in non_numeric. Then we loop over all columns in non_numeric and f-print the number of unique values in each column with .nunique.

```python
# Filter the DataFrame for object columns
non_numeric = planes.select_dtypes("object")

# Loop through columns
for col in non_numeric.columns:

# Print the number of unique values
print(f"Number of unique values in {col} column: ", non_numeric[col].nunique())
  
```
![Pasted image 20230710173017](/images/Pasted%20image%2020230710173017.png)

![Pasted image 20230710191155](/images/Pasted%20image%2020230710191155.png)
![Pasted image 20230710191217](/images/Pasted%20image%2020230710191217.png)
```python
# Create a list of categories
flight_categories = ["Short-haul", "Medium", "Long-haul"]

# Create short-haul values
short_flights = "0h|1h|2h|3h|4h"

# Create medium-haul values
medium_flights = "5h|6h|7h|8h|9h"

# Create long-haul values
long_flights = "10h|11h|12h|13h|14h|15h|16h"
```
![Pasted image 20230710191426](/images/Pasted%20image%2020230710191426.png)
```python
# Create conditions for values in flight_categories to be created

conditions = [

    (planes["Duration"].str.contains(short_flights)),

    (planes["Duration"].str.contains(medium_flights)),

    (planes["Duration"].str.contains(long_flights))

]

  

# Apply the conditions list to the flight_categories

planes["Duration_Category"] = np.select(conditions, 

                                        flight_categories,

                                        default="Extreme duration")

  

# Plot the counts of each category

sns.countplot(data=planes, x="Duration_Category")

plt.show()
```

![Pasted image 20230710191541](/images/Pasted%20image%2020230710191541.png)

***
## .info()

The `.info()` function in pandas is used to display a concise summary of a DataFrame, including information about the column names, data types, and the number of non-null values.

**Function signature:**
```python
DataFrame.info(verbose=True, buf=None, max_cols=None, memory_usage=None, null_counts=None)
```

**Parameters:**
- `verbose` (optional): Controls the amount of information displayed. If `verbose=True`, all the column names and the count of non-null values are shown. If `verbose=False`, a concise summary is displayed. By default, `verbose=None` displays the summary based on the DataFrame's size.
- `buf` (optional): Specifies the buffer where the output is redirected. If `buf=None`, the output is printed to the console. If `buf` is a writable buffer, the output is stored in that buffer.
- `max_cols` (optional): Sets the maximum number of columns to be displayed in the summary. If `max_cols=None`, all columns are displayed.
- `memory_usage` (optional): Specifies whether to include memory usage information in the summary. By default, it's set to `None`, and memory usage is only included if the DataFrame has numeric columns.
- `null_counts` (optional): Specifies whether to include the count of null values in the summary. By default, it's set to `None`, and the count of null values is included if the DataFrame has null values.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['John', 'Alice', 'Bob'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)

# Display the summary information
df.info()
```

The resulting output will be:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Name    3 non-null      object
 1   Age     3 non-null      int64 
 2   City    3 non-null      object
dtypes: int64(1), object(2)
memory usage: 200.0+ bytes
```

In the example, the `.info()` function is used to display a summary of the DataFrame `df`. The summary includes information about the data types (`Dtype`) of each column, the number of non-null values (`Non-Null Count`), and memory usage (`memory usage`).

The `.info()` function is helpful for quickly understanding the structure and properties of a DataFrame. It provides an overview of the column names, data types, and presence of missing values, allowing you to assess the quality and integrity of the data. Additionally, it provides memory usage information, which can be useful when working with large datasets.


***
## .str.replace()

The `.str.replace()` function is used to replace a substring or pattern in the elements of a Series with a specified value.

**Function signature:**
```python
Series.str.replace(pat, repl, n=-1, case=None, flags=0, regex=True)
```

**Parameters:**
- `pat`: The substring or pattern to be replaced.
- `repl`: The value to replace the matched substrings or patterns with.
- `n` (optional): The maximum number of replacements to make. By default, `-1` replaces all occurrences.
- `case` (optional): If `case=True`, the replacement is case-sensitive. If `case=False`, it is case-insensitive.
- `flags` (optional): Flags controlling the behavior of the regex matching. Common flags include `re.IGNORECASE`, `re.MULTILINE`, `re.DOTALL`, etc.
- `regex` (optional): If `True`, treats the `pat` parameter as a regular expression. If `False`, treats it as a literal string.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
s = pd.Series(['apple', 'banana', 'cherry'])

# Replace 'a' with 'X' in the elements of the Series
s_replaced = s.str.replace('a', 'X')
```

The output `s_replaced` will be:
```
0    Xpple
1    bXnXnX
2    cherry
dtype: object
```

In the example, the `pd.Series.str.replace()` function is used to replace all occurrences of the letter 'a' with the letter 'X' in each element of the Series. The resulting Series `s_replaced` contains the modified strings.

This function is particularly useful for performing string manipulations and transformations on Series objects in pandas, allowing you to modify and clean textual data efficiently.


***
## .astype()

The `.astype()` function is used to change the data type of a pandas Series or DataFrame to a specified type.

**Function signature:**
```python
Series.astype(dtype, copy=True, errors='raise')
```

**Parameters:**
- `dtype`: The data type to which the Series or DataFrame will be converted. It can be a numpy dtype or Python type.
- `copy` (optional): If `copy=True`, a new object with the specified data type is returned. If `copy=False`, the operation is performed in-place.
- `errors` (optional): Specifies how to handle errors if the conversion fails. Possible values are 'raise', 'ignore', and 'coerce'. By default, 'raise' is used, which raises an exception. 'ignore' skips the conversion, and 'coerce' converts invalid values to `NaN`.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
s = pd.Series([1, 2, 3, 4, 5])

# Change the data type of the Series to float
s_float = s.astype(float)
```

The resulting `s_float` will be:
```
0    1.0
1    2.0
2    3.0
3    4.0
4    5.0
dtype: float64
```

In the example, the `.astype()` function is used to convert the data type of the Series `s` from integer to float. The resulting Series `s_float` contains the same values, but with the data type changed to `float64`.

This function is helpful when you need to convert the data type of a Series or DataFrame to perform specific operations or ensure compatibility with other parts of your data analysis pipeline.


***
## .groupby()

The `.groupby()` function in pandas is used to group data in a DataFrame based on one or more columns. It is often followed by an aggregation function to compute summary statistics or perform operations on the grouped data.

**Function signature:**
```python
DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, observed=False, dropna=True)
```

**Parameters:**
- `by`: Specifies the column(s) or key(s) to group the DataFrame by. It can be a single column name or a list of column names. Alternatively, it can also be a function or a dictionary.
- `axis`: Specifies the axis to perform the grouping on. `axis=0` (default) groups by rows, while `axis=1` groups by columns.
- `level`: Specifies the level(s) (if the DataFrame has a MultiIndex) to group by.
- `as_index`: If `as_index=True` (default), the grouped column(s) become the index of the resulting DataFrame. If `as_index=False`, the grouped column(s) are kept as regular columns.
- `sort`: If `sort=True` (default), the resulting DataFrame is sorted by the group keys. If `sort=False`, the order of rows in the resulting DataFrame is arbitrary.
- `group_keys`: If `group_keys=True` (default), the group keys are included in the resulting DataFrame. If `group_keys=False`, the group keys are not included.
- `squeeze`: If `squeeze=True`, a single column DataFrame is returned when grouping by a single column. If `squeeze=False` (default), a DataFrame with a MultiIndex is returned.
- `observed`: If `observed=True`, only observed values of categorical groupers are used. If `observed=False` (default), all values of categorical groupers are used.
- `dropna`: If `dropna=True` (default), groups containing NA/null values are excluded. If `dropna=False`, NA/null values are treated as a valid group key.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Group the DataFrame by 'Category' and compute the mean value
grouped = df.groupby('Category').mean()
```

The resulting `grouped` DataFrame will be:
```
         Value
Category      
A            3
B            3
```

In the example, the `.groupby()` function is used to group the DataFrame `df` by the 'Category' column. The resulting `grouped` DataFrame contains the mean value of the 'Value' column for each unique category.

Grouping data allows for analyzing and summarizing subsets of data based on specific criteria. It is often used in combination with aggregation functions such as `.sum()`, `.mean()`, `.count()`, etc., to compute summary statistics for each group. Additionally, `.groupby()` can be used with multiple columns for more complex grouping scenarios.

***
## .transform()

The `.transform()` function in pandas is used to perform group-wise transformations on a DataFrame or Series, where the resulting transformed values have the same shape as the original input.

**Function signature:**
```python
DataFrame.transform(func, axis=0, *args, **kwargs)
```

**Parameters:**
- `func`: The transformation function to apply to each group. It can be a built-in function, a lambda function, or a user-defined function.
- `axis`: Specifies the axis along which the transformation is applied. `axis=0` (default) applies the transformation vertically (column-wise), while `axis=1` applies it horizontally (row-wise).
- `*args` and `**kwargs`: Additional arguments and keyword arguments that can be passed to the transformation function.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Group the DataFrame by 'Category' and transform the 'Value' column by subtracting its mean
transformed = df.groupby('Category')['Value'].transform(lambda x: x - x.mean())
```

The resulting `transformed` Series will be:
```
0   -2
1   -1
2    0
3    1
4    2
Name: Value, dtype: int64
```

In the example, the `.transform()` function is used to group the DataFrame `df` by the 'Category' column and transform the 'Value' column by subtracting its group-wise mean. The resulting `transformed` Series contains the subtracted values, where each value is the original value minus the mean of its respective group.

The `.transform()` function is commonly used in scenarios where you want to apply group-wise operations while maintaining the shape and alignment of the original data. It allows you to perform calculations based on group characteristics and incorporate those transformed values back into the original DataFrame or Series.

***
## .hist()

The `.hist()` function is used to create histograms, which represent the distribution of a numerical variable in the form of bins and their corresponding frequencies.

**Function signature:**
```python
Series.hist(
    bins=10,
    range=None,
    density=False,
    weights=None,
    cumulative=False,
    bottom=None,
    histtype='bar',
    align='mid',
    orientation='vertical',
    rwidth=None,
    log=False,
    color=None,
    label=None,
    stacked=False,
    *,
    data=None,
    **kwargs
)
```

**Parameters:**
- `bins` (optional): Specifies the number of bins or the specific bin edges to use in the histogram.
- `range` (optional): Specifies the range of values to consider for the histogram bins.
- `density` (optional): If `density=True`, the histogram will display the probability density instead of the count.
- `weights` (optional): An array-like object of weights to apply to each data point.
- `cumulative` (optional): If `cumulative=True`, a cumulative histogram is created.
- `bottom` (optional): An array-like object representing the bottom baseline of the histogram bars.
- `histtype` (optional): The type of histogram to plot. It can be `'bar'`, `'barstacked'`, `'step'`, `'stepfilled'`, or `'step'`.
- `align` (optional): The alignment of the histogram bars. It can be `'mid'`, `'left'`, or `'right'`.
- `orientation` (optional): The orientation of the histogram. It can be `'vertical'` or `'horizontal'`.
- `rwidth` (optional): The relative width of the histogram bars as a fraction of the bin width.
- `log` (optional): If `log=True`, the y-axis will have a logarithmic scale.
- `color` (optional): The color or a sequence of colors to use for the histogram bars.
- `label` (optional): A label to assign to the histogram for legend purposes.
- `stacked` (optional): If `stacked=True`, multiple histograms will be stacked on top of each other.

**Example of use:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample Series
s = pd.Series(np.random.randn(1000))

# Create a histogram
s.hist(bins=20, color='steelblue', edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Series')

# Display the histogram
plt.show()
```

In the example, the `.hist()` function is used to create a histogram of the Series `s`. The histogram is visualized using `matplotlib.pyplot` by calling `plt.hist()`. Additional parameters like the number of bins (`bins`), color (`color`), edge color (`edgecolor`), and labels are provided to customize the appearance of the histogram.

Histograms are commonly used to analyze the distribution of numerical data, understand its shape, and identify any patterns or outliers. They are a fundamental tool for exploratory data analysis and provide insights into the underlying characteristics of a variable.

***
## sns.histplot()

**Description:**
The `sns.histplot()` function is used to create histograms, which represent the distribution of a numerical variable in the form of bins and their corresponding frequencies. It is part of the seaborn library and provides enhanced functionality compared to the basic histogram function in matplotlib.

**Function signature:**
```python
seaborn.histplot(
    data=None,
    *,
    x=None,
    y=None,
    hue=None,
    weights=None,
    stat='count',
    bins='auto',
    binwidth=None,
    binrange=None,
    discrete=None,
    cumulative=False,
    common_bins=True,
    common_norm=True,
    multiple='layer',
    element='bars',
    fill=True,
    shrink=1,
    kde=False,
    kde_kws=None,
    line_kws=None,
    thresh=0,
    pthresh=None,
    pmax=None,
    cbar=False,
    cbar_ax=None,
    cbar_kws=None,
    palette=None,
    hue_order=None,
    hue_norm=None,
    color=None,
    log_scale=None,
    legend=True,
    ax=None,
    **kwargs
)
```

**Parameters:**
- `data` (optional): Specifies the DataFrame or array-like object that contains the data.
- `x` and `y` (optional): The variables to be plotted on the x and y axes, respectively. One of these parameters must be specified.
- `hue` (optional): An additional categorical variable used for grouping the data and creating separate histograms with different colors.
- `weights` (optional): An array-like object of weights to apply to each data point.
- `stat` (optional): The type of statistic to compute for each bin. It can be `'count'`, `'frequency'`, `'density'`, or `'probability'`.
- `bins` (optional): Specifies the number of bins to use in the histogram. It can be an integer or a specification such as `'auto'`, `'sturges'`, or `'fd'`.
- `discrete` (optional): If `discrete=True`, the histogram is treated as discrete data, and each unique value will have its own bin.
- `cumulative` (optional): If `cumulative=True`, a cumulative histogram is created.
- `multiple` (optional): Specifies how multiple histograms are visualized when there is a grouping variable. It can be `'layer'`, `'stack'`, or `'dodge'`.
- `element` (optional): The type of visual element to use. It can be `'bars'`, `'step'`, `'poly'`, or `'auto'`.
- `fill` (optional): If `fill=True`, the bars are filled with color. If `fill=False`, only the edges of the bars are shown.
- `kde` (optional): If `kde=True`, a kernel density estimate plot is overlaid on the histogram.
- `color` (optional): The color or a sequence of colors to use for the histogram bars.
- `log_scale` (optional): If `log_scale=True`, the y-axis will have a logarithmic scale.
- `legend` (optional): If `legend=True`, a legend will be included in the plot.

**Example of use:**
```python
import seaborn as sns

# Create a sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Create a histogram plot
sns.histplot(data=df, x='Value', hue='Category', bins=10, kde=True, fill=True)

# Display the plot
plt.show()
```

In the example, the `.histplot()` function is used to create a histogram plot of the DataFrame `df`. The histogram shows the distribution of the 'Value' variable, with separate histograms for each category in the 'Category' variable. Additional parameters like the number of bins (`bins`), inclusion of a kernel density estimate plot (`kde`), color filling (`fill`), and inclusion of a legend (`legend`) are provided to customize the appearance of the histogram plot.

The `.histplot()` function in seaborn provides a high-level interface to create visually appealing and informative histograms. It offers various customization options and can handle complex plotting scenarios, making it a versatile tool for exploratory data analysis and visualization.
***
### Exercises

![Pasted image 20230710203014](/images/Pasted%20image%2020230710203014.png)
![Pasted image 20230710203055](/images/Pasted%20image%2020230710203055.png)
![Pasted image 20230710204536](/images/Pasted%20image%2020230710204536.png)
![Pasted image 20230710204641](/images/Pasted%20image%2020230710204641.png)
```python
# Preview the column
print(planes["Duration"].head())


# Remove the string character
planes["Duration"] = planes["Duration"].str.replace("h", "")
 

# Convert to float data type
planes["Duration"] = planes["Duration"].astype(float)


# Plot a histogram
sns.histplot(x='Duration', data=planes)
plt.show()
```
![Pasted image 20230710205338](/images/Pasted%20image%2020230710205338.png)

![Pasted image 20230714120053](/images/Pasted%20image%2020230714120053.png)
```python
# Price standard deviation by Airline
planes["airline_price_st_dev"] = planes.groupby("Airline")["Price"].transform(lambda x: x.std())

print(planes[["Airline", "airline_price_st_dev"]].value_counts())


# Median Duration by Airline
planes["airline_median_duration"] = planes.groupby("Airline")["Duration"].transform(lambda x: x.median())

print(planes[["Airline","airline_median_duration"]].value_counts())


# Mean Price by Destination
planes["price_destination_mean"] = planes.groupby("Destination")["Price"].transform(lambda x: x.mean())

print(planes[["Destination","price_destination_mean"]].value_counts())
```

* * *
## .describe()

The `.describe()` function in pandas is used to generate descriptive statistics of a DataFrame or Series, providing summary information about the central tendency, dispersion, and shape of the data.

**Function signature:**
```python
DataFrame.describe(percentiles=None, include=None, exclude=None)
```

**Parameters:**
- `percentiles` (optional): Specifies the percentiles to include in the summary. The default is `[.25, .5, .75]`, which includes the 25th, 50th (median), and 75th percentiles.
- `include` (optional): Specifies the data types to include in the summary. It can be a list of data types or None, which includes all types. By default, only numeric columns are included.
- `exclude` (optional): Specifies the data types to exclude from the summary. It can be a list of data types or None, which excludes no types. By default, None.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Category': ['A', 'B', 'C', 'A', 'B'],
    'Value': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# Generate descriptive statistics
summary = df.describe()
```

The resulting `summary` DataFrame will be:
```
          Value
count  5.000000
mean   3.000000
std    1.581139
min    1.000000
25%    2.000000
50%    3.000000
75%    4.000000
max    5.000000
```

In the example, the `.describe()` function is used to generate descriptive statistics for the DataFrame `df`. The resulting `summary` DataFrame includes key summary statistics such as count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum for the numeric column 'Value'.

The `.describe()` function provides a quick overview of the distribution and basic statistical properties of the data. It is helpful for gaining initial insights, detecting outliers, understanding the range of values, and identifying potential issues or anomalies in the dataset.

***
## .quantile()

The `.quantile()` function in pandas is used to compute the quantiles of a Series or DataFrame. Quantiles represent the values that partition a dataset into equal-sized intervals. It is handy for calculating IQR and subsequently outliers.

**Function signature:**
```python
Series.quantile(q=0.5, interpolation='linear')
```

**Parameters:**
- `q`: Specifies the quantile(s) to compute. It can be a float representing a single quantile (e.g., 0.5 for the median) or a list/array of quantiles.
- `interpolation` (optional): Specifies the method to use for interpolation when the desired quantile lies between two data points. It can be `'linear'`, `'lower'`, `'higher'`, `'midpoint'`, or `'nearest'`.

**Example of use:**
```python
import pandas as pd

# Create a sample Series
s = pd.Series([1, 2, 3, 4, 5])

# Compute the median (50th percentile)
median = s.quantile(0.5)

# Compute the 25th and 75th percentiles
percentiles = s.quantile([0.25, 0.75])

# Calculating IQR
iqr = s.quantile(0.75) - s.quantile(0.25)

# Identifying outliers
upper_limit = s.quantile(0.75) + (1.5 * iqr)
lower_limit = s.quantile(0.25) - (1.5 * iqr)

# Subsetting to explore outliers: In a dataframe this would be handy as you could subset for values below the lower limit and above the upper limit, respectively. This would allow you to explore these observations in detail and in relation to other columns in the dataframe

# Subsetting to remove outliers: In the opposite way you could also keep values between these upper and lower limit thereby removing all outliers on both sides.
```

In the example, the `.quantile()` function is used to compute quantiles of the Series `s`. The `q` parameter is set to 0.5 to compute the median (50th percentile) in the first case. In the second case, the `q` parameter is a list `[0.25, 0.75]` to compute the 25th and 75th percentiles.

The resulting values will be:
- `median`: 3.0
- `percentiles`:
  - 25th percentile: 2.0
  - 75th percentile: 4.0

The `.quantile()` function is helpful for understanding the distribution and spread of numerical data. It allows you to compute specific quantiles such as the median, quartiles, or any desired percentiles. This information can be useful for data exploration, understanding the central tendency, and identifying data points that fall within certain ranges.
***
### Exercises
![[Pasted image 20230714151840.png]]
![[Pasted image 20230714151854.png]]
![[Pasted image 20230714151903.png]]
![[Pasted image 20230714151937.png]]
![[Pasted image 20230714152651.png]]
![[Pasted image 20230714152853.png]]
```python
# Plot a histogram of flight prices
sns.histplot(data=planes, x="Price")
plt.show()

# Display descriptive statistics for flight duration
print(planes["Duration"].describe())
```
![[Pasted image 20230714152601.png]]![[Pasted image 20230714152740.png]]
![[Pasted image 20230714153031.png]]
```python
# Find the 75th and 25th percentiles
price_seventy_fifth = planes["Price"].quantile(0.75)
price_twenty_fifth = planes["Price"].quantile(0.25)

# Calculate iqr
prices_iqr = price_seventy_fifth - price_twenty_fifth

# Calculate the thresholds
upper = price_seventy_fifth + (1.5 * prices_iqr)
lower = price_twenty_fifth - (1.5 * prices_iqr)

# Subset the data
planes = planes[(planes["Price"] > lower) & (planes["Price"] < upper)]
  
print(planes["Price"].describe())
```
![[Pasted image 20230714153409.png]]
***
## .read_csv()

The `.read_csv()` function in pandas is used to read a CSV (Comma-Separated Values) file and load its contents into a pandas DataFrame.

**Function signature:**
```python
pandas.read_csv(filepath_or_buffer, sep=',', delimiter=None, header='infer', names=None, index_col=None, usecols=None, dtype=None, parse_dates=False, infer_datetime_format=False, dayfirst=False, nrows=None, skiprows=None, skipfooter=0, na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, encoding=None, squeeze=False, thousands=None, decimal='.', lineterminator=None, comment=None, error_bad_lines=True, warn_bad_lines=True, on_bad_lines=None, delimiter_whitespace=False, low_memory=True, memory_map=False, float_precision=None)
```

**Parameters:**
- `filepath_or_buffer`: Specifies the path to the CSV file or a URL or any object with a `read()` method.
- `sep` (optional): Specifies the delimiter character to separate fields. The default is `','`.
- `header` (optional): Specifies the row number to use as column names. The default is `'infer'`, which uses the first row as column names. If `header=None`, no column names are inferred, and the columns are assigned integer labels.
- `index_col` (optional): Specifies the column(s) to use as the row index. It can be a single column name or a list of column names.
- `parse_dates` (optional): If `parse_dates=True`, the function attempts to parse dates in the CSV file and convert them into datetime objects.
- `dtype` (optional): Specifies the data type for specific columns or the entire DataFrame.
- `na_values` (optional): Specifies the values to consider as missing or NA values.
- Many more parameters are available to handle various file formats and data reading options.

**Example of use:**
```python
import pandas as pd

# Read a CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Print the first few rows of the DataFrame
print(df.head())

# Parsing dates in specific column
df = pd.read_csv("data.csv", parse_dates=["start_date", "end_date"])
```

In the example, the `.read_csv()` function is used to read a CSV file named `'data.csv'` into a pandas DataFrame. The resulting DataFrame is stored in the variable `df`. The `.head()` function is then used to display the first few rows of the DataFrame.

The `.read_csv()` function provides a convenient way to read CSV files and import them into pandas for further data analysis and manipulation. It allows you to handle various file formats, specify custom delimiters, handle missing values, and perform various data reading configurations.

***
## to_datetime()

The `.to_datetime()` function in pandas is used to convert a column or Series of data into a pandas DateTime format. It allows for the transformation of date and time data into a standardized format that can be easily manipulated and analyzed.

**Function signature:**
```python
pandas.to_datetime(arg, format=None, errors='raise', dayfirst=False, yearfirst=False, utc=None, box=True, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=False)
```

**Parameters:**
- `arg`: Specifies the input data to be converted to DateTime format. It can be a single value, an array-like object, a Series, or a column of a DataFrame.
- `format` (optional): Specifies the format of the input data as a string. The format string uses the same directives as the `strftime()` function from the datetime module.
- `errors` (optional): Specifies how errors should be handled during the conversion. It can be set to `'raise'` to raise an exception, `'coerce'` to set invalid values to `NaT`, or `'ignore'` to pass through invalid values as-is.
- `dayfirst` (optional): If `dayfirst=True`, it indicates that the day should be the first element when ambiguous.
- `yearfirst` (optional): If `yearfirst=True`, it indicates that the year should be the first element when ambiguous.
- `utc` (optional): Specifies whether to treat the input data as UTC (Coordinated Universal Time). It can be set to `True` or `False`.
- Many more parameters are available to handle various datetime-related operations and configurations.

**Example of use:**
```python
import pandas as pd

# Create a sample Series with date strings
dates = pd.Series(['2021-01-01', '2021-01-02', '2021-01-03'])

# Convert the strings to DateTime format
date_times = pd.to_datetime(dates)

# Print the resulting DateTime Series
print(date_times)
```

The resulting `date_times` Series will be:
```
0   2021-01-01
1   2021-01-02
2   2021-01-03
dtype: datetime64[ns]
```

In the example, the `.to_datetime()` function is used to convert a Series of date strings into DateTime format. The resulting `date_times` Series contains the same dates but in the standardized DateTime format.

The `.to_datetime()` function is useful for converting string representations of dates or times into a pandas DateTime format. Once in this format, pandas provides a range of powerful functionalities for working with dates and times, such as date arithmetic, resampling, indexing, and more.

![[Pasted image 20230714155654.png]]

### Using dt.month, dt.day, dt.year
![[Pasted image 20230714155757.png]]

***
## .lineplot()

The `.lineplot()` function is used to create a line plot, also known as a line chart or a line graph, in order to visualize the relationship between two numeric variables over a continuous interval.

**Function signature:**
```python
seaborn.lineplot(
    *,
    x=None,
    y=None,
    hue=None,
    size=None,
    style=None,
    data=None,
    palette=None,
    hue_order=None,
    hue_norm=None,
    sizes=None,
    size_order=None,
    size_norm=None,
    dashes=True,
    markers=None,
    style_order=None,
    units=None,
    estimator='mean',
    ci=95,
    n_boot=1000,
    seed=None,
    sort=True,
    err_style='band',
    err_kws=None,
    legend='auto',
    ax=None,
    **kwargs
)
```

**Parameters:**
- `x` and `y` (optional): The variables to be plotted on the x and y axes, respectively. They can be column names from the DataFrame or arrays-like objects.
- `hue` (optional): An additional categorical variable used for grouping the data and creating separate lines with different colors.
- `size` (optional): An additional categorical variable used for grouping the data and creating separate lines with different sizes.
- `style` (optional): An additional categorical variable used for grouping the data and creating separate lines with different styles.
- `data` (optional): Specifies the DataFrame or long-form data object that contains the data to be plotted.
- Many more parameters are available to customize the appearance and behavior of the line plot.

**Example of use:**
```python
import seaborn as sns

# Create a sample DataFrame
data = {
    'Year': [2010, 2011, 2012, 2013, 2014],
    'Sales': [100, 150, 120, 200, 180],
    'Profit': [10, 20, 15, 30, 25]
}
df = pd.DataFrame(data)

# Create a line plot
sns.lineplot(x='Year', y='Sales', data=df)

# Display the plot
plt.show()
```

In the example, the `.lineplot()` function is used to create a line plot of the 'Sales' variable against the 'Year' variable in the DataFrame `df`. The resulting line plot visualizes the trend and changes in sales over the years.

Line plots are useful for displaying the relationship between two continuous variables. They can show trends, patterns, or fluctuations in the data over time or any continuous scale. Additionally, line plots can incorporate categorical variables by using hue, size, or style parameters to represent different groups or dimensions in the data.

***
### Exercises
![[Pasted image 20230714160629.png]]
```python
# Import divorce.csv, parsing the appropriate columns as dates in the import
divorce = pd.read_csv('divorce.csv', parse_dates = ["divorce_date", "dob_man", "dob_woman", "marriage_date"])

print(divorce.dtypes)
```
![[Pasted image 20230714160858.png]]
![[Pasted image 20230714160928.png]]
![[Pasted image 20230714160943.png]]
```python
# Convert the marriage_date column to DateTime values
divorce["marriage_date"] = pd.to_datetime(divorce["marriage_date"])
```
![[Pasted image 20230714162114.png]]
![[Pasted image 20230714162409.png]]
```python
# Define the marriage_year column
divorce["marriage_year"] = divorce["marriage_date"].dt.year

# Create a line plot showing the average number of kids by year
sns.lineplot(data=divorce, x="marriage_year", y="num_kids")
plt.show()
```
![[Pasted image 20230714162545.png]]

***
## .corr()

The `.corr()` function in pandas is used to compute the correlation between columns of a DataFrame or a Series. It calculates the pairwise correlation coefficients, which measure the statistical relationship between two variables.

**Function signature:**
```python
DataFrame.corr(method='pearson', min_periods=1)
```

**Parameters:**
- `method` (optional): Specifies the correlation method to use. It can be `'pearson'` (default), `'kendall'`, or `'spearman'`.
- `min_periods` (optional): Specifies the minimum number of valid pairs required to compute the correlation. If there are fewer valid pairs, the result will be NaN.

**Example of use:**
```python
import pandas as pd

# Create a sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 6, 8, 10],
    'C': [3, 6, 9, 12, 15]
}
df = pd.DataFrame(data)

# Compute the correlation matrix
correlation_matrix = df.corr()
```

The resulting `correlation_matrix` DataFrame will be:
```
     A    B    C
A  1.0  1.0  1.0
B  1.0  1.0  1.0
C  1.0  1.0  1.0
```

In the example, the `.corr()` function is used to compute the correlation matrix of the DataFrame `df`. The resulting `correlation_matrix` DataFrame contains the correlation coefficients between each pair of columns. Since the data in each column is perfectly positively correlated, all correlation coefficients are 1.0.

The `.corr()` function is useful for analyzing the relationship between variables in a dataset. It helps identify whether variables are positively correlated, negatively correlated, or uncorrelated. The correlation coefficients range between -1 and 1, where -1 indicates a perfect negative correlation, 1 indicates a perfect positive correlation, and 0 indicates no correlation.

***
## .heatmap()

The `.heatmap()` function in seaborn is used to plot a heatmap, which visualizes the matrix-like data in the form of a colored grid. It provides a visual representation of the data using different colors to indicate the values.

**Function signature:**
```python
seaborn.heatmap(
    data,
    vmin=None,
    vmax=None,
    cmap=None,
    center=None,
    robust=False,
    annot=None,
    fmt='.2g',
    annot_kws=None,
    linewidths=0,
    linecolor='white',
    cbar=True,
    cbar_kws=None,
    cbar_ax=None,
    square=False,
    xticklabels='auto',
    yticklabels='auto',
    mask=None,
    ax=None,
    **kwargs
)
```

**Parameters:**
- `data`: Specifies the input data for the heatmap. It can be a 2D array, a DataFrame, or a rectangular NumPy array.
- `vmin` and `vmax` (optional): Specifies the minimum and maximum values of the colormap. If not provided, the minimum and maximum values of the input data will be used.
- `cmap` (optional): Specifies the colormap to be used. It can be a string referring to a named colormap or a matplotlib colormap object.
- `annot` (optional): Specifies whether to annotate the heatmap cells with the data values. If `True`, the data values will be displayed. If a rectangular dataset, boolean array, or DataFrame is passed, the corresponding cells will be annotated.
- `fmt` (optional): Specifies the format string for the annotations. It controls the number of decimal places and other formatting options.
- Many more parameters are available to customize the appearance and behavior of the heatmap.

**Example of use:**
```python
import seaborn as sns

# Create a sample DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data)

# Create a heatmap
sns.heatmap(data=df, annot=True, cmap='YlGnBu')

# Display the plot
plt.show()
```

In the example, the `.heatmap()` function is used to create a heatmap of the DataFrame `df`. The resulting heatmap visualizes the values in the DataFrame using different colors, and annotations are added to each cell to display the data values.

Heatmaps are useful for displaying and exploring patterns or relationships in matrix-like data. They are commonly used to visualize correlation matrices, confusion matrices, and other forms of tabular data. Heatmaps allow for quick identification of high and low values, patterns, clusters, or trends within the data.

![[Pasted image 20230714164209.png]]
***
## .scatterplot()

The `.scatterplot()` function in seaborn is used to create a scatter plot, which visualizes the relationship between two continuous variables. It displays data points as individual markers on a 2D coordinate system, where the x-axis represents one variable and the y-axis represents the other variable.

**Function signature:**
```python
seaborn.scatterplot(
    *,
    x=None,
    y=None,
    hue=None,
    style=None,
    size=None,
    data=None,
    palette=None,
    hue_order=None,
    hue_norm=None,
    sizes=None,
    size_order=None,
    size_norm=None,
    markers=True,
    style_order=None,
    x_bins=None,
    y_bins=None,
    units=None,
    estimator=None,
    ci=95,
    n_boot=1000,
    alpha=None,
    x_jitter=None,
    y_jitter=None,
    legend='auto',
    ax=None,
    **kwargs
)
```

**Parameters:**
- `x` and `y` (optional): The variables to be plotted on the x and y axes, respectively. They can be column names from the DataFrame or arrays-like objects.
- `hue` (optional): An additional categorical variable used for grouping the data and creating separate markers with different colors.
- `style` (optional): An additional categorical variable used for grouping the data and creating separate markers with different styles.
- `size` (optional): An additional categorical or numeric variable used for grouping the data and creating markers with different sizes.
- `data` (optional): Specifies the DataFrame or long-form data object that contains the data to be plotted.
- Many more parameters are available to customize the appearance and behavior of the scatter plot.

**Example of use:**
```python
import seaborn as sns

# Create a sample DataFrame
data = {
    'Height': [170, 165, 180, 175, 160],
    'Weight': [65, 60, 75, 70, 55],
    'Gender': ['M', 'F', 'M', 'M', 'F']
}
df = pd.DataFrame(data)

# Create a scatter plot
sns.scatterplot(x='Height', y='Weight', hue='Gender', data=df)

# Display the plot
plt.show()
```

In the example, the `.scatterplot()` function is used to create a scatter plot of the 'Height' variable against the 'Weight' variable in the DataFrame `df`. The resulting scatter plot visualizes the relationship between height and weight, with different markers and colors representing different genders.

Scatter plots are commonly used to explore the relationship between two continuous variables. They can help identify patterns, trends, clusters, or outliers in the data. By incorporating categorical variables through hue, style, or size parameters, scatter plots can provide additional insights and facilitate deeper analysis of the data.

***
## .pairplot()

The `.pairplot()` function in seaborn is used to create a grid of scatter plots, also known as a pair plot, which displays the pairwise relationships between multiple variables in a dataset. It provides a visual representation of the correlations and distributions of variables.

**Function signature:**
```python
seaborn.pairplot(
    data,
    *,
    hue=None,
    hue_order=None,
    palette=None,
    vars=None,
    x_vars=None,
    y_vars=None,
    kind='scatter',
    diag_kind='auto',
    markers=None,
    height=2.5,
    aspect=1,
    dropna=True,
    plot_kws=None,
    diag_kws=None,
    grid_kws=None,
    size=None
)
```

**Parameters:**
- `data`: Specifies the input data for the pair plot. It can be a DataFrame or any other long-form data object.
- `hue` (optional): An additional categorical variable used for coloring the scatter plots based on different categories.
- `vars`, `x_vars`, `y_vars` (optional): Specify the variables to be included in the pair plot. They can be column names from the DataFrame or arrays-like objects.
- `kind` (optional): Specifies the type of plot for the off-diagonal subplots. It can be `'scatter'`, `'reg'`, `'resid'`, `'kde'`, or `'hex'`.
- `diag_kind` (optional): Specifies the type of plot for the diagonal subplots. It can be `'auto'`, `'hist'`, `'kde'`, or `None`.
- Many more parameters are available to customize the appearance and behavior of the pair plot.

**Example of use:**
```python
import seaborn as sns

# Create a sample DataFrame
data = {
    'SepalLength': [5.1, 4.9, 4.7, 4.6, 5.0],
    'SepalWidth': [3.5, 3.0, 3.2, 3.1, 3.6],
    'PetalLength': [1.4, 1.4, 1.3, 1.5, 1.4],
    'PetalWidth': [0.2, 0.2, 0.2, 0.2, 0.2],
    'Species': ['Setosa', 'Setosa', 'Setosa', 'Setosa', 'Setosa']
}
df = pd.DataFrame(data)

# Create a pair plot
sns.pairplot(data=df, hue='Species')

# Display the plot
plt.show()
```

In the example, the `.pairplot()` function is used to create a pair plot of the variables in the DataFrame `df`. The resulting pair plot displays scatter plots for each pairwise combination of variables and differentiates the data points by coloring them based on the 'Species' variable.

Pair plots are useful for exploring the relationships and distributions of multiple variables in a dataset. They allow for the identification of patterns, correlations, and potential outliers in the data. By incorporating hue parameters, pair plots can provide further insights by visualizing the data based on additional categorical variables.

![[Pasted image 20230714164653.png]]
![[Pasted image 20230714164744.png]]
***
### Exercises
![[Pasted image 20230714165034.png]]
![[Pasted image 20230714165044.png]]
![[Pasted image 20230714165137.png]]
```python
# Create the scatterplot
sns.scatterplot(data=divorce, x='marriage_duration', y='num_kids')
plt.show()
```
![[Pasted image 20230714165202.png]]
![[Pasted image 20230714165331.png]]
```python
# Create a pairplot for income_woman and marriage_duration
sns.pairplot(data=divorce, vars=['income_woman', 'marriage_duration'])
plt.show()
```
![[Pasted image 20230714165317.png]]
***
## .kdeplot()

The `.kdeplot()` function in seaborn is used to create a kernel density estimate plot, which visualizes the distribution of a single variable or the joint distribution of two variables. It provides a smoothed representation of the data's underlying probability density function.

**Function signature:**
```python
seaborn.kdeplot(
    data=None,
    *,
    x=None,
    y=None,
    shade=False,
    vertical=False,
    kernel='gau',
    bw='scott',
    gridsize=100,
    cut=3,
    clip=None,
    legend=True,
    cumulative=False,
    shade_lowest=None,
    cbar=False,
    cbar_ax=None,
    cbar_kws=None,
    ax=None,
    **kwargs
)
```

**Parameters:**
- `data`: Specifies the input data for the kernel density estimate plot. It can be a DataFrame or any other long-form data object.
- `x` and `y` (optional): The variable(s) to be plotted on the x and/or y axes. They can be column names from the DataFrame or arrays-like objects.
- `shade` (optional): Specifies whether to shade the area under the density curve. If `True`, the area is shaded.
- `kernel` (optional): Specifies the kernel function used for estimation. It can be `'gau'` (Gaussian), `'cos'` (cosine), `'biw'` (biweight), `'epa'` (Epanechnikov), or `'tri'` (triangular).
- `bw` (optional): Specifies the bandwidth estimation method. It can be `'scott'`, `'silverman'`, a scalar bandwidth value, or a callable function.
- `legend` (optional): Specifies whether to display the legend. If `True`, the legend is displayed.
- Many more parameters are available to customize the appearance and behavior of the kernel density estimate plot.

**Example of use:**
```python
import seaborn as sns

# Create a sample DataFrame
data = {
    'Height': [170, 165, 180, 175, 160],
    'Weight': [65, 60, 75, 70, 55]
}
df = pd.DataFrame(data)

# Create a kernel density estimate plot for the 'Height' variable
sns.kdeplot(data=df, x='Height', shade=True)

# Display the plot
plt.show()
```

In the example, the `.kdeplot()` function is used to create a kernel density estimate plot for the 'Height' variable in the DataFrame `df`. The resulting plot displays a smoothed estimate of the distribution of heights, with the area under the curve shaded.

Kernel density estimate plots are useful for visualizing the distribution and density of a single variable or the joint distribution of two variables. They provide a smooth representation of the underlying probability density function, allowing for insights into the shape, mode, and spread of the data. By customizing the shading, kernel function, bandwidth estimation, and other parameters, the appearance and level of detail of the plot can be adjusted to suit the specific requirements of the analysis.

![[Pasted image 20230714165843.png]]
![[Pasted image 20230714170016.png]]
***
### Exercises
![[Pasted image 20230714170232.png]]
```python
sns.scatterplot(data=divorce, x='woman_age_marriage', y='income_woman', hue='education_woman')
plt.show()
```
![[Pasted image 20230714170444.png]]
![[Pasted image 20230714170507.png]]
```python
sns.kdeplot(data=divorce, x='marriage_duration', hue='num_kids')
plt.show()
```
![[Pasted image 20230714170720.png]]
![[Pasted image 20230714170746.png]]
```python
sns.kdeplot(data=divorce, x='marriage_duration', hue='num_kids', cut=0)
plt.show()
```
![[Pasted image 20230714170825.png]]
![[Pasted image 20230714170838.png]]
```python
sns.kdeplot(data=divorce, x='marriage_duration', hue='num_kids', cut=0, cumulative=True)
plt.show()
```
![[Pasted image 20230714170945.png]]
