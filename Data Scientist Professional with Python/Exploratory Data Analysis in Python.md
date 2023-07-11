

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
## sns.countplot

*Used to show the count of observations in each category of a categorical variable.*

The `sns.countplot` function in the seaborn library is used to display a count plot, which shows the count of observations in each category of a categorical variable. It is a useful tool for visualizing the distribution and frequency of different categories within a single variable.

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
## Exercises

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

*Used to display a concise summary of a DataFrame, including its column names, data types, and the count of non-null values.

**Function signature:**
```python
df.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None)
```

**Parameters:**

- `verbose` (optional): Controls the amount of information displayed. If `verbose=True`, all the column names and the count of non-null values are shown. If `verbose=False`, a concise summary is displayed. By default, `verbose=None` displays the summary based on the DataFrame's size.

- `buf` (optional): Specifies the buffer where the output is redirected. If `buf=None`, the output is printed to the console. If `buf` is a writable buffer, the output is stored in that buffer.

- `max_cols` (optional): Sets the maximum number of columns to be displayed in the summary. If `max_cols=None`, all columns are displayed.

- `memory_usage` (optional): Specifies whether to include memory usage information in the summary. By default, it's set to `None`, and memory usage is only included if the DataFrame has numeric columns.

- `null_counts` (optional): Specifies whether to include the count of null values in the summary. By default, it's set to `None`, and the count of null values is included if the DataFrame has null values.


***
## pd.Series.str.replace()

The `pd.Series.str.replace()` function is used to replace a substring or pattern in the elements of a Series with a specified value.

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

The `.histplot()` function is used to create histograms, which represent the distribution of a numerical variable in the form of bins and their corresponding frequencies. It is part of the seaborn library and provides enhanced functionality compared to the basic histogram function in matplotlib.

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
## Exercises

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


test

![](Pasted%20image%2020230711195555.png)
