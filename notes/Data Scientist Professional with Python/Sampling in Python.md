## `.sample()`

In Python, the `.sample()` method is commonly used with data structures like Pandas DataFrames and NumPy arrays to randomly select a subset of elements from the data. This method is useful for tasks such as random sampling of data for analysis, creating random test sets, or shuffling data.

**Method Syntax for Pandas DataFrames:**
```python
DataFrame.sample(n=None, frac=None, replace=False, random_state=None)
```

**Parameters:**
- `n` (int): Specifies the number of random items to select. You can use either `n` or `frac`, but not both. If neither `n` nor `frac` is provided, a single random element is selected by default.
- `frac` (float): Represents the fraction of items to select from the DataFrame. You can use either `frac` or `n`, but not both.
- `replace` (bool): If `True`, sampling is done with replacement, allowing the same item to be selected multiple times. If `False` (default), each item is selected at most once.
- `random_state` (int or RandomState, optional): Controls the random number generator's seed for reproducibility.

**Method Syntax for NumPy Arrays:**
```python
numpy.random.choice(a, size=None, replace=True, p=None)
```

**Parameters (NumPy Version):**
- `a`: The array-like object from which to sample.
- `size` (int or tuple of ints, optional): Determines the shape of the output array. If not provided, a single random item is returned.
- `replace` (bool): If `True`, sampling is done with replacement, allowing the same item to be selected multiple times. If `False` (default), each item is selected at most once.
- `p` (array_like, optional): The probabilities associated with each entry in the input array. If not provided, each entry is assumed to have an equal probability of being selected.

**Examples (Pandas DataFrame and NumPy Array):**

Pandas DataFrame:
```python
import pandas as pd

# Sample DataFrame
data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Randomly sample two rows without replacement
sampled_data = data.sample(n=2, random_state=42)
print(sampled_data)
```

NumPy Array:
```python
import numpy as np

# Sample NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Randomly sample three elements with replacement
sampled_elements = np.random.choice(arr, size=3, replace=True)
print(sampled_elements)
```

In the Pandas DataFrame example, we use the `.sample()` method to randomly select two rows from the DataFrame `data` without replacement. In the NumPy Array example, we use `numpy.random.choice()` to randomly select three elements from the NumPy array `arr` with replacement.

Both methods are handy for generating random samples from data, which can be useful in various data analysis and machine learning tasks.

---
## `np.random.__`

Here is a list of commonly used probability distributions in the `numpy.random` module, along with their names and descriptions:

1. **Uniform Distribution (`numpy.random.uniform()`):**
   - Description: Generates random numbers from a uniform distribution within a specified range `[low, high)`.
   - Syntax:
     ```python
     numpy.random.uniform(low, high, size=None)
     ```


2. **Normal (Gaussian) Distribution (`numpy.random.normal()`):**
   - Description: Generates random numbers from a normal distribution with a specified mean (`loc`) and standard deviation (`scale`).
   - Syntax:
     ```python
     numpy.random.normal(loc=0.0, scale=1.0, size=None)
     ```


3. **Standard Normal (Gaussian) Distribution (`numpy.random.randn()`):**
   - Description: Generates random numbers from a standard normal distribution (mean 0, standard deviation 1).
   - Syntax:
     ```python
     numpy.random.randn(d0, d1, ..., dn)
     ```


4. **Binomial Distribution (`numpy.random.binomial()`):**
   - Description: Generates random numbers from a binomial distribution with parameters `n` (number of trials) and `p` (probability of success).
   - Syntax:
     ```python
     numpy.random.binomial(n, p, size=None)
     ```


5. **Poisson Distribution (`numpy.random.poisson()`):**
   - Description: Generates random numbers from a Poisson distribution with a specified rate (`lam`).
   - Syntax:
     ```python
     numpy.random.poisson(lam, size=None)
     ```


6. **Exponential Distribution (`numpy.random.exponential()`):**
   - Description: Generates random numbers from an exponential distribution with a specified scale (`scale`).
   - Syntax:
     ```python
     numpy.random.exponential(scale=1.0, size=None)
     ```


7. **Gamma Distribution (`numpy.random.gamma()`):**
   - Description: Generates random numbers from a gamma distribution with shape parameter `shape`.
   - Syntax:
     ```python
     numpy.random.gamma(shape, scale=1.0, size=None)
```


8. **Log-Normal Distribution (`numpy.random.lognormal()`):**
   - Description: Generates random numbers from a log-normal distribution with parameters `mean` and `sigma`.
   - Syntax:
     ```python
     numpy.random.lognormal(mean=0.0, sigma=1.0, size=None)
     ```


9. **Geometric Distribution (`numpy.random.geometric()`):**
   - Description: Generates random numbers from a geometric distribution with probability of success `p`.
   - Syntax:
     ```python
     numpy.random.geometric(p, size=None)
     ```


10. **Hypergeometric Distribution (`numpy.random.hypergeometric()`):**
    - Description: Generates random numbers from a hypergeometric distribution with parameters `ngood` (number of good items), `nbad` (number of bad items), and `nsample` (number of samples).
    - Syntax:
      ```python
      numpy.random.hypergeometric(ngood, nbad, nsample, size=None)
      ```


11. **Beta Distribution (`numpy.random.beta()`):**
    - Description: Generates random numbers from a beta distribution with shape parameters `a` and `b`.
    - Syntax:
      ```python
      numpy.random.beta(a, b, size=None)
      ```


12. **Chi-Square Distribution (`numpy.random.chisquare()`):**
    - Description: Generates random numbers from a chi-square distribution with degrees of freedom `df`.
    - Syntax:
      ```python
      numpy.random.chisquare(df, size=None)
      ```


13. **F Distribution (`numpy.random.f()`):**
    - Description: Generates random numbers from an F distribution with parameters `dfnum` (numerator degrees of freedom) and `dfden` (denominator degrees of freedom).
    - Syntax:
      ```python
      numpy.random.f(dfnum, dfden, size=None)
      ```


14. **Uniform Distribution over Integers (`numpy.random.randint()`):**
    - Description: Generates random integers from a uniform distribution within a specified range `[low, high)`.
    - Syntax:
      ```python
      numpy.random.randint(low, high, size=None)
      ```


15. **Multinomial Distribution (`numpy.random.multinomial()`):**
    - Description: Generates random samples from a multinomial distribution with parameters `n` (number of trials) and `pvals` (array of probabilities).
    - Syntax:
      ```python
      numpy.random.multinomial(n, pvals, size=None)
      ```


16. **von Mises Distribution (`numpy.random.vonmises()`):**
    - Description: Generates random numbers from a von Mises distribution with parameters `mu` (mean angle) and `kappa` (concentration).
    - Syntax:
      ```python
      numpy.random.vonmises(mu, kappa, size=None)
      ```


17. **Triangular Distribution (`numpy.random.triangular()`):**
    - Description: Generates random numbers from a triangular distribution within a specified range `[left, mode, right]`.
    - Syntax:
      ```python
      numpy.random.triangular(left, mode, right, size=None)
      ```

---
## `numpy.random.seed()`

The `numpy.random.seed()` function is used to set the seed for the random number generator in NumPy. Setting the seed allows you to control the randomness of random number generation, making the results reproducible.

**Function Syntax:**
```python
numpy.random.seed(seed=None)
```

**Parameters:**
- `seed` (optional): An integer or an array of integers. If provided, it sets the seed for the random number generator. If `seed` is not provided or `None`, the generator is initialized with a system-generated random seed.

**Example:**
```python
import numpy as np

# Set a specific seed for reproducible results
np.random.seed(42)

# Generate random numbers
random_numbers = np.random.rand(5)
print(random_numbers)
```

In this example:
- We use `np.random.seed(42)` to set the seed for the random number generator to the value 42.
- After setting the seed, we generate random numbers using `np.random.rand(5)`. Since we set the seed to 42, the sequence of random numbers generated will be the same every time the code is run.

Setting a seed is useful in situations where you want to ensure that your random experiments or simulations produce the same results each time, which is important for reproducibility and debugging. It allows you to control the randomness while maintaining consistency across different runs of your code.

---
## `numpy.random.normal()`

The `numpy.random.normal()` function is used to generate random numbers from a normal (Gaussian) distribution. The normal distribution is a continuous probability distribution characterized by its mean (μ) and standard deviation (σ). It is also known as the Gaussian distribution.

**Function Syntax:**
```python
numpy.random.normal(loc=0.0, scale=1.0, size=None)
```

**Parameters:**
- `loc` (optional): The mean (μ) of the normal distribution. Default is 0.0.
- `scale` (optional): The standard deviation (σ) of the normal distribution. Default is 1.0.
- `size` (optional): The size of the output, which can be an integer or tuple of integers. If not provided, a single random number is generated.

**Return Value:**
- An array of random numbers drawn from the normal distribution with the specified mean and standard deviation.

**Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random numbers from a normal distribution
mean = 5  # Mean (μ)
std_dev = 2  # Standard deviation (σ)
size = 1000  # Number of random samples

random_samples = np.random.normal(mean, std_dev, size)

# Plot a histogram of the random samples
plt.hist(random_samples, bins=30, density=True, alpha=0.6, color='b', label='Normal Distribution')
plt.xlabel('Random Numbers')
plt.ylabel('Probability Density')
plt.legend(loc='upper right')
plt.title('Random Numbers from Normal Distribution')
plt.show()
```

In this example:
- We use `numpy.random.normal()` to generate `size` random numbers from a normal distribution with a specified mean and standard deviation.
- We then create a histogram to visualize the distribution of these random numbers.

The normal distribution is a fundamental probability distribution and is commonly used in statistical analysis, hypothesis testing, and modeling various natural phenomena. It is characterized by its bell-shaped curve and is widely applicable in many fields, including statistics, finance, and engineering.

---
## `numpy.random.uniform()`

The `numpy.random.uniform()` function is used to generate random numbers from a uniform distribution. In a uniform distribution, all values within a specified range are equally likely to be selected.

**Function Syntax:**
```python
numpy.random.uniform(low=0.0, high=1.0, size=None)
```

**Parameters:**
- `low` (optional): The lower bound of the range from which random numbers are drawn. Default is 0.0.
- `high` (optional): The upper bound of the range (exclusive) from which random numbers are drawn. Default is 1.0.
- `size` (optional): The size of the output, which can be an integer or tuple of integers. If not provided, a single random number is generated.

**Return Value:**
- An array of random numbers drawn from the uniform distribution within the specified range.

**Example:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random numbers from a uniform distribution
low = 0  # Lower bound
high = 10  # Upper bound
size = 1000  # Number of random samples

random_samples = np.random.uniform(low, high, size)

# Plot a histogram of the random samples
plt.hist(random_samples, bins=30, density=True, alpha=0.6, color='b', label='Uniform Distribution')
plt.xlabel('Random Numbers')
plt.ylabel('Probability Density')
plt.legend(loc='upper right')
plt.title('Random Numbers from Uniform Distribution')
plt.show()
```

In this example:
- We use `numpy.random.uniform()` to generate `size` random numbers from a uniform distribution within the specified range `[low, high)`.
- We then create a histogram to visualize the distribution of these random numbers.

The uniform distribution is often used when you want to generate random data that is evenly distributed over a given interval. It is useful in various applications, such as random sampling, simulations, and generating random parameters within a specified range.

---
In NumPy, the `numpy.random` module provides functions for generating random numbers and random data. It is a sub-module of the NumPy library that allows you to work with random numbers, random distributions, and random sampling. Here are some commonly used functions and capabilities within the `numpy.random` module:

1. **Random Number Generation:**
   - `numpy.random.rand()`: Generates random numbers from a uniform distribution over [0, 1).
   - `numpy.random.randn()`: Generates random numbers from a standard normal distribution (mean 0, standard deviation 1).
   - `numpy.random.randint()`: Generates random integers from a specified range.

2. **Random Distributions:**
   - `numpy.random.uniform()`: Generates random numbers from a uniform distribution within a specified range.
   - `numpy.random.normal()`: Generates random numbers from a normal distribution with specified mean and standard deviation.
   - `numpy.random.exponential()`: Generates random numbers from an exponential distribution with a specified scale.

3. **Random Sampling:**
   - `numpy.random.choice()`: Performs random sampling from a given array with or without replacement.
   - `numpy.random.shuffle()`: Shuffles the elements of an array randomly.

4. **Random Seed:**
   - `numpy.random.seed()`: Sets the seed for the random number generator, allowing for reproducible results.

Here are a few examples of how to use some of these functions:

```python
import numpy as np

# Generate random numbers from a uniform distribution
random_uniform = np.random.rand(5)
print(random_uniform)

# Generate random integers within a specified range
random_integers = np.random.randint(1, 10, size=5)
print(random_integers)

# Perform random sampling from an array
array = np.array([1, 2, 3, 4, 5])
random_sample = np.random.choice(array, size=3, replace=False)
print(random_sample)

# Set a random seed for reproducibility
np.random.seed(42)
random_numbers = np.random.rand(3)
print(random_numbers)
```

The `numpy.random` module is a powerful tool for generating random data and simulating random processes in various applications, including statistics, machine learning, and scientific computing. Setting a seed with `np.random.seed()` is often used to ensure that the same sequence of random numbers is generated each time, making results reproducible.

---
## `.iloc[::]`

In Python, the `.iloc[]` method is used to select specific rows and columns from a Pandas DataFrame by their integer positions. It allows for integer-based indexing and slicing of the DataFrame.

**Method Syntax for Selection by Integer Position:**
```python
DataFrame.iloc[row_selection, column_selection]
```

- `row_selection` (optional): Specifies which rows to select based on integer positions. It can be a single integer, a list of integers, or a slice.
- `column_selection` (optional): Specifies which columns to select based on integer positions. It can be a single integer, a list of integers, or a slice.

**Examples:**

1. Select a Single Row by Integer Position:
   ```python
   single_row = df.iloc[2]  # Selects the third row (0-based indexing)
   ```

2. Select Specific Rows by Integer Position:
   ```python
   selected_rows = df.iloc[[0, 2, 4]]  # Selects the first, third, and fifth rows
   ```

3. Select a Single Column by Integer Position:
   ```python
   single_column = df.iloc[:, 1]  # Selects the second column (1st column with 0-based indexing)
   ```

4. Select Specific Columns by Integer Position:
   ```python
   selected_columns = df.iloc[:, [0, 2, 3]]  # Selects the first, third, and fourth columns
   ```

5. Select Specific Rows and Columns by Integer Position:
   ```python
   subset = df.iloc[[0, 2, 4], [1, 3]]  # Selects specific rows and columns
   ```

6. Use Slicing to Select Rows and Columns:
   ```python
   sliced_data = df.iloc[1:4, 2:5]  # Selects rows 1, 2, 3 and columns 2, 3, 4
   ```

7. Use '::' to select every \#'th row:
   ```python
   everyth_data = df.iloc[::25]  # Selects every 25th row, so 25, 50, 75 and so on.
   ```
The `.iloc[]` method is useful when you want to extract specific parts of a DataFrame based on their integer positions rather than their labels. It provides flexibility for subsetting and analyzing data within the DataFrame.

---
## `.plot()`

The `.plot()` method is a versatile function in Python's data visualization libraries, such as Matplotlib and Pandas, that is used to create various types of plots and charts to visualize data. The specific syntax and options may vary depending on the library used. Here, I'll provide a general overview of the `.plot()` method.

**Method Syntax (Pandas DataFrame):**
```python
DataFrame.plot(
    x=None,  # Label or position for the x-axis
    y=None,  # Label or position for the y-axis
    kind='line',  # Type of plot (e.g., 'line', 'bar', 'scatter', 'hist')
    ax=None,  # Matplotlib axis object to which the plot is added
    subplots=False,  # If True, create separate subplots for each column
    figsize=None,  # Size of the plot (width, height)
    title=None,  # Title of the plot
    legend=True,  # Display the legend
    grid=False,  # Display grid lines
    style=None,  # Plot style (e.g., 'k--' for black dashed line)
    colormap=None,  # Colormap for the plot
    ...  # Additional plot-specific options
)
```

**Method Syntax (Matplotlib):**
```python
import matplotlib.pyplot as plt

plt.plot(
    x_data,  # x-axis data
    y_data,  # y-axis data
    label=None,  # Label for the plot
    color=None,  # Line color
    linestyle=None,  # Line style (e.g., 'dashed', 'dotted')
    marker=None,  # Marker style (e.g., 'o' for circles)
    ...  # Additional plot-specific options
)
```

**Examples:**

1. **Line Plot (Pandas):**
   ```python
   import pandas as pd

   # Create a DataFrame
   data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 1, 3, 5]})

   # Create a line plot
   data.plot(x='x', y='y', kind='line', title='Line Plot')
   ```

2. **Scatter Plot (Pandas):**
   ```python
   import pandas as pd

   # Create a DataFrame
   data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 1, 3, 5]})

   # Create a scatter plot
   data.plot(x='x', y='y', kind='scatter', title='Scatter Plot')
   ```

3. **Line Plot (Matplotlib):**
   ```python
   import matplotlib.pyplot as plt

   x = [1, 2, 3, 4, 5]
   y = [2, 4, 1, 3, 5]

   # Create a line plot
   plt.plot(x, y, label='Line Plot', color='blue', linestyle='-', marker='o')
   plt.xlabel('X-Axis')
   plt.ylabel('Y-Axis')
   plt.title('Line Plot Example')
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

The `.plot()` method is a fundamental tool for creating various types of plots, including line plots, scatter plots, bar plots, histograms, and more. Depending on the library used (Pandas or Matplotlib), the specific options and syntax may vary, but the concept of using `.plot()` to visualize data remains consistent. You can customize plots by specifying attributes like labels, colors, markers, and titles to effectively communicate information from your data.

---
In Python, the `.reset_index()` method is commonly used with Pandas DataFrames to reset the index of the DataFrame. When data is manipulated or filtered in a DataFrame, the index may become disorganized or contain gaps. The `.reset_index()` method allows you to reorganize the index and, optionally, move the current index into a new column.

**Method Syntax:**
```python
DataFrame.reset_index(
    level=None,        # Specifies which levels of the index to reset (default: reset all)
    drop=False,        # If True, drops the index levels instead of converting them to columns
    inplace=False,     # If True, modifies the DataFrame in place and returns None
    col_level=0,       # For DataFrames with multi-level columns, specifies the column level to reset
    col_fill=''        # Value to use for filling missing entries when resetting columns
)
```

**Parameters:**
- `level` (optional): Specifies which levels of the index to reset. By default, it resets all levels. You can provide an integer, a label, or a list of integers/labels.
- `drop` (optional): If `True`, drops the index levels instead of converting them to columns. Default is `False`.
- `inplace` (optional): If `True`, modifies the DataFrame in place and returns `None`. Default is `False`.
- `col_level` (optional): For DataFrames with multi-level columns, specifies the column level to reset. Default is `0`.
- `col_fill` (optional): Value to use for filling missing entries when resetting columns. Default is an empty string.

**Examples:**

```python
import pandas as pd

# Sample DataFrame
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
})

# Set the 'Name' column as the index
data.set_index('Name', inplace=True)

# Reset the index
reset_data = data.reset_index()
print(reset_data)
```

In this example, we first set the 'Name' column as the index for the DataFrame using `set_index()`. Then, we use `.reset_index()` to reset the index, which moves the 'Name' column back as a regular column and assigns a default integer index.

The `.reset_index()` method is handy when you want to reorganize the index of a DataFrame, especially after filtering or performing operations that change the structure of the data. It helps maintain the consistency and integrity of the DataFrame's structure.

---
## `numpy.where()`

In NumPy, the `numpy.where()` function is used to return the indices of elements in an array that satisfy a specified condition. It is a versatile function that allows you to perform conditional operations on NumPy arrays and find the positions of elements that meet certain criteria.

**Function Syntax:**
```python
numpy.where(condition, x, y)
```

**Parameters:**
- `condition`: A boolean array or a condition expression that defines the condition for selection.
- `x`: Values to be used for elements that satisfy the condition.
- `y`: Values to be used for elements that do not satisfy the condition.

**Return Value:**
- An array of indices where the condition is `True`.

**Examples:**

1. Using a boolean condition:
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
condition = (arr > 3)

indices = np.where(condition)
print(indices)  # Output: (array([3, 4]),)
```

2. Using `x` and `y` parameters:
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
condition = (arr > 3)

result = np.where(condition, arr * 2, arr * 10)
print(result)  # Output: [10 20 30  8 10]
```

In the first example, we use `numpy.where()` with a boolean condition to find the indices where the elements in the array `arr` are greater than 3. The result is an array of indices where the condition is `True`.

In the second example, we use `numpy.where()` with the `x` and `y` parameters to conditionally modify elements in the array. If the condition is `True`, we double the element value; otherwise, we multiply it by 10.

The `numpy.where()` function is widely used for conditional operations and indexing in NumPy arrays. It allows you to perform element-wise selection and modification based on specified conditions, making it a valuable tool for data manipulation and filtering.

---
Certainly, let's continue with the original output style:

## `.cat.remove_unused_categories()`

In Pandas, the `.cat.remove_unused_categories()` method is used to remove unused categories from a categorical (or "cat") data type within a Pandas Series. Categorical data is a data type that represents categories or labels, and sometimes categories that are not present in the data may be retained. This method allows you to remove these unused categories, making the data more memory-efficient.

**Method Syntax:**
```python
Series.cat.remove_unused_categories()
```

**Example:**
```python
import pandas as pd

# Create a Pandas Series with categorical data
data = pd.Series(["A", "B", "A", "C", "B"], dtype="category")

# Remove unused categories
data.cat.remove_unused_categories(inplace=True)

print(data)
```

In this example:
- We create a Pandas Series called `data` with categorical data using the `dtype="category"` argument.
- Then, we use the `.cat.remove_unused_categories()` method to remove any categories that are not used in the `data` Series.
- By setting `inplace=True`, we modify the original Series `data` in place, and it will no longer contain unused categories.

This method is particularly useful when you are working with categorical data, and you want to optimize memory usage by removing categories that do not appear in the dataset. It helps keep the categorical data more memory-efficient and improves performance when performing operations on the data.

---
## `numpy.random.choice()`

In NumPy, the `numpy.random.choice()` function is used to generate random samples (randomly selected elements) from a given array or sequence. You can use this function to perform random sampling with or without replacement.

**Function Syntax:**
```python
numpy.random.choice(a, size=None, replace=True, p=None)
```

**Parameters:**
- `a`: The array-like object from which to sample. It can be an array, list, or similar data structure.
- `size` (optional): Determines the number of random samples to draw. It can be an integer or a tuple of integers specifying the shape of the output. If not provided, a single random sample is drawn.
- `replace` (optional): If `True` (default), sampling is done with replacement, meaning the same item can be selected multiple times. If `False`, sampling is done without replacement, ensuring that each item is selected at most once.
- `p` (optional): An array-like object representing the probabilities associated with each entry in the input array `a`. If provided, it specifies the probability distribution for selecting elements. The probabilities should sum to 1.

**Examples:**

1. Randomly sample elements from an array with replacement:
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Sample three elements with replacement
random_samples = np.random.choice(arr, size=3, replace=True)
print(random_samples)
```

2. Randomly sample elements from an array without replacement:
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Sample three elements without replacement
random_samples = np.random.choice(arr, size=3, replace=False)
print(random_samples)
```

3. Randomly sample elements with specified probabilities:
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Sample three elements with specified probabilities
probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]
random_samples = np.random.choice(arr, size=3, replace=True, p=probabilities)
print(random_samples)
```

The `numpy.random.choice()` function is useful for various tasks such as random sampling, creating random subsets of data, and generating random values based on specified probabilities. It is a versatile function that can be applied in a wide range of data analysis and simulation scenarios.

---
## `.std()`

In Python's Pandas library, the `.std()` method is used to calculate the standard deviation of numeric data in a Pandas Series or DataFrame. The standard deviation is a measure of the spread or variability of the data points in a dataset. It quantifies how much individual data points deviate from the mean (average) value.

**Method Syntax for Series:**
```python
Series.std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)
```

**Method Syntax for DataFrame:**
```python
DataFrame.std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)
```

**Parameters:**
- `axis` (optional): Specifies the axis along which the standard deviation is calculated. By default, it computes the standard deviation of the entire Series or DataFrame.
- `skipna` (optional): A boolean value that indicates whether to exclude missing (NaN) values when calculating the standard deviation. By default (`skipna=None`), it follows the global setting `pd.options.mode.use_inf_as_na`.
- `level` (optional): For DataFrames with hierarchical indexing, specifies the level on which to perform the standard deviation calculation.
- `ddof` (optional): Represents the "delta degrees of freedom," which adjusts the divisor in the formula for calculating the sample standard deviation. By default, it is set to 1, which computes the sample standard deviation. Use `ddof=0` for population standard deviation.
- `numeric_only` (optional): If `True`, only numeric data types are considered when calculating the standard deviation. Non-numeric data types are ignored.

**Example:**

```python
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5])

# Calculate the standard deviation
std_deviation = data.std()
print(std_deviation)
```

In this example, we have a Pandas Series `data` containing numeric values. We use the `.std()` method to calculate the standard deviation of the data. The result is a single numeric value representing the standard deviation of the Series.

The standard deviation is a valuable statistic in data analysis and is used to understand the spread or dispersion of data points. It provides insights into the variability of the data, with higher standard deviations indicating greater dispersion from the mean.

---
## `numpy.std()`

In Python's NumPy library, the `numpy.std()` function is used to calculate the standard deviation of numeric data in a NumPy array. The standard deviation is a measure of the spread or variability of the data points in a dataset. It quantifies how much individual data points deviate from the mean (average) value.

**Function Syntax:**
```python
numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)
```

**Parameters:**
- `a`: The input array containing the numeric data for which you want to calculate the standard deviation.
- `axis` (optional): Specifies the axis along which the standard deviation is calculated. By default (`axis=None`), it computes the standard deviation for the entire array.
- `dtype` (optional): Specifies the data type of the output. If not provided, the data type is inferred from the input data.
- `out` (optional): An alternative output array in which to place the result. If not provided, a new array is created.
- `ddof` (optional): Represents the "delta degrees of freedom," which adjusts the divisor in the formula for calculating the sample standard deviation. By default, it is set to 0, which computes the population standard deviation. Use `ddof=1` for the sample standard deviation.
- `keepdims` (optional): If `True`, the dimensions of the result are kept the same as the input data. If `False` (default), dimensions with size 1 are removed from the output.

**Example:**

```python
import numpy as np

data = np.array([1, 2, 3, 4, 5])

# Calculate the standard deviation
std_deviation = np.std(data)
print(std_deviation)
```

In this example, we have a NumPy array `data` containing numeric values. We use the `numpy.std()` function to calculate the standard deviation of the data. The result is a single numeric value representing the standard deviation of the array.

The standard deviation is a valuable statistic in data analysis and is used to understand the spread or dispersion of data points. It provides insights into the variability of the data, with higher standard deviations indicating greater dispersion from the mean.

---
## `numpy.quantile()`

In Python's NumPy library, the `numpy.quantile()` function is used to compute the quantile of a given dataset. A quantile is a statistical measure that divides a dataset into equal portions. The function allows you to find the value below which a specific percentage of the data falls, which is known as a quantile.

**Function Syntax:**
```python
numpy.quantile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
```

**Parameters:**
- `a`: The input array or sequence containing the dataset for which you want to compute the quantile.
- `q`: The quantile(s) to compute, represented as a float or an array of floats between 0 and 1. For a single quantile, you can pass a single float (e.g., 0.25 for the first quartile or 0.5 for the median). For multiple quantiles, pass an array of floats.
- `axis` (optional): Specifies the axis along which to compute the quantiles. By default (`axis=None`), it computes the quantiles for the entire array.
- `out` (optional): An alternative output array in which to place the result. If not provided, a new array is created.
- `overwrite_input` (optional): If `True`, the function may perform computations in place and modify the input array `a`. Default is `False`.
- `interpolation` (optional): Specifies the method to use when interpolating between data points to compute quantiles. Options include 'linear', 'lower', 'higher', 'midpoint', and others. Default is 'linear'.
- `keepdims` (optional): If `True`, the dimensions of the result are kept the same as the input data. If `False` (default), dimensions with size 1 are removed from the output.

**Examples:**

1. Compute the median (50th percentile) of an array:
```python
import numpy as np

data = np.array([1, 2, 3, 4, 5])

# Compute the median (50th percentile)
median = np.quantile(data, 0.5)
print(median)
```

2. Compute multiple quantiles (e.g., quartiles) of an array:
```python
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Compute quartiles (25th, 50th, and 75th percentiles)
quartiles = np.quantile(data, [0.25, 0.5, 0.75])
print(quartiles)
```

In these examples, we use the `numpy.quantile()` function to compute quantiles of the given datasets. The function returns the specified quantile(s) as output.

Quantiles are commonly used to understand the distribution of data, identify outliers, and summarize data in statistical analysis. You can specify different quantiles to obtain values such as quartiles (25th, 50th, and 75th percentiles) or any other desired quantile.