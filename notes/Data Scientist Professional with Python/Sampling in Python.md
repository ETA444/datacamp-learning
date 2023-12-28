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
