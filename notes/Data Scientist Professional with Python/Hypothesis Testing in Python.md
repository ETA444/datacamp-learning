
Hypothesis tests are statistical methods used to make inferences and draw conclusions about populations based on sample data. Here's a brief and simple overview of common hypothesis tests and how to use them in Python:

1. **T-Tests**:
   - **Purpose**: T-tests are used to determine whether there is a significant difference between the means of two groups.
   - **Python Function**: You can perform t-tests using the `scipy.stats.ttest_ind` function for independent samples (two-sample t-test) or `scipy.stats.ttest_rel` for related samples (paired t-test).

   Example (two-sample t-test):
   ```python
   from scipy import stats
   group1 = [30, 35, 40, 45, 50]
   group2 = [25, 28, 32, 38, 42]
   t_stat, p_value = stats.ttest_ind(group1, group2)
   ```

2. **Proportion Tests**:
   - **Purpose**: Proportion tests are used to determine whether the proportion of a certain category in a sample is significantly different from a known or hypothesized proportion.
   - **Python Function**: You can perform proportion tests using `statsmodels` or other libraries.

   Example (proportion test using `statsmodels`):
   ```python
   import statsmodels.api as sm
   success_count = 45
   total_count = 100
   p_value = sm.stats.proportions_ztest(success_count, total_count, value=0.5)
   ```

3. **Chi-Square Tests**:
   - **Purpose**: Chi-square tests are used to assess the association or independence between categorical variables.
   - **Python Function**: You can perform chi-square tests using `scipy.stats.chi2_contingency` for contingency table analysis.

   Example (chi-square test):
   ```python
   from scipy import stats
   observed_data = [[10, 20], [15, 25]]
   chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed_data)
   ```

These are just a few common hypothesis tests. The choice of which test to use depends on the type of data and the research question you are trying to answer. Python libraries like `scipy` and `statsmodels` provide functions to conduct these tests, making it relatively straightforward to perform hypothesis testing in Python once you have collected your data.

---
## `norm.cdf()`

In Python, the `norm.cdf()` function is used to calculate the cumulative distribution function (CDF) of the standard normal distribution (also known as the Z-distribution) or a normal distribution with a specified mean and standard deviation. The CDF represents the probability that a random variable following a normal distribution is less than or equal to a given value.

The `norm.cdf()` function is typically used in the context of the SciPy library, which provides comprehensive functionality for scientific and technical computing, including statistical calculations.

**Function Syntax:**
```python
scipy.stats.norm.cdf(x, loc=0, scale=1)
```

**Parameters:**
- `x`: The value at which to evaluate the cumulative distribution function.
- `loc` (optional): The mean (average) of the normal distribution. Default is `0`.
- `scale` (optional): The standard deviation of the normal distribution. Default is `1`.

**Return Value:**
- The cumulative probability (CDF) value at the specified `x`.

**Example:**

```python
import scipy.stats as stats

# Calculate the CDF of the standard normal distribution at x=1.0
cdf_value = stats.norm.cdf(1.0)

print(cdf_value)
```

In this example, we use `scipy.stats.norm.cdf()` to calculate the cumulative distribution function (CDF) of the standard normal distribution at `x=1.0`. This function returns the probability that a randomly selected value from the standard normal distribution is less than or equal to `1.0`. The result is a probability value between 0 and 1.

You can also use `scipy.stats.norm.cdf()` to calculate the CDF for non-standard normal distributions by specifying the `loc` (mean) and `scale` (standard deviation) parameters to customize the distribution.

---
