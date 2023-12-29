
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
Certainly, here's the updated information with the "alternative" parameter included:

## `pingouin.ttest()`

In Python, the `pingouin.ttest()` function is part of the Pingouin library, which is used for statistical analysis. Specifically, `pingouin.ttest()` is used for conducting independent two-sample t-tests to compare the means of two groups or conditions. This test is also known as the Student's t-test.

**Function Syntax:**
```python
pingouin.ttest(x, y, tail='two-sided', paired=False, equal_var=True, alternative='two-sided')
```

**Parameters:**
- `x`: The data for the first group or condition.
- `y`: The data for the second group or condition.
- `tail` (optional): The type of t-test to perform. It can be 'two-sided' (default), 'one-sided', or 'greater'. Use 'two-sided' for a two-tailed test, 'one-sided' for a one-tailed test, and 'greater' for a one-tailed test with greater-than comparison.
- `paired` (optional): If `True`, performs a paired t-test (related samples). If `False` (default), performs an independent t-test (unrelated samples).
- `equal_var` (optional): If `True` (default), assumes equal variances for both groups. If `False`, assumes unequal variances.
- `alternative` (optional): Specifies the alternative hypothesis for the test. It can be 'two-sided' (default), 'less', or 'greater', indicating whether you are testing for two-tailed or one-tailed significance.

**Return Value:**
- A Pandas DataFrame containing the t-statistic, degrees of freedom, p-value, and test result (e.g., 'significant' or 'not significant').

**Example:**

```python
import pingouin as pg
import pandas as pd

# Sample data for two groups
group1 = [75, 80, 85, 90, 95]
group2 = [65, 70, 75, 80, 85]

# Create a DataFrame
data = pd.DataFrame({'Group1': group1, 'Group2': group2})

# Perform an independent two-sample t-test with a one-tailed test
result = pg.ttest(data['Group1'], data['Group2'], alternative='greater')

# Print the result
print(result)
```

In this example, we use `pingouin.ttest()` to perform an independent two-sample t-test between two groups represented by `Group1` and `Group2`. The "alternative" parameter is set to 'greater', indicating a one-tailed test for the greater-than comparison. The function returns a DataFrame containing the t-statistic, degrees of freedom, p-value, and test result.

The `pingouin.ttest()` function is part of the Pingouin library, which provides a wide range of statistical tests and tools for data analysis in Python. It is particularly useful for conducting various statistical tests, including t-tests, ANOVA, correlation analysis, and more.

---
## `pingouin.anova()`

In Python, the `pingouin.anova()` function is used to perform an analysis of variance (ANOVA) to compare the means of two or more groups or conditions. ANOVA is a statistical test commonly used to determine whether there are significant differences between group means.

**Function Syntax:**
```python
pingouin.anova(data=None, dv=None, between=None, detailed=False, effsize='np2', correction=True)
```

**Parameters:**
- `data`: A Pandas DataFrame containing the data for the ANOVA.
- `dv`: The dependent variable (column name) to be analyzed.
- `between`: The independent variable(s) or factor(s) to compare. This can be a single factor or a list of factors.
- `detailed` (optional): If `True`, returns detailed ANOVA results including SS (sums of squares) and MS (mean squares). Default is `False`.
- `effsize` (optional): Specifies the effect size measure to calculate. Options include 'np2' (partial eta-squared, default), 'h2' (eta-squared), and 'omega2' (omega-squared).
- `correction` (optional): If `True` (default), performs Greenhouse-Geisser correction for violations of sphericity when appropriate.

**Return Value:**
- A Pandas DataFrame containing ANOVA results, including F-statistic, p-value, and effect size.

**Example:**

```python
import pingouin as pg

# Sample data in a Pandas DataFrame
data = pg.read_dataset('anova')

# Perform a one-way ANOVA
result = pg.anova(data=data, dv='Scores', between='Group')

# Print the ANOVA result
print(result)
```

In this example, we use `pingouin.anova()` to perform a one-way ANOVA on a dataset with dependent variable 'Scores' and independent variable 'Group'. The function returns a Pandas DataFrame containing ANOVA results, including the F-statistic, p-value, and effect size.

The `pingouin.anova()` function is part of the Pingouin library, which provides a wide range of statistical tests and tools for data analysis in Python. It is commonly used for conducting various statistical analyses, including ANOVA, t-tests, correlation analysis, and more.

---
## `pingouin.pairwise_tests()`

In Python, the `pingouin.pairwise_tests()` function is used to perform pairwise post hoc tests following an analysis of variance (ANOVA) or repeated measures ANOVA. Post hoc tests are used to compare specific groups or conditions after finding a significant effect in the initial ANOVA.

**Function Syntax:**
```python
pingouin.pairwise_tests(data=None, dv=None, between=None, parametric=True, p_adjust='bonf', coverage=0.95, effsize='cohen', tail='two-sided')
```

**Parameters:**
- `data`: A Pandas DataFrame containing the data for the ANOVA.
- `dv`: The dependent variable (column name) to be analyzed.
- `between`: The independent variable(s) or factor(s) for which post hoc tests are performed. This can be a single factor or a list of factors.
- `parametric` (optional): If `True` (default), performs parametric post hoc tests. If `False`, performs non-parametric tests.
- `p_adjust` (optional): Specifies the method for p-value adjustment. Options include 'bonf' (Bonferroni, default), 'sidak', 'holm', 'fdr_bh' (Benjamini-Hochberg false discovery rate), and others.
- `coverage` (optional): The coverage probability for confidence intervals. Default is `0.95`.
- `effsize` (optional): Specifies the effect size measure to calculate for post hoc tests. Options include 'cohen' (Cohen's d, default), 'hedges', 'r', and others.
- `tail` (optional): The type of tail(s) for the post hoc tests. It can be 'two-sided' (default), 'one-sided', or 'greater'. Use 'two-sided' for two-tailed tests and 'one-sided' for one-tailed tests.

**Return Value:**
- A Pandas DataFrame containing the results of the pairwise post hoc tests, including p-values, confidence intervals, and effect size measures.

**Example:**

```python
import pingouin as pg

# Sample data in a Pandas DataFrame
data = pg.read_dataset('anova')

# Perform a one-way ANOVA
result_anova = pg.anova(data=data, dv='Scores', between='Group')

# Perform pairwise post hoc tests
posthoc = pg.pairwise_tests(data=data, dv='Scores', between='Group')

# Print the post hoc test results
print(posthoc)
```

In this example, we first perform a one-way ANOVA using `pingouin.anova()`. Then, we use `pingouin.pairwise_tests()` to perform pairwise post hoc tests on the same dataset. The function returns a Pandas DataFrame containing the results of the pairwise post hoc tests, including p-values, confidence intervals, and effect size measures.

The `pingouin.pairwise_tests()` function is a useful tool for conducting post hoc tests after performing ANOVA or repeated measures ANOVA to identify specific group differences.