
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

---
## `proportions_ztest()`

In Python, the `proportions_ztest()` function is used to perform a Z-test for comparing proportions or testing the equality of proportions between two groups. This test is commonly used to assess whether there is a significant difference in the proportions of a specific event or outcome in two independent samples.

**Function Syntax:**
```python
scipy.stats.proportions_ztest(count, nobs, value=None, alternative='two-sided', prop_var=False)
```

**Parameters:**
- `count`: The number of successes (events of interest) in the samples.
- `nobs`: The total number of observations or trials in the samples.
- `value` (optional): The null hypothesis value for the proportion. If not provided, it is assumed to be 0.
- `alternative` (optional): The alternative hypothesis for the test. It can be 'two-sided' (default), 'larger', or 'smaller', indicating the direction of the test.
- `prop_var` (optional): If `True`, assumes that the variances of the proportions in the two samples are not equal. Default is `False`, assuming equal variances.

**Return Value:**
- A tuple containing the test statistic (Z-score) and the p-value for the Z-test.

**Example:**

```python
import scipy.stats as stats

# Sample data
successes1 = 45  # Number of successes in sample 1
nobs1 = 100      # Total number of observations in sample 1
successes2 = 60  # Number of successes in sample 2
nobs2 = 100      # Total number of observations in sample 2

# Perform a two-sample Z-test for proportions
z_statistic, p_value = stats.proportions_ztest([successes1, successes2], [nobs1, nobs2])

# Print the Z-score and p-value
print("Z-statistic:", z_statistic)
print("P-value:", p_value)
```

In this example, we use `scipy.stats.proportions_ztest()` to perform a two-sample Z-test for proportions. We compare the proportions of successes in two independent samples, `sample1` and `sample2`. The function returns a tuple containing the Z-statistic and p-value for the Z-test.

The `proportions_ztest()` function is commonly used in hypothesis testing scenarios to determine whether there is a statistically significant difference in proportions between two groups or samples. It helps assess whether an observed difference in proportions is likely to have occurred by chance.

## `statsmodels.stats.proportion.proportions_ztest()`

In Python's `statsmodels` library, the `proportions_ztest()` function is used to perform a Z-test for comparing proportions or testing the equality of proportions between two groups. This test is commonly used to assess whether there is a significant difference in the proportions of a specific event or outcome in two independent samples.

**Function Syntax:**
```python
statsmodels.stats.proportion.proportions_ztest(count, nobs, value=None, alternative='two-sided', prop_var=False)
```

**Parameters:**
- `count`: The number of successes (events of interest) in the samples.
- `nobs`: The total number of observations or trials in the samples.
- `value` (optional): The null hypothesis value for the proportion. If not provided, it is assumed to be 0.
- `alternative` (optional): The alternative hypothesis for the test. It can be 'two-sided' (default), 'larger', or 'smaller', indicating the direction of the test.
- `prop_var` (optional): If `True`, assumes that the variances of the proportions in the two samples are not equal. Default is `False`, assuming equal variances.

**Return Value:**
- A tuple containing the test statistic (Z-score) and the p-value for the Z-test.

**Example:**

```python
import statsmodels.api as sm
import numpy as np

# Sample data
successes1 = 45  # Number of successes in sample 1
nobs1 = 100      # Total number of observations in sample 1
successes2 = 60  # Number of successes in sample 2
nobs2 = 100      # Total number of observations in sample 2

# Perform a two-sample Z-test for proportions using statsmodels
z_statistic, p_value = sm.stats.proportions_ztest([successes1, successes2], [nobs1, nobs2])

# Print the Z-score and p-value
print("Z-statistic:", z_statistic)
print("P-value:", p_value)
```

In this example, we use `statsmodels.stats.proportion.proportions_ztest()` to perform a two-sample Z-test for proportions. We compare the proportions of successes in two independent samples, `sample1` and `sample2`. The function returns a tuple containing the Z-statistic and p-value for the Z-test.

The `proportions_ztest()` function in `statsmodels` is commonly used in hypothesis testing scenarios to determine whether there is a statistically significant difference in proportions between two groups or samples. It helps assess whether an observed difference in proportions is likely to have occurred by chance.

---
## `pingouin.chi2_independence()`

In Python, the `pingouin.chi2_independence()` function is used to perform a chi-squared test of independence for contingency tables. This test is used to determine whether there is a significant association between two categorical variables.

**Function Syntax:**
```python
pingouin.chi2_independence(data, x, y, correction=True, lambda_='log-likelihood', tail='two-sided')
```

**Parameters:**
- `data`: A Pandas DataFrame containing the data for the chi-squared test.
- `x`: The name of the first categorical variable (column name).
- `y`: The name of the second categorical variable (column name).
- `correction` (optional): If `True` (default), applies Yates' continuity correction for 2x2 contingency tables when appropriate.
- `lambda_` (optional): The method used to calculate the association measure. Options include 'log-likelihood' (default), 'cramer', 'phi', and others.
- `tail` (optional): The type of tail for the chi-squared test. It can be 'two-sided' (default), 'greater', or 'less', indicating the direction of the test.

**Return Value:**
- A Pandas DataFrame containing the chi-squared test result, including the chi-squared statistic, degrees of freedom, p-value, and association measure.

**Example:**

```python
import pingouin as pg
import pandas as pd

# Sample data in a Pandas DataFrame
data = pd.DataFrame({'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
                     'Preference': ['Yes', 'No', 'Yes', 'Yes', 'No']})

# Perform a chi-squared test of independence
result = pg.chi2_independence(data=data, x='Gender', y='Preference')

# Print the chi-squared test result
print(result)
```

In this example, we use `pingouin.chi2_independence()` to perform a chi-squared test of independence on a contingency table with categorical variables 'Gender' and 'Preference'. The function returns a Pandas DataFrame containing the chi-squared test result, including the chi-squared statistic, degrees of freedom, p-value, and association measure.

The `chi2_independence()` function in Pingouin is commonly used to assess the independence between two categorical variables and determine whether there is a statistically significant association between them.

---
## `plt.bar()`

The `plt.bar()` function is part of the Matplotlib library in Python, and it is used to create bar plots or bar charts to visualize data. Bar plots are commonly used to represent categorical data and display the values of different categories or groups.

**Function Syntax:**
```python
plt.bar(x, height, width=0.8, align='center', color=None, edgecolor='black', linewidth=1.0, tick_label=None, label=None)
```

**Parameters:**
- `x`: The x-coordinates or positions of the bars.
- `height`: The height or value of each bar.
- `width` (optional): The width of the bars. Default is 0.8.
- `align` (optional): The alignment of the bars with respect to their x-coordinates. Options include 'center' (default), 'edge', or 'align'.
- `color` (optional): The color of the bars. It can be a single color or a list of colors for individual bars.
- `edgecolor` (optional): The color of the edges of the bars. Default is 'black'.
- `linewidth` (optional): The width of the edges of the bars. Default is 1.0.
- `tick_label` (optional): Labels for the x-axis ticks.
- `label` (optional): A label for the bars, which can be used for creating a legend.

**Example:**
```python
import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]

# Create a bar plot
plt.bar(categories, values, width=0.6, color='blue', edgecolor='black', linewidth=1.2, label='Data')

# Add labels and a title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot Example')

# Add a legend
plt.legend()

# Show the plot
plt.show()
```

The `plt.bar()` function in Matplotlib is a versatile tool for creating bar plots to visualize categorical data. You can further customize the appearance of the plot by adjusting parameters to suit your specific needs.

---
## `scipy.stats.chisquare()`

In Python's SciPy library, the `scipy.stats.chisquare()` function is used to perform a chi-squared goodness-of-fit test. This test is used to determine whether the observed frequency distribution of categorical data fits a specified theoretical or expected distribution.

**Function Syntax:**
```python
scipy.stats.chisquare(f_obs, f_exp=None, ddof=0, axis=0)
```

**Parameters:**
- `f_obs`: The observed frequencies or values. This is typically an array or list.
- `f_exp` (optional): The expected frequencies or values. If not provided, it assumes a uniform distribution. Also, it can be an array or list.
- `ddof` (optional): The degree of freedom correction. Default is 0.
- `axis` (optional): The axis along which the chi-squared test is applied. Default is 0.

**Return Value:**
- A tuple containing the chi-squared test statistic and the p-value.

**Example:**
```python
import scipy.stats as stats
import numpy as np

# Observed frequencies
observed = np.array([18, 22, 16, 5, 19])

# Expected frequencies (a uniform distribution)
expected = np.array([16, 16, 16, 16, 16])

# Perform the chi-squared goodness-of-fit test
chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

# Print the test statistic and p-value
print("Chi-squared statistic:", chi2_stat)
print("P-value:", p_value)
```

In this example, we use `scipy.stats.chisquare()` to perform a chi-squared goodness-of-fit test. We provide the observed frequencies (`observed`) and expected frequencies (`expected`). The function returns a tuple containing the chi-squared test statistic and the p-value.

The chi-squared goodness-of-fit test is useful for determining whether the observed data follows a specified distribution. The p-value can be used to assess the goodness of fit, where a low p-value suggests that the observed and expected distributions significantly differ.

Please note that the `scipy.stats.chisquare()` function can be used for other chi-squared tests as well, such as the chi-squared test for independence in contingency tables, by appropriately providing the observed and expected frequencies.

---
## `scipy.stats.rankdata()`

In Python's SciPy library, the `scipy.stats.rankdata()` function is used to compute the ranks of elements in an array. It assigns a rank to each element based on their values, with ties handled according to the specified method.

**Function Syntax:**
```python
scipy.stats.rankdata(a, method='average')
```

**Parameters:**
- `a`: The input array for which ranks are computed.
- `method` (optional): The method used to assign ranks to tied elements. Options include 'average' (default), 'min', 'max', 'dense', and 'ordinal'.

**Return Value:**
- An array of ranks for the elements in the input array `a`.

**Example:**
```python
import scipy.stats as stats
import numpy as np

# Input array
data = np.array([3, 1, 2, 2, 4])

# Compute ranks with the 'average' method
ranks = stats.rankdata(data, method='average')

# Print the ranks
print(ranks)
# Output: [3. 1. 2.5 2.5 5.]
```

In this example, we use `scipy.stats.rankdata()` to compute the ranks of elements in the input array `data` using the 'average' method. The function returns an array of ranks, with tied elements assigned ranks based on the average of their positions.

The `scipy.stats.rankdata()` function is commonly used in various statistical analyses and non-parametric tests where ranking data is required. Different ranking methods can be chosen based on the specific requirements of the analysis.

---
## `pingouin.wilcoxon()`

In Python's Pingouin library, the `pingouin.wilcoxon()` function is used to perform a Wilcoxon signed-rank test, also known as the Wilcoxon matched-pairs signed-rank test. This non-parametric test is used to determine whether there is a significant difference between two related groups or conditions.

**Function Syntax:**
```python
pingouin.wilcoxon(x, y=None, paired=True, correction=False, alternative='two-sided')
```

**Parameters:**
- `x`: The data for the first group or condition.
- `y` (optional): The data for the second group or condition. If provided, the test is conducted for paired data (default). If not provided, a one-sample Wilcoxon test is performed.
- `paired` (optional): If `True` (default), the test is conducted for paired data. If `False`, a two-sample Wilcoxon test is performed.
- `correction` (optional): Whether to apply continuity correction for the two-sample test (default is `False`).
- `alternative` (optional): The alternative hypothesis for the two-sample test. Options include 'two-sided' (default), 'less', or 'greater'.

**Return Value:**
- A dictionary containing the test statistic, p-value, and the result of the test.

**Example:**
```python
import pingouin as pg
import numpy as np

# Sample data
group1 = np.array([14, 18, 20, 24, 25])
group2 = np.array([10, 16, 22, 18, 28])

# Perform a Wilcoxon signed-rank test (paired)
result = pg.wilcoxon(group1, group2)
print(result)
```

In this example, we use `pingouin.wilcoxon()` to perform a Wilcoxon signed-rank test on two related groups (`group1` and `group2`). The function returns a dictionary containing the test statistic, p-value, and the result of the test.

The Wilcoxon signed-rank test is useful for comparing two related groups or conditions, such as before and after measurements, when the data may not follow a normal distribution. It helps determine whether there is a significant difference between the groups.

---
Parametric and non-parametric tests are two broad categories of statistical tests, and they differ in their assumptions and practical applications. Here's a brief overview of the practical differences between these two types of tests:

1. **Assumptions**:
   - **Parametric Tests**: Parametric tests assume that the data follow a specific probability distribution, typically the normal (Gaussian) distribution. These tests make specific assumptions about the population parameters, such as the mean and variance. Common parametric tests include t-tests, ANOVA, and linear regression.

   - **Non-parametric Tests**: Non-parametric tests, also known as distribution-free tests, do not assume any particular probability distribution for the data. They are more flexible and can be used with data that do not meet the assumptions of parametric tests. Non-parametric tests include the Wilcoxon rank-sum test, Mann-Whitney U test, and Kruskal-Wallis test.

2. **Data Types**:
   - **Parametric Tests**: Parametric tests are generally suitable for continuous or interval data, as well as categorical data that can be transformed into a continuous format (e.g., by using dummy coding).

   - **Non-parametric Tests**: Non-parametric tests are more versatile and can be applied to various types of data, including ordinal data and non-normally distributed data.

3. **Sample Size**:
   - **Parametric Tests**: Parametric tests are often more robust and powerful when sample sizes are large. They tend to perform well when the data distribution approximates the assumed parametric distribution.

   - **Non-parametric Tests**: Non-parametric tests are often preferred when dealing with small sample sizes or when the data distribution significantly deviates from the assumed parametric distribution.

4. **Hypothesis Testing**:
   - **Parametric Tests**: Parametric tests are typically used for comparing means, variances, and regression relationships. They provide more precise parameter estimates and can be more powerful when the parametric assumptions are met.

   - **Non-parametric Tests**: Non-parametric tests are used for comparing medians, distributions, and non-linear relationships. They are less sensitive to extreme outliers and deviations from normality.

5. **Sensitivity**:
   - **Parametric Tests**: Parametric tests can be sensitive to violations of parametric assumptions. If the data do not meet the assumptions (e.g., normality, homoscedasticity), the results may be inaccurate.

   - **Non-parametric Tests**: Non-parametric tests are less sensitive to the underlying data distribution and are often considered more robust in the presence of outliers or non-normally distributed data.

In practical terms, the choice between parametric and non-parametric tests depends on several factors, including the nature of your data, the assumptions you can reasonably make, the research question you are addressing, and the specific statistical test you intend to perform. It's important to select the most appropriate test based on your data and research objectives to obtain reliable and meaningful results.

---
## `.pivot()`

In Pandas, the `.pivot()` method is used to reshape and transform data in a DataFrame. It allows you to convert long-format data (also known as "stacked" or "molten" data) into wide-format data (also known as "unstacked" or "pivoted" data). This method is particularly useful for data analysis and visualization when you want to reorganize your data.

**Method Syntax:**
```python
dataframe.pivot(index=None, columns=None, values=None)
```

**Parameters:**
- `index`: The column or columns to be used as the new index (rows) in the pivoted DataFrame.
- `columns`: The column or columns to be used as the new columns in the pivoted DataFrame.
- `values`: The column or columns to be used as the values to fill the cells of the pivoted DataFrame.

**Return Value:**
- A new DataFrame with the data reshaped according to the specified index, columns, and values.

**Example:**
```python
import pandas as pd

# Sample data in long format
data = {'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'Variable': ['A', 'B', 'A', 'B'],
        'Value': [10, 15, 12, 18]}

df = pd.DataFrame(data)

# Pivot the DataFrame
pivot_df = df.pivot(index='Date', columns='Variable', values='Value')

print(pivot_df)
```

In this example, we have a DataFrame `df` with long-format data containing dates, variables, and values. We use the `.pivot()` method to reshape the data into a wide-format DataFrame where the unique values in the 'Date' column become the index, the unique values in the 'Variable' column become the columns, and the 'Value' column becomes the cell values.

The resulting `pivot_df` DataFrame will look like this:

```
Variable      A   B
Date              
2023-01-01   10  15
2023-01-02   12  18
```

The `.pivot()` method is a powerful tool for data manipulation in Pandas, allowing you to efficiently transform data for analysis and visualization purposes.

---
## `pingouin.mwu()`

In Python's Pingouin library, the `pingouin.mwu()` function is used to perform a Mann-Whitney U test, also known as the Wilcoxon rank-sum test. This non-parametric test is used to determine whether there is a significant difference between two independent groups.

**Function Syntax:**
```python
pingouin.mwu(x, y, alternative='two-sided', method='auto')
```

**Parameters:**
- `x`: The data for the first group.
- `y`: The data for the second group.
- `alternative` (optional): The type of test to perform. Options include 'two-sided' (default), 'less', or 'greater'.
- `method` (optional): The method used to compute the U statistic. Options include 'auto' (default), 'exact', 'asymptotic', 'normal', or 'best'.

**Return Value:**
- A dictionary containing the test statistic, p-value, and the result of the test.

**Example:**
```python
import pingouin as pg
import numpy as np

# Sample data for two independent groups
group1 = np.array([22, 30, 40, 38, 25, 28, 32])
group2 = np.array([15, 20, 18, 27, 19, 24, 16])

# Perform a Mann-Whitney U test
result = pg.mwu(group1, group2, alternative='two-sided')
print(result)
```

In this corrected example, we use `pingouin.mwu()` to perform a Mann-Whitney U test on two independent groups (`group1` and `group2`) while specifying the `alternative` parameter for the type of test to perform. The function returns a dictionary containing the test statistic, p-value, and the result of the test.

Thank you for bringing this to my attention, and I appreciate your understanding.

---
## `pingouin.kruskal()`

In Python's Pingouin library, the `pingouin.kruskal()` function is used to perform a Kruskal-Wallis H test. This non-parametric test is used to determine whether there are statistically significant differences between the means of three or more independent groups.

**Function Syntax:**
```python
pingouin.kruskal(data, dv, between, detailed=False)
```

**Parameters:**
- `data`: The DataFrame containing the data.
- `dv`: The dependent variable (column name) that you want to analyze.
- `between`: The grouping variable (column name) that defines the groups for comparison.
- `detailed` (optional): If `True`, returns additional detailed statistics (default is `False`).

**Return Value:**
- A dictionary containing the test statistic (`H`), p-value (`p-unc`), and optionally, detailed statistics.

**Example:**
```python
import pingouin as pg
import pandas as pd

# Sample data in a DataFrame
data = pd.DataFrame({
    'Group': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Score': [15, 18, 12, 17, 20, 22]
})

# Perform a Kruskal-Wallis H test
result = pg.kruskal(data, dv='Score', between='Group')
print(result)
```

In this example, we use `pingouin.kruskal()` to perform a Kruskal-Wallis H test on three independent groups defined by the 'Group' column in the DataFrame. The function returns a dictionary containing the test statistic (`H`), p-value (`p-unc`), and additional detailed statistics if `detailed=True` is specified.

The Kruskal-Wallis H test is useful when you want to compare the means of multiple independent groups, and the data may not meet the assumptions of parametric tests. It helps determine whether there are statistically significant differences between the groups.

You can adjust the `detailed` parameter to obtain more or less detailed statistics based on your analysis needs.