# Code Exercises from Hypothesis Testing in Python #

## Chapter 1

### --- Exercise 1 --- ###

# Print the late_shipments dataset
print(late_shipments.head())

# Print the late_shipments dataset
print(late_shipments)

# Calculate the proportion of late shipments
late_prop_samp = (late_shipments['late'] == 'Yes').mean()

# Print the results
print(late_prop_samp)


### --- Exercise 2 --- ###

# Hypothesize that the proportion is 6%
late_prop_hyp = 0.06

# Calculate the standard error
std_error = np.std(late_shipments_boot_distn, ddof=1)

# Find z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Print z_score
print(z_score)


### --- Exercise 3 --- ###

# Calculate the z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Calculate the p-value
p_value = 1 - norm.cdf(z_score, loc=0, scale=1)
                 
# Print the p-value
print(p_value) 


### --- Exercise 4 --- ###

# Calculate 95% confidence interval using quantile method
lower = np.quantile(late_shipments_boot_distn,0.025)
upper = np.quantile(late_shipments_boot_distn,0.975)

# Print the confidence interval
print((lower, upper))



## Chapter 2

### --- Exercise 1 --- ###

# Calculate the numerator of the test statistic
numerator = (xbar_yes - xbar_no)

# Calculate the denominator of the test statistic
denominator = np.sqrt(s_yes**2/n_yes + s_no**2/n_no)

# Calculate the test statistic
t_stat = numerator / denominator

# Print the test statistic
print(t_stat)


### --- Exercise 2 --- ###

# Calculate degrees of freedom
degrees_of_freedom = n_no + n_yes - 2

# Calculate the p-value using t-statistic and degrees of freedom
p_value = t.cdf(t_stat, df=degrees_of_freedom)

# Print the p_value
print(p_value)


### --- Exercise 3 --- ###

# Calculate the differences from 2012 to 2016
sample_dem_data['diff'] = sample_dem_data['dem_percent_12'] - sample_dem_data['dem_percent_16']

# Find the mean of the diff column
xbar_diff = sample_dem_data['diff'].mean()

# Find the standard deviation of the diff column
s_diff = sample_dem_data['diff'].std()

# Plot a histogram of diff with 20 bins
sample_dem_data['diff'].hist(bins=20)
plt.show()


### --- Exercise 4 --- ###

# Conduct a t-test on diff
test_results = pingouin.ttest(
                    x=sample_dem_data['diff'],
                    y=0,
                    alternative='two-sided'
                    )

                              
# Print the test results
print(test_results)


### --- Exercise 5 --- ###

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
paired_test_results = pingouin.ttest(
    x=sample_dem_data['dem_percent_12'],
    y=sample_dem_data['dem_percent_16'],
    alternative='two-sided'
    )

                              
# Print the paired test results
print(paired_test_results)


### --- Exercise 6 --- ###

# Calculate the mean pack_price for each shipment_mode
xbar_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].mean()

# Print the grouped means
print(s_pack_by_mode)

# Calculate the standard deviation of the pack_price for each shipment_mode
s_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].std()

# Print the grouped standard deviations
print(s_pack_by_mode)

# Boxplot of shipment_mode vs. pack_price
sns.boxplot(
    x='pack_price',
    y='shipment_mode',
    data=late_shipments
)
plt.show()


### --- Exercise 7 --- ###

# Run an ANOVA for pack_price across shipment_mode
anova_results = pingouin.anova(
    data=late_shipments,
    dv='pack_price',
    between='shipment_mode'
)



# Print anova_results
print(anova_results)


### --- Exercise 8 --- ###

# Perform a pairwise t-test on pack price, grouped by shipment mode
pairwise_results = pingouin.pairwise_tests(
    data=late_shipments,
    dv='pack_price',
    between='shipment_mode'
) 

# Print pairwise_results
print(pairwise_results)


### --- Exercise 9 --- ###

# Perform a pairwise t-test on pack price, grouped by shipment mode
pairwise_results = pingouin.pairwise_tests(
    data=late_shipments,
    dv='pack_price',
    between='shipment_mode',
    p_adjust='bonf'
) 

# Print pairwise_results
print(pairwise_results)



## Chapter 3


### --- Exercise 1 --- ###

# Hypothesize that the proportion of late shipments is 6%
p_0 = 0.06

# Calculate the sample proportion of late shipments
p_hat = (late_shipments['late'] == "Yes").mean()

# Calculate the sample size
n = len(late_shipments)

# Calculate the numerator and denominator of the test statistic
numerator = p_hat - p_0
denominator = np.sqrt(p_0 * (1 - p_0) / n)

# Calculate the test statistic
z_score = numerator / denominator

# Calculate the p-value from the z-score
p_value = 1 - norm.cdf(z_score)

# Print the p-value
print(p_value)


### --- Exercise 2 --- ###

# Calculate the pooled estimate of the population proportion
p_hat = (((ns['expensive']*p_hats[('expensive', 'Yes')])+(ns['reasonable']*p_hats[('reasonable', 'Yes')]))/(ns['expensive']+ns['reasonable']))

# Print the result
print(p_hat)

# Calculate p_hat one minus p_hat
p_hat_times_not_p_hat = p_hat * (1 - p_hat)

# Divide this by each of the sample sizes and then sum
p_hat_times_not_p_hat_over_ns = (p_hat_times_not_p_hat / ns['reasonable']) + (p_hat_times_not_p_hat / ns['expensive'])

# Calculate the standard error
std_error = np.sqrt(p_hat_times_not_p_hat_over_ns)

# Print the result
print(std_error)

# Calculate the z-score
z_score = ((p_hats['expensive']-p_hats['reasonable'])/std_error)

# Print z_score
print(z_score)

# Calculate the p-value from the z-score
p_value = 1 - norm.cdf(z_score)

# Print p_value
print(p_value)


### --- Exercise 3 --- ###

# Count the late column values for each freight_cost_group
late_by_freight_cost_group = late_shipments.groupby("freight_cost_group")['late'].value_counts()

# Create an array of the "Yes" counts for each freight_cost_group
success_counts = np.array([late_by_freight_cost_group[('expensive', 'Yes')], late_by_freight_cost_group[('reasonable', 'Yes')]])

# Create an array of the total number of rows in each freight_cost_group
n = np.array([sum(late_by_freight_cost_group['expensive']), sum(late_by_freight_cost_group['reasonable'])])

# Run a z-test on the two proportions
stat, p_value = proportions_ztest(success_counts, n, alternative='larger')


# Print the results
print(stat, p_value)



### --- Exercise 4 --- ###

# Proportion of freight_cost_group grouped by vendor_inco_term
props = late_shipments.groupby('vendor_inco_term')['freight_cost_group'].value_counts(normalize=True)

# Convert props to wide format
wide_props = props.unstack()

# Proportional stacked bar plot of freight_cost_group vs. vendor_inco_term
wide_props.plot(kind="bar", stacked=True)
plt.show()

# Determine if freight_cost_group and vendor_inco_term are independent
expected, observed, stats = pingouin.chi2_independence(late_shipments, 'freight_cost_group', 'vendor_inco_term')

# Print results
print(stats[stats['test'] == 'pearson']) 


### --- Exercise 5 --- ###

# Find the number of rows in late_shipments
n_total = len(late_shipments)

# Create n column that is prop column * n_total
hypothesized["n"] = hypothesized["prop"] * n_total

# Plot a red bar graph of n vs. vendor_inco_term for incoterm_counts
plt.bar(
   incoterm_counts['vendor_inco_term'],
   incoterm_counts['n'],
   color="red",
   label="Observed"
)

# Add a blue bar plot for the hypothesized counts
plt.bar(
   hypothesized['vendor_inco_term'],
   hypothesized['n'],
   label="Hypothesized",
   alpha=0.5,
   color='blue'
)

plt.legend()
plt.show()



### --- Exercise 6 --- ###

# Perform a goodness of fit test on the incoterm counts n
gof_test = chisquare(f_obs=incoterm_counts['n'], f_exp=hypothesized['n'])


# Print gof_test results
print(gof_test)



## Chapter 4

### --- Exercise 1 --- ###

# Count the freight_cost_group values
counts = late_shipments['freight_cost_group'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough
print((counts >= 30).all())

# Count the late values
counts = late_shipments['late'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough
print((counts >= 10).all())

# Count the values of freight_cost_group grouped by vendor_inco_term
counts = late_shipments.groupby('vendor_inco_term')['freight_cost_group'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough
print((counts >= 5).all())

# Count the shipment_mode values
counts = late_shipments['shipment_mode'].value_counts()

# Print the result
print(counts)

# Inspect whether the counts are big enough
print((counts >= 30).all())


### --- Exercise 2 --- ###

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
paired_test_results = pingouin.ttest(
                        x=sample_dem_data['dem_percent_12'],
                        y=sample_dem_data['dem_percent_16'],
                        alternative='two-sided',
                        paired=True
) 

# Print paired t-test results
print(paired_test_results)

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
wilcoxon_test_results = pingouin.wilcoxon(
                        x=sample_dem_data['dem_percent_12'],
                        y=sample_dem_data['dem_percent_16'],
                        alternative='two-sided'
) 

# Print paired t-test results
print(wilcoxon_test_results)


### --- Exercise 3 --- ###

# Select the weight_kilograms and late columns
weight_vs_late = late_shipments[['weight_kilograms','late']]

# Convert weight_vs_late into wide format
weight_vs_late_wide = weight_vs_late.pivot(columns='late', 
                                           values='weight_kilograms')


# Run a two-sided Wilcoxon-Mann-Whitney test on weight_kilograms vs. late
wmw_test = pingouin.mwu(
                        x=weight_vs_late_wide['Yes'],
                        y=weight_vs_late_wide['No'],
                        alternative='two-sided'
)



# Print the test results
print(wmw_test)


### --- Exercise 4 --- ###

# Run a Kruskal-Wallis test on weight_kilograms vs. shipment_mode
kw_test = pingouin.kruskal(
                        data=late_shipments,
                        dv='weight_kilograms',
                        between='shipment_mode'
)



# Print the results
print(kw_test)