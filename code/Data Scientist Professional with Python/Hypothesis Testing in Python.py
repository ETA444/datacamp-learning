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





### --- Exercise 2 --- ###





### --- Exercise 3 --- ###





### --- Exercise 4 --- ###





### --- Exercise 5 --- ###





### --- Exercise 6 --- ###





### --- Exercise 7 --- ###





### --- Exercise 8 --- ###




### --- Exercise 9 --- ###




### --- Exercise 10 --- ###





## Chapter 4

### --- Exercise 1 --- ###




### --- Exercise 2 --- ###




### --- Exercise 3 --- ###




### --- Exercise 4 --- ###




### --- Exercise 5 --- ###




### --- Exercise 6 --- ###




### --- Exercise 7 --- ###




### --- Exercise 8 --- ###




### --- Exercise 9 --- ###




### --- Exercise 10 --- ###




