#!/usr/bin/env python
# coding: utf-8

# # question 01
Q1. Calculate the 95% confidence interval for a sample of data with a mean of 50 and a standard deviation
of 5 using Python. Interpret the results.
# In[1]:


import scipy.stats as stats

# sample mean and standard deviation
sample_mean = 50
sample_std = 5

# sample size (assuming a large enough sample)
n = 100

# calculate the standard error of the mean
sem = sample_std / (n ** 0.5)

# calculate the 95% confidence interval
ci = stats.norm.interval(0.95, loc=sample_mean, scale=sem)

print("95% confidence interval:", ci)


# # question 02

# In[4]:


# observed_freq = [40, 30, 50, 20, 25, 35]
expected_freq = [0.2 * 200, 0.2 * 200, 0.2 * 200, 0.1 * 200, 0.1 * 200, 0.2 * 200]
import scipy.stats as stats

# observed and expected frequencies
observed_freq = [40, 30, 50, 20, 25, 35]
expected_freq = [0.2 * 200, 0.2 * 200, 0.2 * 200, 0.1 * 200, 0.1 * 200, 0.2 * 200]

# perform chi-square test
chi2, pval = stats.chisquare(observed_freq, expected_freq)

# print results
print("Chi-square statistic:", chi2)
print("P-value:", pval)


# # question 03

# In[3]:


import numpy as np
import scipy.stats as stats

# create contingency table
table = np.array([[20, 15], [10, 25], [15, 20]])

# perform chi-square test
chi2, pval, dof, expected = stats.chi2_contingency(table)

# print results
print("Chi-square statistic:", chi2)
print("P-value:", pval)


# # questtion 04

# In[5]:


import statsmodels.stats.proportion as proportion

# sample size and number of successes
n = 500
successes = 60

# calculate confidence interval
conf_interval = proportion.proportion_confint(successes, n, alpha=0.05)

# print results
print("95% Confidence Interval:", conf_interval)


# # question 05
Calculate the 90% confidence interval for a sample of data with a mean of 75 and a standard deviation
of 12 using Python. Interpret the results.
# In[6]:


import scipy.stats as stats

# sample size
n = 1

# sample mean and standard deviation
mean = 75
std = 12

# calculate confidence interval
conf_interval = stats.t.interval(0.90, df=n-1, loc=mean, scale=std)

# print results
print("90% Confidence Interval:", conf_interval)


# # question 06
Use Python to plot the chi-square distribution with 10 degrees of freedom. Label the axes and shade the
area corresponding to a chi-square statistic of 15.
# In[7]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# degrees of freedom
df = 10

# create x values
x = np.linspace(0, 30, 500)

# calculate chi-square distribution for degrees of freedom
chi2 = stats.chi2.pdf(x, df)

# create plot
fig, ax = plt.subplots()
ax.plot(x, chi2, 'b-', lw=2, alpha=0.6, label='chi2 pdf')

# shade area corresponding to chi-square statistic of 15
x_shade = np.linspace(15, 30, 100)
y_shade = stats.chi2.pdf(x_shade, df)
ax.fill_between(x_shade, y_shade, color='gray', alpha=0.4)

# add labels to axes
ax.set_xlabel('Chi-square statistic')
ax.set_ylabel('Probability density function')
ax.set_title('Chi-square distribution with 10 degrees of freedom')

# show plot
plt.show()


# # question 07
A random sample of 1000 people was asked if they preferred Coke or Pepsi. Of the sample, 520
preferred Coke. Calculate a 99% confidence interval for the true proportion of people in the population who
prefer Coke.CI = p ± z*sqrt((p*(1-p))/n)
import math

# sample size and proportion
n = 1000
p = 0.52

# z-score for 99% confidence interval
z = 2.576

# calculate standard error and margin of error
se = math.sqrt((p*(1-p))/n)
me = z*se

# calculate confidence interval
lower = p - me
upper = p + me

# print results
print("99% Confidence Interval: ({:.4f}, {:.4f})".format(lower, upper))

# # question 08
A researcher hypothesizes that a coin is biased towards tails. They flip the coin 100 times and observe
45 tails. Conduct a chi-square goodness of fit test to determine if the observed frequencies match the
expected frequencies of a fair coin. Use a significance level of 0.05.
# In[10]:



import numpy as np
from scipy.stats import chisquare

# observed frequencies
observed_freq = np.array([55, 45])

# expected frequencies
expected_freq = np.array([50, 50])

# conduct chi-square test
chi2, p_value = chisquare(observed_freq, expected_freq)

# print results
print("Chi-square test statistic: {:.4f}".format(chi2))
print("p-value: {:.4f}".format(p_value))


# # question 09
A study was conducted to determine if there is an association between smoking status (smoker or
non-smoker) and lung cancer diagnosis (yes or no). The results are shown in the contingency table below.
Conduct a chi-square test for independence to determine if there is a significant association between
smoking status and lung cancer diagnosis.

Use a significance level of 0.05.
Group A

Outcome 1 20 15
Outcome 2 10 25
Outcome 3 15 20
Group B

Lung Cancer: Yes

Smoker 60 140
Non-smoker 30 170
# In[11]:


import numpy as np
from scipy.stats import chi2_contingency

# observed frequencies
observed_freq = np.array([[20, 15], [10, 25], [15, 20], [60, 140], [30, 170]])

# conduct chi-square test
chi2, p_value, dof, expected_freq = chi2_contingency(observed_freq)

# print results
print("Chi-square test statistic: {:.4f}".format(chi2))
print("p-value: {:.4f}".format(p_value))
print("Degrees of freedom: {}".format(dof))
print("Expected frequencies:\n", expected_freq)


# # question 10
Q10. A study was conducted to determine if the proportion of people who prefer milk chocolate, dark
chocolate, or white chocolate is different in the U.S. versus the U.K. A random sample of 500 people from
the U.S. and a random sample of 500 people from the U.K. were surveyed. The results are shown in the
contingency table below. Conduct a chi-square test for independence to determine if there is a significant
association between chocolate preference and country of origin.U.S. (n=500) 200 150 150
U.K. (n=500) 225 175 100Step 1: State the null and alternative hypotheses

The null hypothesis (H0) is that there is no association between chocolate preference and country of origin.

The alternative hypothesis (HA) is that there is an association between chocolate preference and country of origin.

Step 2: Set the significance level

The significance level (alpha) is set at 0.05.

Step 3: Calculate the expected frequencies

We can calculate the expected frequencies assuming that there is no association between chocolate preference and country of origin using the formula:

Expected frequency = (row total x column total) / grand total

The expected frequencies are shown in the table below:

Milk Chocolate	Dark Chocolate	White Chocolate	Row Total
U.S.	187.5	187.5	125	500
U.K.	237.5	237.5	25	500
Column Total	425	425	150	1000
Grand Total				1000
Step 4: Calculate the chi-square test statistic

The chi-square test statistic can be calculated using the formula:

chi-square = sum((observed frequency - expected frequency)^2 / expected frequency)

The degrees of freedom for the test are calculated using the formula:

df = (number of rows - 1) x (number of columns - 1)

Plugging in the values, we get:

chi-square = 74.57
df = 2

Using Python, we can find the p-value associated with this test statistic:
# In[12]:


import scipy.stats as stats

chi2_statistic = 74.57
df = 2
p_value = 1 - stats.chi2.cdf(chi2_statistic, df)
print("p-value:", p_value)

Step 5: Interpret the results

The chi-square test for independence indicates that there is a significant association between chocolate preference and country of origin (χ2(2) = 74.57, p < 0.05). We can see that the proportion of people who prefer white chocolate is different between the two countries, with a higher proportion in the U.S. (25%) compared to the U.K. (5%).
# # question 11
Q11. A random sample of 30 people was selected from a population with an unknown mean and standard
deviation. The sample mean was found to be 72 and the sample standard deviation was found to be 10.
Conduct a hypothesis test to determine if the population mean is significantly different from 70. Use a
significance level of 0.05.
We can conduct a one-sample t-test to test the hypothesis:

Null hypothesis: the population mean is equal to 70 (µ = 70)

Alternative hypothesis: the population mean is not equal to 70 (µ ≠ 70)

We can use a t-test because the population standard deviation is unknown and the sample size is less than 30.

The test statistic is calculated as:

t = (x̄ - µ) / (s / sqrt(n))

where x̄ is the sample mean, µ is the hypothesized population mean, s is the sample standard deviation, and n is the sample size.

Substituting the values, we get:

t = (72 - 70) / (10 / sqrt(30)) = 1.095

Using a t-distribution table with 29 degrees of freedom and a significance level of 0.05 (two-tailed), the critical values are ±2.045.

Since the calculated t-value of 1.095 falls within the acceptance region (-2.045 < t < 2.045), we fail to reject the null hypothesis.

Therefore, we do not have sufficient evidence to conclude that the population mean is significantly different from 70 at the 0.05 level of significance.
# In[ ]:




