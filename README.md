# Real Estate Multi-Regression Analysis

## Summary

**Goal:** Run multiple-regression analysis over scraped data in the Tokyo area to find a general formula that can be used for a real estate agent to recommend housing listings to clients.

**Method:** Applied multiple-regression analysis over 191509 scraped data by using Python 3.

**Conclusion:** Used general formulas to calculate housing listings that can meet the needs of a customer who is looking for housing in Tokyo, Japan. Queried dataset using SQL to filter listings to recommend specific rental options.

Here is the snippet of code we used to apply multiple-regression analysis (used Python3, NumPy, Pandas, and Matplotlib):

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm

pd.options.display.max_columns
pd.options.display.max_rows
df = pd.read_csv("suumo_tokyo_data.csv", sep=",")

x = df[['Building Age','Time[min]', 'Deposit', 'Square Meters']]
y = df[['Price']]

#run multiple regresion
sscaler = preprocessing.StandardScaler()
sscaler.fit(x)
xss_sk = sscaler.transform(x) 
sscaler.fit(y)
yss_sk = sscaler.transform(y)

#show summary plots
x_add_const = sm.add_constant(xss_sk)
model_sm = sm.OLS(yss_sk, x_add_const).fit()

results = model_sm.summary(yname = 'Price', xname=['Const', 'Building Age', 'Time to Station', 'Deposit', 'Square Meters'])
print(results)

#show correlation coefficient
sscaler = preprocessing.StandardScaler()
sscaler.fit(x)
xss_sk = sscaler.transform(x) 
sscaler.fit(y)
yss_sk = sscaler.transform(y)

print(xss_sk)
print(yss_sk)
```

The first result we obtained is the following:

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-13_at_22.50.13.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-13_at_22.50.13.png)

## Explanation of the Data

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.57.43.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.57.43.png)

### Variable Selection Process

By choosing a fewer variables from the data, we are able to eliminate bias and increase the efficiency of analysis.

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.24.56.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.24.56.png)

**Floor:** In general Tokyo area, it is not significant to consider which floor you are on

**Administrative Costs and Gratuity:** They directly affect to the dependent variable “Price”; therefore, create bias

**Number of Rooms:** By looking at coefficient, it is negatively correlated; therefore, it is not a reliable output

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.26.06.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.26.06.png)

## Evaluation of the Data

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.27.30.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.27.30.png)

**R-squared**: Determines either the result fits to the ideal model or not

Based on the result, the data has the R-squared of 87.5%

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.28.18.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.28.18.png)

**P-value**: If it is under the significant level (0.05), the prediction would be outside of the 95% interval.

Which means that this result is a reliable output.

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.28.47.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.28.47.png)

## Real World Application Example Ⅰ

Case: If the housing is 5-square meters bigger, how much does the price increase?

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.29.28.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.29.28.png)

Coefficient of Square Meters is 0.3604

Price = 0.3604 * 5 * 10000[yen] = 18020 yen

Therefore, if the housing is 5 square meters bigger, the price tend to increase by 18020 yen

## Real World Application Example Ⅱ

Case: The client is looking for a housing under 200000 yen but 60 square meters wider

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.29.28%201.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.29.28%201.png)

Predicted Price = 0.3604 * 60 * 10000[yen] = 216240 yen

over 200000 yen

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.30.33.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.30.33.png)

However, if it is 15 minutes away from the station,

15 * -0.1026 * 10000[yen] = -15390 yen

21624 - 1539 = 200850 ≈ 200000 yen

## Real World Application Example Ⅲ

Case: Find the cheapest but a bigger housing from the interval

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.31.13.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.31.13.png)

If the housing is 50 square meters, the average price is:

50 * 0.3604 * 10000[yen] = 180200 yen

However, according to lower limit 2.5%,

50 * 0.358 * 10000[yen] = 179000 yen

Therefore, we choose 179000 yen because it is 12000 yen cheaper and the top 2.5% cheapest.

Here is the housing information which matches the client’s needs.

1 out of 191,509 housing listings matches this criteria.

We used the following SQL code to filter the data.

```sql
**SELECT ***

**FROM tokyo**

**WHERE price <= 17900**

**AND sq_m >= 50;**
```

![Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.32.01.png](Multi-Regression%20Analysis%20Report%20324085edfb1645859d79327ea901bc0f/Screen_Shot_2020-04-14_at_00.32.01.png)