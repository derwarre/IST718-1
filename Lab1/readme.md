

# 1.1 Introduction

## Goals:

1. Recommend the salary for the Syracuse football coach
2. Predict the coach&#39;s salary under the following circumstances:
  1. If Syracuse was still in the Big East conference
  2. If Syracuse was in the Big Ten conference
3. What schools were dropped from the data set and why?
4. What effect does graduation rate have on the projected salary?
5. How accurate is the model?
6. What is the single biggest impact on salary size?

## Data

A dataset was provided with the assignment containing coach salary data for a school year prior to 2013. Given the dataset&#39;s age, I elected to capture new data with the BeautifulSoup library.

Four years of data was gathered for the seasons between 2014 and 2017. This data was gathered from a college football records reference website, along with information such as the school&#39;s location and stadium capacity. This data was matched with the coach data for the years 2015 through 2018 gathered from the USA Today. The salary value used was the school pay only, which is the base salary specified in the contract. The combination of the different years used means that the last season&#39;s records are matched with the next season&#39;s salaries.

| Variable | Description | Comments |
| --- | --- | --- |
| Year | The year of the salary data |   |
| School | The school name |   |
| Coach | The coach&#39;s name |   |
| Conference | The athletic conference name |   |
| Salary | Salary in US Dollars for that year | Contract salary value, not including bonuses |
| Season | The season year the record numbers are relevant to |   |
| Wins | The number of wins during the season | Includes bowl games |
| Losses | The number of losses during the season | Includes bowl games |
| City\_State | The city and state of the school |   |
| COL\_Score | Cost of Living score normalized to the average across the United States |   |
| Capacity | Stadium capacity | 0 was used for schools that do not have a stadium |
| GSR | Graduation Success Rate calculate by the NCAA | This data wasn&#39;t available historically, so the same value was used across all seasons for a school |
| Reduced\_salary | Salary value divided by 1,000,000 for ease of displaying |   |
| WL\_Ratio | Number of wins / Total games in season |   |

Table 1: Data descriptions

59 rows were dropped from the data set for one of two reasons:

- No cost of living data was available for the location (32 rows)
  - Air Force, Arkansas State, Boston College, Georgia, Kentucky, Northern Illinois, Penn State, Vanderbilt
- No salary information was available for the year (27 rows)
  - 2017 (8)
    - Army, Baylor, Brigham Young, Miami (Fla.), Southern California, Syracuse, Temple, Tulane
  - 2016 (8)
    - Baylor, Brigham Young, Miami (Fla.), Pittsburgh, Southern California, Southern Methodist, Tulane, Tulsa
  - 2015 (6)
    - Brigham Young, Pittsburgh, Southern California, Southern Methodist, Tulsa, Wake Forest
  - 2014 (5)
    - Brigham Young, Southern California, Syracuse, Temple, Wake Forest

## Test / Train Datasets

The SKLearn library in Python has a method of creating test and train datasets with specified parameters. Using the library, I created a 30/70 split of data. With the final data set this equates to approximately 318 records used for training data and 137 records for testing.

## Outliers

The data set of the 2014 through 2017 seasons includes several coaches with extremely high salaries such as Nick Saban of the University of Alabama ($11.1 million USD in 2017) and Jim Harbaugh of the University of Michigan ($9 million USD is 2017). In an effort to create a model suitable for more mainstream coaches, a separate data set was created that contained only the 95th percentile by salary for each conference. This resulted in reducing the data set to 428 records.

# 2 Observations

While examining the data, it was found that the win/loss ratio and stadium capacity both had positive linear relationships with salary which fits with the expectations based on prior studies.

![Salary Scatterplot](https://github.com/mencarellic/IST718/blob/master/Lab1/Scatter_Stadium.png)

When looking at the salary by win/loss ratio and breaking it out by conference, the smaller conferences according to Google Trend data (C-USA, MAC, Sun Belt, Mt West) have a much tighter spread than the larger conferences such as the SEC, ACC, etc.

![Salary by Conference](https://github.com/mencarellic/IST718/blob/master/Lab1/Scatterplot_Conference.png)

The salary data overall is right skewed when comparing salary values. This is expected given the large spread of data between the minimum salary observed ($225K for Scott Satterfield in 2014) versus the highest salary ($11.1M for Nick Saban in 2017). After removing outliers such as Nick Saban&#39;s salary, the histogram normalized somewhat but is still very much a right skewed distribution.

![Distribution of Salary Including Outliers](https://github.com/mencarellic/IST718/blob/master/Lab1/Histogram_Salary.png)
![Distribution of Salary Excluding Outliers](https://github.com/mencarellic/IST718/blob/master/Lab1/Histogram_Salary_95th.png)

Putting aside the obvious correlation of wins and losses. The next strongest correlation is between a salary and stadium capacity. This holds true even when removing outliers, which indicates this isn&#39;t affected by skewed values. One of the next strongest correlations is between salary and number of wins, which follows logic. Notably, GSR and col\_score did not have very strong effects on any of the other variables besides themselves.

![Correlation Matrix](https://github.com/mencarellic/IST718/blob/master/Lab1/Correlation_Matrix.png)

# 3 Analysis

## Model Development

With the strong evidence that this is a linear relationship, the method used for the analysis was the ordinary least squares model. The generalized least squares and weighted least squares (with estimated weighting). After it was discovered that there could be some multi-collinearity in the data set, multiple variables were removed from the model.

| Predictor Variables | R-Squared | Notes |
| --- | --- | --- |
| conference, wl\_ratio, col\_score, GSR, capacity | 0.733 | First take at formula. |
| wl\_ratio, col\_score, GSR, capacity | 0.650 | Remove conference to determine the effect on the outcome |
| conference, wl\_ratio, col\_score | 0.731 | Removed GSR due to high P-Value (0.315) |
| conference, wl\_ratio, capacity | 0.751 | Removed col\_score due to high P-Value |
| conference, wl\_ratio, capacity | 0.789 | Calculated using the 95th percentile dataset |

Table 2: Predictor Variables and resulting R-Squared values

The final model used ended up being an OLS model using conference, wl\_ratio, and stadium capacity using only the data that remained after removing the 95th percentile of the salary data. This, of course, means it would be difficult to predict some salaries such as Saban&#39;s or Harbaugh&#39;s, but these would be the weaker predictions in the model regardless due to the large difference between them and the rest of the population.

Since multiple years of data was available, the opportunity presented itself to compare the actual predictive model. Using the 2015 data, the salaries for 2016 were predicted. The mean difference between the predicted salary for the 2016 season and the actual salary was $192K.

![Actual vs Predicted Salaries for 2016](https://github.com/mencarellic/IST718/blob/master/Lab1/Salary_Predicted.png)

## Syracuse University Specifics

The end goal of this exercise is to be able to provide a customer (a coach in this case) with a reasonable salary that is based on statistical measure. Using the model specified above, Dino Babers&#39; salary for the 2018 season should be somewhere near the range of $2.3M USD which is approximately $100K less than his current salary. The difference could be attributed to the fact that Babers was offered more than he was worth initially with the expectation that he would turn the program around from an underperforming one. When running the prediction with the last season being a perfect 12-0, Babers would be able to expect to see his salary raise to almost $3M.

From the perspective of the school, Syracuse has no reason to pay Babers any extraordinary amount above a normal adjustment since he has seen lackluster results since taking over the program for the 2016 season. Any aggressive contract negotiation by Babers should be conducted with caution.

Regarding if Syracuse was still in the Big East (Now the AAC), Babers would likely have a salary closer to $1.4M. If Syracuse were to move to the Big Ten he could expect his salary to raise only by approximately $100K.

GSR or the NCAA&#39;s graduation success rate had a minimal effect. On the final model, the model variance did not change at all and the R-Squared only changed by 0.02. It&#39;s worth noting that when investigating the possibility of multi-collinearity, GSR was the variable that had the highest variance inflation factor (13.91) indicating that it was strongly related to other measures and ended up being dropped from later models.

The final model had a mean absolute error of 644.29 and a root mean squared error of 556,434.47 which is a pretty wide error variance. This likely can be expected to be reduced using additional data and tuning the model further.

Out of the predictors used, conference was the biggest influencer followed by the last seasons win/loss ratio, and finally the stadium size. For example, a move to the SEC conference would see Babers&#39; salary possibly increase 34% to $3.2M.

# 4 Future Analysis and Planned Changes

In future iterations of this model and problem gathering additional predictors would be beneficial. Student population, coach&#39;s total years&#39; experience, coach&#39;s overall record, program endowments, ESPN program ranking to name a few.

Additionally, spending more time exploring the effects the variables on each other could assist in tuning the model.

# 5 References and Sources

Coach Salaries:
http://sports.usatoday.com/ncaa/salaries

Program records, school locations, stadium sizes:
https://www.sports-reference.com/cfb/

Cost of Living values for cities:
https://www.bestplaces.net/cost\_of\_living/city/

NCAA GSR Scores (2018):
http://ncaa.s3.amazonaws.com/files/research/gsr-asr/2018RES\_File5-DI-SquadAggregation\_SA\_Coh0710\_v1\_20180314.csv

2016-2017 Syracuse Coach Salary:
https://www.nunesmagician.com/2018/6/4/17422510/federal-tax-filings-show-syracuse-can-pay-competitive-football-coachs-salary-dino-babers-orange

Multi-collinearity check function:
https://stackoverflow.com/a/48826255/9761981

