

# 1 Introduction

## Goals

1. Develop time series plots for four Arkansas metro areas:
   1. Hot Springs, Little Rock, Fayetteville, Searcy
2. Develop a model for forecasting average median housing values for 2014
   1. Use the first quarter of 2014 as the test set
3. What three zip codes provide the best investment opportunity?

## Data

Data was provided by the professor contained Zillow median housing value by zip code between the years 1996 and 2018. Additional data was gathered for population data by zip code for the year 2010 and for tax return gross income by zip code for 2014.

| Source | Column Name | Comment |
| --- | --- | --- |
| Zillow | RegionID | Zillow region ID. Not used in this model |
| RegionName | ZIP code. Primary identifier for data values |
| City | Primary city for the zip code |
| State | State zip code resides in |
| Metro | Major metropolitan area this zip code is classified under |
| CountyName | Primary county for the zip code |
| SizeRank | Numerical rank for the zip code by median housing value |
| 1996-04 … 2018-06 | Median housing value by year and month for the zip code |
| US Census | Zip Code ZCTA | Zip code |
| 2010 Census Population | Population count as of the 2010 census |
| US Tax Return Data | State | State the zip code resides in |
| Zip Code | Zip code that the data is being reported on is from |
| N1 | The number of returns for the zip code |
| A02650 | The mean adjusted gross income for the zip code |

Table 1-1: Data Source Explanation

## Test / Train Datasets

Data for the years 2009 to 2013 were used as training data. Data from the year quarter 1 2014 was used as the testing data set for the model.

When running the RandomForest for alternate model approaches, a 30/70 split was used for validation and train data. Additional test data for Q1 of 2014 was used as well prior to prediction through to the end of 2014.


# 2 Arkansas Metro Area Analysis


The Little Rock and Fayetteville metro areas have a fairly regular distribution when charted. The Searcy is mostly normal with one irregular peak around the 90k price range. The Hot Springs metro area is completely bimodal with a lower peak of approximately 90k and a second peak near 140k.

![Arkansas Metro Area Analysis](https://github.com/mencarellic/IST718/blob/master/Lab2/images/AK_Metro_Dist.png)
Figure 2-1: Arkansas Distribution Plot by Metro Area

Each metro area consisted of 204 records with median house prices ranging from 62k to 146k. The largest spread was in the Hot Spring metro area with a range of just under 63k.

| Data Set | Record Count | Mean House Price | House Price Range |
| --- | --- | --- | --- |
| Hot Springs | 204 | 109,539 | 72,225 – 135,200 |
| Little Rock | 204 | 123,023 | 86,772 – 146,155 |
| Fayetteville | 204 | 107,202 | 79,572 – 132,754 |
| Searcy | 204 | 81,920 | 62,180 – 96,340 |
| All Metro Areas | 816 | 105,421 | 62,180 – 146,155 |

Table 2-1: Arkansas Metro Area Statistics

![Arkansas House Prices Over Time (2009-13)](https://github.com/mencarellic/IST718/blob/master/Lab2/images/AK_Metro_Line.png)
Figure 2-2: Arkansas House Prices Over Time (2009-13)

Plotting the change of prices over time reveals that the values peaked for the metro areas (except for Searcy) near the year 2007. Searcy peaked closed to 2010-11.

Given the population of Little Rock, 193,524 in the 2010 census, it&#39;s unsurprising that the range of house values is so wide. That spread may mean a good market to purchase a low-cost property and let it grow in value before selling for a profit. However, in terms of growth according to the US Census, Fayetteville was estimated to grow almost 8 times as much as Little Rock between 2010 and 2017 which given only the slight difference in house prices, would indicate that Fayetteville may be the best option of the four metro areas to purchase a piece of property with the intent to turn a profit.

# 3 Best Investment Opportunity

## Decision Criteria

Before beginning the model creation to find the best investment opportunity, certain criteria were selected to limit the risk for the organization.

- Low median cost of property
- High year over year growth

These two criteria were selected to allow for the best potential growth opportunity for the least amount of capital.

## Subsetting the Data

With more than 15,000 zip codes and over 20 years of home prices to analyze, the data had to be slimmed down before useful conclusions could be drawn. Since the test data was assigned to be the first quarter of 2014, it was decided to use data from the years 2009 through 2013 for the most recent information to test against.

Zip codes that were missing data in a month between 2009-01 and 2014-03 were dropped, this resulted in 1683 zip codes being removed. Additionally, 115 zip codes were missing from the tax return data set obtained from the IRS and 1 zip code was missing from the 2010 census data set. These zip codes were dropped from the forecast and model. A full list of dropped zip codes is included in in the file &quot;Dropped\_Zips.csv&quot;.

The data was aggregated to both the quarter and annual mean by zip code. The quarter period was chosen due to it fitting well with the test data being the first quarter of 2014. Annual was chosen to fill the assignment requirement and for easier data processing.

## Moving Average Model and Forecast

After running the autocorrelation and partial autocorrelation functions with the dataset, a simple moving average was picked as the first model. The autocorrelation indicated that a moving average with a lag of two would likely be the best option. The average was forecasted out through the end of 2014 on a quarterly basis.


The ACF plot to the left is for 60657. Several zip codes were tested, most showed two lags. This was the justification for using a two-period moving average

![Autocorrelation Plot for 60657 Zip Code](https://github.com/mencarellic/IST718/blob/master/Lab2/images/ACF_2009_2013_60657_by_year.png)
Figure 3-1: Autocorrelation Plot for 60657 Zip Code

The moving average proved to be relatively close considering the large range of values across the zip codes. The mean absolute error for the quarter 1 2014 predictions was 5863.53, while the mean absolute error for all predictions between 2009 and 2014 was 4129.22.

![Actual Prices Compared to Moving Average Prediction](https://github.com/mencarellic/IST718/blob/master/Lab2/images/HousePrice_OverTime_WithMA.png)
Figure 3-2: Actual Prices Compared to Moving Average Prediction

Using a minimum of 30% growth over the previous year and a maximum median house cost of $100k, the moving average model produced seven potential zip codes that could be good investment opportunities.


| Zip Code | Median Home Price | Growth from Last Year |
| --- | --- | --- |
| 35805 | 49,016.67 | 44.35% |
| 89107 | 96,983.33 | 38.27% |
| 89110 | 94,916.67 | 33.86% |
| 89115 | 84,975.00 | 33.57% |
| 89104 | 96,483.33 | 32.2% |
| 61723 | 80,475.00 | 30.75% |
| 48220 | 82,941.67 | 30.02% |

Table 3-1: Top Zips from Moving Average Model

![Top Zip Codes from Moving Average Model Over Time](https://github.com/mencarellic/IST718/blob/master/Lab2/images/MA_TopZipsOverTime.png)
Figure 3-3: Top Zip Codes from Moving Average Model Over Time

## Random Forest Model and Forecast

Several random forest models were run against the same time frame of data. A training set consisting of 188,328 (70%) of the records from 2009-2013 were used. The results were first tested against the remaining 30% of data and then against the quarter 1 2014 data. Little to no improvement of the model occurred past 100 iterations, so that value was used for the final version of the RF model. The RMSE for against the test set was 27,275 and 56,702 against the 2014 data. This is significantly higher than the moving average model.

When comparing the actual 2014 values to the Random Forest predicted values, the RF seems to consistently undershoot the actual value of the house prices. This can likely be attributed to having so few variables to key off for the model. Figure 3-4 depicts the actual values versus the predicted values with the black line indicating the position of any two values for a zip code being equal.

![Random Forest Prediction Results vs Actual Prices (2014)](https://github.com/mencarellic/IST718/blob/master/Lab2/images/HousePrices_ActualvsRF.png)
Figure 3-4: Random Forest Prediction Results vs Actual Prices (2014)

The Random Forest model suggested significantly more high growth zip codes while maintain a low median house cost. 29 were identified using the same criteria from the moving average model. Of those 29, the seven that were selected from the moving average model were present.

The chart presented in Table 3-2 and Figure 3-5 shows the top seven locations and their historical values from 2009 to 2013 with the 2014 predicted values attached.

| Zip Code | Median Home Price | Growth from Last Year |
| --- | --- | --- |
| 60411 | 90,076.62 | 58.61% |
| 85631 | 68,783.85 | 58.15% |
| 19134 | 37,847.40 | 48.32% |
| 39567 | 90,716.53 | 46.87% |
| 35805 | 49,016.67 | 44.35% |
| 46203 | 50,596.76 | 44.15% |
| 44102 | 35,059.00 | 43.73% |

Table 3-2: Top Zips from Random Forest Model

![Top Zip Codes from Random Forest Model Over Time](https://github.com/mencarellic/IST718/blob/master/Lab2/images/RF_TopZipsOverTime.png)
Figure 3-5: Top Zip Codes from Random Forest Model Over Time

# 4 Conclusions

After creating and analyzing both the moving average and random forest models the final recommendation is to purchase property in Huntsville, AL (35805); Atlanta, IL (61723) and Fernsdale, MI (48220) from the highest confidence of profit to lowest. Additional data for these three locations were captured from the City-Data website to provide further insight.

## Huntsville, AL – 35805

Huntsville was present on the list of high growth zip codes for both models. It ranked as the number one highest growth for the moving average model and was the fifth best performing zip code using the Random Forest model. With the low median house value this zip code would likely offer the best bang for the buck.

Located in the south-western area of Huntsville, this area has a low cost of living and a higher than average renting percentage which given its rate of growth would allow the organization to capture significant profit, if desired, prior to selling the property.

## Atlanta, IL – 61723

Atlanta, Illinois was 26th on the Random Forest forecast, but ranked 6th on the moving average model. The median home cost is much higher than the Huntsville zip code, but with an estimated 30% growth on both models, this cost may be recuperated quickly.

## Fernsdale, MI (48220)

Fernsdale was listed as 7th on the moving average list of top potential zip codes and 29th on the Random Forest. Again, the median cost of each property is higher than Huntsville, but not prohibitively so. It&#39;s estimated that approximately 42% of residents in this zip code rent their home, which opens the door to potentially holding the property for a steady source of income while property value increases over time. This zip code also has significantly more homes that were built in 1939 or earlier than either of the other two top potential choices which could lead to higher maintenance costs (35% in Fernsdale, MI vs 30% in Atlanta, IL and 7% in Huntsville, AL).

## Other Choices

The models found several other suitable areas that could be investigated further including several suburbs of Las Vegas, NV which appeared in both the Random Forest and the moving average models. These locations were not chosen due primarily to their high entry cost compared to the final choices.

# 5 Future Analysis

In future analysis, considering business properties and empty land would provide a more conclusive recommendation even if the purpose is still to purchase a house as an investment property. The prices and transactions with these additional property types could reveal a lot of an area&#39;s growth and overall value.

Additionally, while some tax data was used, only data for residents were used. Expanding this to various other taxes such as property tax would allow for a more complete picture.

# 6 References and Sources

https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2014-zip-code-data-soi

https://blog.splitwise.com/2013/09/18/the-2010-us-census-population-by-zip-code-totally-free/

https://www.census.gov/

http://files.zillowstatic.com/research/public/Zip/Zip\_Zhvi\_SingleFamilyResidence.csv

http://mech.at.ua/Forecasting.pdf

http://www.city-data.com/

