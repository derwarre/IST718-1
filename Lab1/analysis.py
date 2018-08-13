import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as statsform
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os
import sys
import statsmodels.stats.outliers_influence as statsoutliers
from scipy import stats


## Setting random seed before it's I forget
np.random.seed(10)

## Reading locally
#df = pd.read_csv("C:\\git\\IST718\\Lab1\\mencarelli_data.csv")

## Reading in the CSV from scrape.py
try:
    file = dir_path = os.path.dirname(os.path.realpath(__file__)) + "\\mencarelli_data.csv"
    print("Looking for: {}".format(file))
    df = pd.read_csv(file)
except:
    print("Can't find file. Use file from zip or run scrape.py first")
    sys.exit()


## Ties column is useless, dropping (there are none)
df.drop('ties',axis=1, inplace=True)

## The data set used for the salary data didn't have Syracuse's salary for the last 2 years
## A press release found that the 2016 season, Dino Babers was paid 2.4 million.
## Setting that value here so we can predict future numbers better
df.at[df.loc[df['school'] == 'Syracuse'].loc[df['season'] == 2016].index[0], 'salary'] = 2400000

## Assigning data types
df['year'] = pd.to_datetime(df['year'], format="%Y")
df['year'] = df['year'].dt.year
df['season'] = pd.to_datetime(df['season'], format="%Y")
df['season'] = df['season'].dt.year
df['salary'] = pd.to_numeric(df['salary'])
df['wins'] = pd.to_numeric(df['wins'])
df['losses'] = pd.to_numeric(df['losses'])
df['col_score'] = pd.to_numeric(df['col_score'])
df['GSR'] = pd.to_numeric(df['GSR'])
df['capacity'] = pd.to_numeric(df['capacity'])

## Creating a salary number based off of 1 million to reduce number of 0s
df['reduced_salary'] = df['salary']/1000000

## Cleaning up some discrepancies
### It seems that the site I scraped from went from CUSA to C-USA after 2015
### Also salary and cost of living scores could be 0, so if they're blank I'm setting them to numpy's nan
df['conference'].loc[df['conference'] == 'CUSA'] = "C-USA"
df['conference'].loc[df['conference'] == 'Pac-12'] = "PAC-12"
df['salary'].loc[df['salary'] == 0] = np.nan
df['col_score'].loc[df['col_score'] == 0] = np.nan
df['wl_ratio'] = round(df['wins']/(df['wins'] + df['losses']), 4)

## Creating df with no colscore for future model
df_no_colscore = df.copy()
df_no_colscore.drop(['col_score'], axis=1, inplace=True)
df_no_colscore.dropna(inplace=True)

## Dropping NA Values from the data frame
df_orig = df.copy()
df.dropna(inplace=True)

## Storing the NA rows before dropping them
## Sent to CSV for some info gathering in Excel for the report
df_nulls = df_orig[~df_orig.index.isin(df.index)]
dropped_schools = df_nulls['school'].unique()
#df_nulls.to_csv("dropped_schools.csv")


## Creating sliced dataframes with each season's data
df_2014 = df[df['season'] == 2014].copy()
df_2015 = df[df['season'] == 2015].copy()
df_2016 = df[df['season'] == 2016].copy()
df_2017 = df[df['season'] == 2017].copy()

## Creating a df with only the 95th percentile of salary by conference
df_nooutliers = df.copy()
conferences = (df['conference'].unique())
for conf in conferences:
    quant = df_nooutliers[df_nooutliers['conference'] == conf]['salary'].quantile(0.95)
    indexes = df.index[(df['salary'] > quant) & (df['conference'] == conf)]
    df_nooutliers.drop(indexes, inplace=True)


## Description Statistics
print('Original Data Frame')
print(df.describe())

print('Salary Outliers Dropped at 95th percentile')
print(df_nooutliers.describe())


## Setting Seaborn S# tyles
sns.set(style="whitegrid")
sns.set_palette((sns.color_palette("Set2")))
dims = (11.7, 8.27)

## Histogram of salaries
fig1, ax1 = plt.subplots(figsize=dims)
plot1 = sns.distplot(df['reduced_salary'], bins=11, hist=True, rug=True)
plt.axis([.1, 12, 0, .5])
fig1.suptitle('Distribution of Salaries (Including outliers)')
ax1.set_ylabel('Probability')
ax1.set_xlabel('Salary (Millions of US Dollars)')
ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
fig1s = plot1.get_figure()
fig1s.savefig('Histogram_Salary.png')

## Histogram of salaries 95th percentile only
fig4, ax4 = plt.subplots(figsize=dims)
plot4 = sns.distplot(df_nooutliers['reduced_salary'], bins=11, hist=True, rug=True)
plt.axis([.1, 9, 0, .7])
fig4.suptitle('Distribution of Salaries (Excluding outliers)')
ax4.set_ylabel('Probability')
ax4.set_xlabel('Salary (Millions of US Dollars)')
ax4.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
fig4s = plot4.get_figure()
fig4s.savefig('Histogram_Salary_95th.png')

## Box and whisker of salary by conference
fig2, ax2 = plt.subplots(figsize=dims)
plot2 = sns.boxplot(y='reduced_salary', x='conference', data=df, linewidth=1.25)
fig2.suptitle('Salary by Conference (Including outliers)')
ax2.set_ylabel('Salary (Millions of US Dollars)')
ax2.set_xlabel('Conference')
ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
fig2s = plot2.get_figure()
fig2s.savefig('BoxWhisker_Salary.png')

## Box and whisker of salary by conference 95th percentile only
fig5, ax5 = plt.subplots(figsize=dims)
plot5 = sns.boxplot(y='reduced_salary', x='conference', data=df_nooutliers, linewidth=1.25)
fig5.suptitle('Salary by Conference (Excluding outliers)')
ax5.set_ylabel('Salary (Millions of US Dollars)')
ax5.set_xlabel('Conference')
ax5.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
fig5s = plot5.get_figure()
fig5s.savefig('BoxWhisker_Salary_95th.png')

## Scatterplots by conference
plot3 = sns.lmplot(x='wl_ratio', y='reduced_salary', col='conference', data=df, col_wrap=3, sharex=False, sharey=False)
plot3.set_axis_labels('Win/Loss Ratio', 'Salary (Millions of US Dollars)')
plot3.set(xlim=(0,1), ylim=(0,12))
plot3.fig.subplots_adjust(wspace=.25, hspace=.25)
plot3.set_yticklabels(['2', '4', '6', '8', '10'])
for ax in plot3.axes.flatten():
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plot3.set_titles('Conference: {col_name}')
plot3.savefig('Scatterplot_Conference.png')

## Scatter plot of stadium capacity by salary
fig6, ax6 = plt.subplots(figsize=dims)
plot6 = sns.regplot(x='capacity', y='reduced_salary', data=df_2017, scatter=True)
plt.axis([-1000, 110000, -.1, 12])
fig6.suptitle('Salary by Stadium Capacity (2017 Season)')
ax6.set_ylabel('Salary (Millions of US Dollars)')
ax6.set_xlabel('Stadium Capacity')
ax6.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

fig6s = plot6.get_figure()
fig6s.savefig('Scatter_Stadium.png')


## Creating a copy of the df for the correlation matrix
## Dropping some derived columns as well
corr_df = df.copy()
corr_df.drop('reduced_salary', axis=1, inplace=True)
corr_df.drop('wl_ratio', axis=1, inplace=True)
corr_df.drop('year', axis=1, inplace=True)
corr_df.drop('season', axis=1, inplace=True)

## Creating the correlation df
correlation = corr_df.corr()

## Creating the mask so it's halved.
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

fig7, ax7 = plt.subplots(figsize=dims)
plot7 = sns.heatmap(data=correlation, mask=mask, vmax=1, cmap='Set2', square=True, linewidths=1, annot=True, cbar=False)
fig7.suptitle('Correlation Matrix')
fig7s = plot7.get_figure()
fig7s.savefig('Correlation_Matrix.png')


### Linear Regression Modelling
## Creating the test and train data on a 70/30 split
train_df, test_df = train_test_split(df, test_size=0.3)
train_df_nooutlines, test_df_nooutlines = train_test_split(df_nooutliers, test_size=0.3)


## Model 1
## Salary using conference, wl_ratio, col_score, GSR, capacity
test1_df = test_df.copy()
train1_df = train_df.copy()

model1 = str('salary ~ conference + wl_ratio + col_score + GSR + capacity')

train1_fit = statsform.ols(model1, data=train1_df).fit()
train1_df['predicted_salary'] = train1_fit.fittedvalues
test1_df['predicted_salary'] = train1_fit.predict(test1_df)

test_variance1 = round(np.power(test1_df['salary'].corr(test1_df['predicted_salary']),2),3)
print('Test Set Variance Accounted for: ', test_variance1)

fit1 = statsform.ols(model1, data = test1_df).fit()
print(fit1.summary())



## Model 2
## Salary using wl_ratio, col_score, GSR, capacity
test2_df = test_df.copy()
train2_df = train_df.copy()

model2 = str('salary ~ wl_ratio + col_score + GSR + capacity')

train2_fit = statsform.ols(model2, data=train2_df).fit()
train2_df['predicted_salary'] = train2_fit.fittedvalues
test2_df['predicted_salary'] = train2_fit.predict(test2_df)

test_variance2 = round(np.power(test2_df['salary'].corr(test2_df['predicted_salary']),2),3)
print('Test Set Variance Accounted for: ', test_variance2)

fit2 = statsform.ols(model2, data = test2_df).fit()
print(fit2.summary())


## Model 3
## Salary using conference, wl_ratio, GSR, col score
test3_df = test_df.copy()
train3_df = train_df.copy()

model3 = str('salary ~ conference + wins + GSR + capacity')

train3_fit = statsform.ols(model3, data=train3_df).fit()
train3_df['predicted_salary'] = train3_fit.fittedvalues
test3_df['predicted_salary'] = train3_fit.predict(test3_df)

test_variance3 = round(np.power(test3_df['salary'].corr(test3_df['predicted_salary']),2),3)
print('Test Set Variance Accounted for: ', test_variance3)

fit3 = statsform.ols(model3, data=train3_df).fit()
print(fit3.summary())


## Creating a DF copy and dropping derived and text variables.
df_vif = df.copy()
df_vif.drop('year', axis=1, inplace=True)
df_vif.drop('school', axis=1, inplace=True)
df_vif.drop('coach', axis=1, inplace=True)
df_vif.drop('conference', axis=1, inplace=True)
df_vif.drop('season', axis=1, inplace=True)
df_vif.drop('city_state', axis=1, inplace=True)
df_vif.drop('reduced_salary', axis=1, inplace=True)
df_vif.drop('losses', axis=1, inplace=True)
df_vif.drop('wins', axis=1, inplace=True)

## Found this on the internet ( https://stackoverflow.com/a/48826255/9761981 )
## It looks like it checks the data types and then iterates through checking the VIF value
## It drops the highest value based on the threshold (default 5)
def multicollinearity_check(X, thresh=5.0):
    data_type = X.dtypes
    # print(type(data_type))
    int_cols = \
    X.select_dtypes(include=['int', 'int16', 'int32', 'int64', 'float', 'float16', 'float32', 'float64']).shape[1]
    total_cols = X.shape[1]
    try:
        if int_cols != total_cols:
            raise Exception('All the columns should be integer or float, for multicollinearity test.')
        else:
            variables = list(range(X.shape[1]))
            dropped = True
            print('''\n\nThe VIF calculator will now iterate through the features and calculate their respective values.
            It shall continue dropping the highest VIF features until all the features have VIF less than the threshold of 5.\n\n''')
            while dropped:
                dropped = False
                vif = [statsoutliers.variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in variables]
                print('\n\nvif is: ', vif)
                maxloc = vif.index(max(vif))
                if max(vif) > thresh:
                    print('dropping \'' + X.iloc[:, variables].columns[maxloc] + '\' at index: ' + str(maxloc))
                    # del variables[maxloc]
                    X.drop(X.columns[variables[maxloc]], 1, inplace=True)
                    variables = list(range(X.shape[1]))
                    dropped = True

            print('\n\nRemaining variables:\n')
            print(X.columns[variables])
            # return X.iloc[:,variables]
            # return X
    except Exception as e:
        print('Error caught: ', e)

### Prints the display of the analysis
#multicollinearity_check(df_vif)


## Model 4
## Checking for mulit-collinearity, it looks like wl_ratio, salary, and col_score are
## the least correlated variables in the data set
## Rerunning with just those variables
test4_df = test_df.copy()
train4_df = train_df.copy()

model4 = str('salary ~ conference + wl_ratio + col_score')

train4_fit = statsform.ols(model4, data=train4_df).fit()
train4_df['predicted_salary'] = train4_fit.fittedvalues
test4_df['predicted_salary'] = train4_fit.predict(test4_df)

test_variance4 = round(np.power(test4_df['salary'].corr(test4_df['predicted_salary']),2),3)
print('Test Set Variance Accounted for: ', test_variance4)

fit4 = statsform.ols(model4, data=train4_df).fit()
print(fit4.summary())


## Model 5
## Was curious what would happen if I re-added capacity
test5_df = test_df.copy()
train5_df = train_df.copy()

model5 = str('salary ~ conference + wl_ratio + capacity')

train5_fit = statsform.ols(model5, data=train5_df).fit()
train5_df['predicted_salary'] = train5_fit.fittedvalues
test5_df['predicted_salary'] = train5_fit.predict(test5_df)

test_variance5 = round(np.power(test5_df['salary'].corr(test5_df['predicted_salary']),2),3)
print('Test Set Variance Accounted for: ', test_variance5)

fit5 = statsform.ols(model5, data=train5_df).fit()
print(fit5.summary())

## Model 6
## Model 5 with the data that removed outliers
test6_df = test_df_nooutlines.copy()
train6_df = train_df_nooutlines.copy()

model6 = str('salary ~ conference + wl_ratio + capacity')

train6_fit = statsform.ols(model6, data=train6_df).fit()
train6_df['predicted_salary'] = train6_fit.fittedvalues
test6_df['predicted_salary'] = train6_fit.predict(test6_df)

test_variance6 = round(np.power(test6_df['salary'].corr(test6_df['predicted_salary']),2),3)
print('Test Set Variance Accounted for: ', test_variance6)

## Originally was just using the training data for the set.
## Since this is the model that is ultimately used, expanding it to the whole data set
#fit6 = statsform.ols(model6, data=train6_df).fit()
fit6 = statsform.ols(model6, data=df_nooutliers).fit()
print(fit6.summary())

mae = np.sqrt(mean_absolute_error(test6_df['salary'], test6_df['predicted_salary']))
print('Mean Absolute Error: {}'.format(mae))

rms = np.sqrt(mean_squared_error(test6_df['salary'], test6_df['predicted_salary']))
print('Mean Squared Error: {}'.format(rms))

## Model 7
## Model 6 using WLS
test7_df = test_df_nooutlines.copy()
train7_df = train_df_nooutlines.copy()
w = np.ones(len(train7_df))
model7 = str('salary ~ conference + wl_ratio + capacity')

train7_fit = statsform.wls(model7, data=train7_df, weights=1./(w ** 2)).fit()
train7_df['predicted_salary'] = train7_fit.fittedvalues
test7_df['predicted_salary'] = train7_fit.predict(test7_df)

test_variance7 = round(np.power(test7_df['salary'].corr(test7_df['predicted_salary']),2),3)
print('Test Set Variance Accounted for: ', test_variance7)

fit7 = statsform.wls(model7, data=train7_df, weights=1./(w ** 2)).fit()
print(fit7.summary())

## Model 8
## Model 6 using GLS
test8_df = test_df_nooutlines.copy()
train8_df = train_df_nooutlines.copy()

model8 = str('salary ~ conference + wl_ratio + capacity')

train8_fit = statsform.gls(model8, data=train8_df).fit()
train8_df['predicted_salary'] = train8_fit.fittedvalues
test8_df['predicted_salary'] = train8_fit.predict(test8_df)

test_variance8 = round(np.power(test8_df['salary'].corr(test8_df['predicted_salary']),2),3)
print('Test Set Variance Accounted for: ', test_variance8)

fit8 = statsform.gls(model8, data=train8_df).fit()
print(fit8.summary())


## Setting some base variables so I can easily change my inputs from prediction to prediction
year = '2017'
school = 'Syracuse'
coach = 'Dino Babers'
conference = 'ACC'
wins = 4
losses = 8
wl_ratio = round(wins/(wins + losses), 4)
capacity = 49250
GSR = 82


## This forces Pandas to display the actual number vs the scientific notation
pd.options.display.float_format = '{:20,.2f}'.format

## Making a df copy with just Syracuse data
predict_df = df.loc[df['school'] == 'Syracuse']

## Using model 3, predicting the salaries
predict_df['predicted_salary'] = fit6.predict(predict_df)

## What if Syracuse was still in the Big East (Now known as the AAC?
conference = 'AAC'
predict_df.loc[len(predict_df)] = [year, school, coach, conference, 0, '', wins,
                                   losses, '', 0, capacity, GSR, 0,
                                   wl_ratio, 0]

## What if Syracuse was in the Big Ten?
conference = 'Big Ten'
predict_df.loc[len(predict_df)] = [year, school, coach, conference, 0, '', wins,
                                   losses, '', 0, capacity, GSR, 0,
                                   wl_ratio, 0]

## What if Syracuse was in the SEC?
conference = 'SEC'
predict_df.loc[len(predict_df)] = [year, school, coach, conference, 0, '', wins,
                                   losses, '', 0, capacity, GSR, 0,
                                   wl_ratio, 0]

## What if Syracuse had a fantastic season?
conference = 'ACC'
wins = 12
losses = 0
wl_ratio = round(wins/(wins + losses), 4)
predict_df.loc[len(predict_df)] = [year, school, coach, conference, 0, '', wins,
                                   losses, '', 0, capacity, GSR, 0,
                                   wl_ratio, 0]

## What if Syracuse got a brand new stadium?
wins = 4
losses = 8
capacity = 66045 ## This is the 75th percentile of capacity
wl_ratio = round(wins/(wins + losses), 4)
predict_df.loc[len(predict_df)] = [year, school, coach, conference, 0, '', wins,
                                   losses, '', 0, capacity, GSR, 0,
                                   wl_ratio, 0]
## Next season's prediction for Syracuse
capacity = 49250
year = 2018
wl_ratio = round(wins/(wins + losses), 4)
predict_df.loc[len(predict_df)] = [year, school, coach, conference, 0, '', wins,
                                   losses, '', 0, capacity, GSR, 0,
                                   wl_ratio, 0]
predict_df['predicted_salary'] = fit6.predict(predict_df)

print(predict_df)


## Prediction using the 2015 season data to 2016.
df_2015_nooutliers = df_nooutliers[df_nooutliers['season'] == 2015].copy()
df_2016_nooutliers = df_nooutliers[df_nooutliers['season'] == 2016].copy()
df_2015['predicted_salary'] = train6_fit.predict(df_2015)
df_2015_nooutliers['predicted_salary'] = train6_fit.predict(df_2015_nooutliers)

## Printing out the Syracuse row from the 2015 df and the 2016 df
salary_2016predicted = round(df_2015.at[df_2015.loc[df_2015['school'] == 'Syracuse'].index[0], 'predicted_salary'],2)
salary_2016actual = round(df_2016.at[df_2016.loc[df_2016['school'] == 'Syracuse'].index[0], 'salary'],2)
salary_2016_nooutliers_predicted = round(df_2015_nooutliers.at[df_2015_nooutliers.loc[df_2015_nooutliers['school'] == 'Syracuse'].index[0], 'predicted_salary'],2)

print("Predicted salary: {}".format(salary_2016predicted))
print("Predicted salary (With the dropped outliers): {}".format(salary_2016_nooutliers_predicted))
print("Actual salary: {}".format(salary_2016actual))
print("Difference (Pred-Actual): {}".format(round(salary_2016predicted-salary_2016actual,2)))
print("Difference (Pred(No Outliers)-Actual): {}".format(round(salary_2016_nooutliers_predicted-salary_2016actual,2)))

print("Mean difference between predicted 2016 salary and actual: {}".format(round(np.abs(df_2015['predicted_salary'].mean()-df_2016['salary'].mean())),2))


## Comparing salaries predicted using 2015's data to the actuals in 2016.
fig8, ax8 = plt.subplots(figsize=dims)
plot8 = sns.regplot(x='wl_ratio', y='salary', data=df_2016_nooutliers, label="Actual", scatter_kws={'alpha': 0.5})
plot8 = sns.regplot(x='wl_ratio', y='predicted_salary', data=df_2015_nooutliers, label="Predicted", scatter_kws={'alpha': 0.5})
fig8.suptitle('Actual vs Predicted Salary by Win/Lose Ratio (2016 Season)')
ax8.set_ylabel('Salary (US Dollars)')
ax8.set_xlabel('Win/Lose Ratio')
ax8.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plot8.legend()

fig8s = plot8.get_figure()
fig8s.savefig('Salary_Predicted.png')




