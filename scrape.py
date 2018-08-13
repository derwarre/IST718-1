from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
import requests

## Dates for data from archive.org
dates = ['20180718125608', '20170713111636', '20160716091733', '20150706182249' ]
## Listing the football seasons from the dates the data was gathered
seasons = [int(i[:4])-1 for i in dates]

## Spoofing headers so scraping looks less suspicious
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'}

## Quick function to strip out commas, dollar signs, and spaces out of the salary
## If the data wasn't in the table ("--" in all four years), it's replaced with N/A
def cleanData(data):
    badChars = ',$'
    if '--' in data:
        return None
    else:
        return data.translate(str.maketrans("", "", badChars)).strip()

## Create data frame and assign dtypes when needed
columns = ['year', 'school', 'coach', 'conference', 'salary', 'season', 'wins', 'losses', 'ties', 'city_state', 'col_score', 'capacity']
df = pd.DataFrame(columns=columns)


for date in dates:
    url = "https://web.archive.org/web/{}/http://sports.usatoday.com/ncaa/salaries".format(date)
    response = requests.get(url, headers=headers)
    scraped_html = BeautifulSoup(response.text, 'html.parser')

## Capturing the date for conditional and data capture
    year = date[:4]
    season = int(year)-1

## Resetting the variables for each year. Makes sure no data leaks between the requests
    coach = "N/A"
    conference = "N/A"
    school = "N/A"
    salary = 0
    wins = 0
    losses = 0
    ties = 0
    city_state = "N/A"
    col_score =0
    capacity = 0

## The 2017 and 2018 data is formatted differently than the 2015 and 2016 data.
## Using a simple conditional to handle them differently
    if "2018" in year or "2017" in year:
## Finding the start of the data table and removing the first and last rows (header and footer data)
        scrape = scraped_html.find('table', attrs={'class': 'datatable datatable-salaries fixed-column'}).find_all('tr')[1:-1]

        for row in scrape:
## The next four lines grab the school, conference, coach, and school paid salary
## I find the data based on the HTML tag and attribute/value pairs in the tag
## Alternatively I find the data based on it's position in the table
            school = cleanData(row.find('td', attrs={'data-position': 'school'}).a.text)
            conference = cleanData(row.find_all('td')[2].text)
            coach = cleanData(row.find('td', attrs={'data-position': 'coach'}).a.text)
            salary = cleanData(row.find_all('td')[4].text)

            df.loc[len(df)] = [year, school, coach, conference, salary, season, wins, losses, ties, city_state, col_score, capacity]

    elif "2016" in year or "2015" in year:
## Finding the start of the data table and removing the first row (header data)
        scrape = scraped_html.find('table', attrs={'class': 'sports-dynamic-table-scroll'}).find_all('tr')[1:]

        for row in scrape:
## These lines are similar to the ones for the 2017 and 2018 data
            school = cleanData(row.find('a', attrs={'data-position': 'school'}).text)
            conference = cleanData(row.find_all('td')[1].text)
            coach = cleanData(row.find('a', attrs={'data-position': 'coach'}).text)
            salary = cleanData(row.find_all('td')[3].text)

            df.loc[len(df)] = [year, school, coach, conference, salary, season, wins, losses, ties, city_state, col_score, capacity]

## Replacing nones with NaN using the numpy package
df.fillna(value=np.nan, inplace=True)

school_list = (df['school'].unique())
for schl in school_list:
## Some case specific items to enable proper URL creation
    if "(Ohio)" in schl:
        school = "Miami Oh"
    elif "(Fla.)" in schl:
        school = "Miami Fl"
    elif "Miami" in schl:
        school = "Miami Fl"
    elif "at Birmingham" in schl:
        school = "Alabama Birmingham"
    elif "A&M" in schl:
        school = "Texas AM"
    elif "Middle Tennessee" in schl:
        school = "Middle Tennessee State"
    elif "Bowling Green" in schl:
        school = "Bowling Green State"
    elif "LSU" in schl:
        school = "Louisiana State"
    else:
        school = schl
## Doing some quick cleaning of the school name for the next round of scraping
    school = school.lower().replace(" ", "-")

## Creating the URL and sending a request to the server
    url = "https://www.sports-reference.com/cfb/schools/" + school + "/"
    record_req = requests.get(url, headers=headers)

## Capture the HTML and find the first 5 TR tags in the first table tag
    record_scrape = BeautifulSoup(record_req.text, 'html.parser')
    record = record_scrape.find('table').tbody.find_all('tr', limit=5)
## For each TR found, scrape the year
## If the year found is in the season list then capture the season stats
## Insert those stats in the df
    for rec in record:
        scraped_year = rec.find('td', attrs={'data-stat': 'year_id'}).text

        if int(scraped_year) in seasons:
            if ((df['school'] == schl) & (df['season'] == int(scraped_year))).any():
                print(schl)
                wins = int(rec.find('td', attrs={'data-stat': 'wins'}).text)
                losses = int(rec.find('td', attrs={'data-stat': 'losses'}).text)
                ties = int(rec.find('td', attrs={'data-stat': 'ties'}).text)
                location = record_scrape.find('strong', text="Location:").next_sibling.strip()
                stadium_row = record_scrape.find('strong', text="Stadium:")
                if stadium_row is None:
                    capacity = 0
                else:
                    stadium = stadium_row.next_sibling.strip()
                    capacity = re.search('^.*\(.{5}(.*)\)', stadium).group(1).replace(',', '')

                rownum = df.loc[df['school'] == schl].loc[df['season'] == int(scraped_year)].index[0]
                df.set_value(rownum, 'wins', wins)
                df.set_value(rownum, 'losses', losses)
                df.set_value(rownum, 'ties', ties)
                df.set_value(rownum, 'city_state', location)
                df.set_value(rownum, 'capacity', capacity)

## Deduping the city_state list
## Similar to the season record stats. I iterate through a list of locations
## and scrape the COL score. With this site, the 404 page doesn't show as a 404
## so I had to check if the original tag that's being searched for shows up as a
## None type. If it was I set the score to 0 and continue processing
city_list = (df['city_state'].unique())
for city_state in city_list:
    city = city_state.split(', ')[0].replace(" ", "_").lower()
    state = city_state.split(', ')[1].replace(" ", "_").lower()

    url = "https://www.bestplaces.net/cost_of_living/city/" + state + "/" + city
    col_req = requests.get(url, headers=headers)

    col_scrape = BeautifulSoup(col_req.text, 'html.parser')
    col_findtable = col_scrape.find('table', attrs={'class': 'table table-striped'})
    if col_findtable is None:
        col_score = 0
    else:
        col_score = col_findtable.find_all('tr', limit=2)[1].find_all('td')[1].text
    rows = df.loc[df['city_state'] == city_state]['col_score'].index
    for index in rows:
        df.set_value(index, 'col_score', col_score)

## Including the GSR score from the NCAA
## This data was gathered from http://ncaa.s3.amazonaws.com/files/research/gsr-asr/2018RES_File5-DI-SquadAggregation_SA_Coh0710_v1_20180314.csv
## However, due to the nature of the data, values had to be manually assigned via dict and a for loop
## It was terrible. Given more time, I might be able to find a way.

df['GSR'] = 0
GSR = {'Air Force': 81, 'Akron': 66, 'Alabama at Birmingham': 75, 'Alabama': 84, 'Appalachian State': 74, 'Arizona State': 76,
       'Arizona': 74, 'Arkansas State': 76, 'Arkansas': 60, 'Army': 83, 'Auburn': 70, 'Ball State': 71, 'Baylor': 82,
       'Boise State': 86, 'Boston College': 90, 'Bowling Green': 82, 'Brigham Young': 52, 'Buffalo': 67, 'California': 68,
       'Central Florida': 92, 'Central Michigan': 72, 'Charlotte': 0, 'Cincinnati': 84, 'Clemson': 85, 'Coastal Carolina': 78,
       'Colorado State': 70, 'Colorado': 79, 'Connecticut': 80, 'Duke': 96, 'East Carolina': 68, 'Eastern Michigan': 67,
       'Florida Atlantic': 75, 'Florida International': 68, 'Florida State': 74, 'Florida': 74, 'Fresno State': 68,
       'Georgia Southern': 63, 'Georgia State': 33, 'Georgia Tech': 82, 'Georgia': 53, 'Hawaii': 80, 'Houston': 60,
       'Idaho': 55, 'Illinois': 77, 'Indiana': 84, 'Iowa State': 77, 'Iowa': 76, 'Kansas State': 77, 'Kansas': 79,
       'Kent State': 77, 'Kentucky': 73, 'Louisiana Tech': 67, 'Louisiana-Lafayette': 83, 'Louisiana-Monroe': 64,
       'Louisville': 76, 'LSU': 67, 'Marshall': 73, 'Maryland': 79, 'Massachusetts': 62, 'Memphis': 75, 'Miami (Fla.)': 88,
       'Miami (Ohio)': 85, 'Miami': 88, 'Michigan State': 72, 'Michigan': 82, 'Middle Tennessee State': 85, 'Middle Tennessee': 85,
       'Minnesota': 83, 'Mississippi State': 84, 'Mississippi': 68, 'Missouri': 85, 'Navy': 79, 'Nebraska': 85, 'Nevada': 75,
       'Nevada-Las Vegas': 56, 'New Mexico State': 67, 'New Mexico': 64, 'North Carolina State': 74, 'North Carolina': 66,
       'North Texas': 76, 'Northern Illinois': 84, 'Northwestern': 99, 'Notre Dame': 96, 'Ohio State': 69, 'Ohio': 41,
       'Oklahoma State': 51, 'Oklahoma': 72, 'Old Dominion': 70, 'Oregon State': 69, 'Oregon': 67, 'Penn State': 84,
       'Pittsburgh': 74, 'Purdue': 81, 'Rice': 91, 'Rutgers': 82, 'San Diego State': 76, 'San Jose State': 81,
       'South Alabama': 64, 'South Carolina': 98, 'South Florida': 73, 'Southern California': 73, 'Southern Methodist': 68,
       'Southern Mississippi': 75, 'Stanford': 96, 'Syracuse': 82, 'Temple': 83, 'Tennessee': 65, 'Texas A&M': 68,
       'Texas Christian': 73, 'Texas State': 70, 'Texas Tech': 70, 'Texas': 71, 'Texas-El Paso': 71, 'Texas-San Antonio': 82,
       'Toledo': 79, 'Troy': 68, 'Tulane': 84, 'Tulsa': 71, 'UCLA': 83, 'Utah State': 89, 'Utah': 83, 'Vanderbilt': 90,
       'Virginia Tech': 86, 'Virginia': 82, 'Wake Forest': 93, 'Washington State': 77, 'Washington': 81, 'West Virginia': 63,
       'Western Kentucky': 74, 'Western Michigan': 71, 'Wisconsin': 74, 'Wyoming': 78}

for key, value in GSR.items():
    rownum = df.loc[df['school'] == key].index
    for row in rownum:
        df.set_value(row, 'GSR', value)



## Exporting for use in the analysis
df.to_csv("mencarelli_data.csv", index=False)

print("Saved file to: {}".format(os.path.abspath("mencarelli_data.csv")))