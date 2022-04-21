#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing modules and packages

import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup


# ### Web Scraping English Premier League (EPL) Data

# In[2]:


url="https://www.football-data.co.uk/englandm.php"

# Make a GET request to fetch the raw HTML content
html_content = requests.get(url).text

# Parse the html content
soup = BeautifulSoup(html_content, "lxml")
#print(soup.prettify()) # print the parsed data of html


# In[3]:


import re

data = []
# unwanted = ['Season 1993/1994','Season 1994/1995','Season 1995/1996','Season 1996/1997','Season 1997/1998',
#             'Season 1998/1999','Season 1999/2000','Season 2000/2001','Season 2001/2002']

for link in soup.find_all('a', href=True, text=re.compile('Premier League')):
    data.append("https://www.football-data.co.uk/"+link.get("href"))

data = data[1:20]
print(data)

len(data)


# In[4]:


frames = []

for i in range(len(data)):
    globals()[f'df{i}'] = pd.read_csv(data[i], encoding="latin-1", on_bad_lines='skip')
    df_object = globals()[f'df{i}']
    frames.append(df_object)


# In[5]:


df = pd.DataFrame()

for d in frames:
    df =  df.append(d, ignore_index=True)


# In[6]:


df


# ### Data cleaning and subsetting required data
# Remove betting data as we don't need it for prediction.

# In[7]:


#we only require the first 22 columns for the purpose of prediction

df_new = df.iloc[:,:23]
df_new = df_new.drop(['Div','Date','Time'],axis=1)

table_features = df.iloc[:,:7]
table_features = table_features.drop(['FTHG','FTAG','Div','Date'],axis=1)


# In[8]:


df_new.head()


# In[9]:


df_new.shape[0]


# In[11]:


avg_home_scored = df_new.FTHG.sum()*1.0 / df_new.shape[0]
avg_away_scored = df_new.FTAG.sum()*1.0 / df_new.shape[0]

avg_home_conceded = avg_away_scored
avg_away_conceded = avg_home_scored

print("Average number of goals at home =",avg_home_scored)
print("Average number of goals away =", avg_away_scored)
print("Average number of goals conceded at home =",avg_home_conceded)
print("Average number of goals conceded away",avg_away_conceded)


# In[12]:


result_home = df_new.groupby(['HomeTeam'])
result_away = df_new.groupby('AwayTeam')


# Goals Scored -> For Home :
# Average Goal Scored = Full-Time Goal Scored (home) / Total Home Games Played | 
# For Away :
# Average Goal Scored = Full-Time Goal Scored (away) / Total Away Games Played
# 
# Goals Conceded -> For Home :
# Average Goal Conceded = Full-Time Goal Conceded (home) / Total Home Games Played | 
# For Away :
# Average Goal Conceded = Full-Time Goal Conceded (away) / Total Away Games Played

# In[40]:


# table['Team']= result_home['HomeTeam'].all().values

table = pd.DataFrame()

table['HGS'] = result_home['FTHG'].sum()
table['HGC'] = result_home['FTAG'].sum()
table['AGS'] = result_away['FTAG'].sum()
table['AGC'] = result_away['FTHG'].sum()

# table = pd.concat([HGS, HGC, AGS, AGC],axis='columns',sort=False)
table.head()


# In[41]:


table.reset_index(inplace=True)
table.head()


# In[42]:


#renaming the columns

table.columns.values[0] = "Team"
# table.columns.values[1] = "HGS"
# table.columns.values[2] = "HGC"
# table.columns.values[3] = "AGS"
# table.columns.values[4] = "AGC"
table.head()


# Attack Score = Average Goals Scored / League Average Goal Scored
# 
# Defense Score = Average Goals Conceded / League Average Goal Conceded

# In[43]:


#Home/Away Attack & Defense Strength

table['HAS'] = table['HGS'] / avg_home_scored
table['AAS'] = table['AGS'] / avg_away_scored
table['HDS'] = table['HGC'] / avg_home_conceded
table['ADS'] = table['AGC'] / avg_away_conceded
table.head()


# In[44]:


cols = ['HAS', 'AAS', 'HDS', 'ADS']

table[cols] = table[cols].round(3)

table.head()


# ### Working on the feature table
# 
# feature_table contains all the fixtures in the current season | 
# ftr = full time result | 
# hst = home shots on target | 
# ast = away shots on target |

# In[45]:


feature_table = df.iloc[:,:23]

feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST']]
feature_table


# In[51]:


extracted = table.iloc[:,5:9]

extracted.head()


# In[54]:


# feature_table = feature_table.join(extracted)
feature_table.head(10)


# In[55]:


#Converts results (H,A or D) into numeric values

def transformResult(row):
    if(row.FTR == 'H'):
        return 1
    elif(row.FTR == 'A'):
        return -1
    else:
        return 0


# In[58]:


feature_table["Result"] = feature_table.apply(lambda row: transformResult(row),axis=1)
feature_table.head(10)


# In[67]:


feature_table.to_csv('final_data.csv')

