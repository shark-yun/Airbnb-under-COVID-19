#!/usr/bin/env python
# coding: utf-8

# In[118]:


# modules for research report
from datascience import *
import numpy as np
import random
import pandas as pd
import folium
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

# module for YouTube video
from IPython.display import YouTubeVideo

# okpy config
from client.api.notebook import Notebook
ok = Notebook('airbnb-final-project.ok')
_ = ok.auth(inline=True)


# # Airbnb Listings and Evictions
# 
# The dataset you will be using is from [Inside Airbnb](http://insideairbnb.com/get-the-data.html), an independent investigatory project that
# collects and hosts substantial Airbnb data on more than 100 cities around the world. The data collected by Inside Airbnb are web-scraped from
# the Airbnb website on a monthly basis. Inside Airbnb was started to investigate the effects of Airbnb on affordable housing and gentrification.
# Its data are made public for free and open for use.  
# 
# We have prepared for you a random subset of Inside Airbnb data from San Francisco collected in June 2020. The data have been
# cleaned for your convenience: all missing values have been removed, and low-quality observations and variables have been filtered
# out. A brief descriptive summary of the dataset is provided below. 
# 
# We are aware that this dataset is potentially significantly larger (in both rows and columns) than other datasets for the project. As a result, 
# you will have many potential directions to conduct your analysis in. At the same time, it is very easy to become overwhelmed or lost with the data.
# We encourage you to reach out by posting your questions on the relevant Piazza thread, or by sending Angela (guanangela@berkeley.edu) or Alan
# (alanliang@berkeley.edu) an email if you need any help.
# 
# **NB: You may not copy any public analyses of this dataset. Doing so will result in a zero.**

# ## Summary
# >Airbnb offers a platform to connect hosts with guests for short-term or long-term lodging accommodations. Compared to similar firms offering vacation rental services
# such as VRBO or HomeAway, Airbnb is the largest and most prominent, with more than 7 million listings worldwide and 2 million people staying in one of its listings
# per night in 2018. Since its founding in 2008, hosts on the platform have served more than 750 million guests, and the firm has grown at an exponential rate globally
# pre-COVID.
# 
# 
# >The data presented are completely from web scraping the Airbnb website in June 2020 for random subset of listings in San Francisco. As a result, the data only contain
# information that a visitor to Airbnb’s site can see. This includes the `listings` table that records all Airbnb units and the `calendar` table that records the
# availabilities for the next 365 days and quoted price per night over the next year of each listing. What each table specifically describes will be gone over in the
# Data Description section below. Note that we do not observe Airbnb transactions or bookings, but only the dates that are available or unavailable through `calendar`.
# 
# 
# >The primary identifier for each listing is the `listing_id` or `id` column (the column name changes depending on the title). Each ID uniquely determines a listing,
# and every listing only has 1 ID. You can visit each listing's URL on Airbnb by going to https://www.airbnb.com/rooms/YOUR_ID_HERE with the id to look up the listing
# on the airbnb website.

# ## Data Description

# The dataset consists of many tables stored in the `data` folder. **You do not have use to all of the tables in your analysis.**
# 1. `listings` provides information on 2000 Airbnb listings in San Francisco. Each row is a unique listing.
# 2. `ratings` contains average ratings for the Airbnb listings across 6 categories and its overall rating. Guests who stay at an Airbnb are eligible to score a listing on each of the categories and on the overall score out of 5.
# 3. `calendar` contains each listing's availability and price over the next year. This data is the same as the calendar that pops up when users try to select the dates of a reservation for a particular listing. For example, the first row means that the listing with ID 40138 was not available on June 8th, 2020. The price per night of this listing is \\$67. 
# 4. `evictions` contains information on evictions in San Francisco, and may be useful if you are interested in determining relationships between Airbnbs and gentrification or evictions.
# 
# 
# There are a lot of columns for many of these datasets, and you probably will only use a few of them. We've selected some of the variables that may be of interest below:
# 
# `listings`:
# * `id`: listing ID.  You can visit each listing's URL on Airbnb by going to https://www.airbnb.com/rooms/YOUR_ID_HERE with the id to look up the listing on the airbnb website.
# * `Name`: listing or rental name
# * `neighborhood` and `neighbourhood_cleansed`: neighborhood of listing
# * `latitude`, `longtitude`: latitude and longitude of listing location. Note that for privacy reasons, this may be approximate.
# * `calculated_host_listings_count`: the number of different listings the host has on Airbnb.
# * `property_type`: type of property the listing is in (e.g. Apartment, Condo, House, etc)
# * `room_type`: type of place (e.g. entire home, private room, etc)
# * `accommodates`: max number of guests
# * `minimum_nights` and `maximum_nights`: minimum and maximum number of nights a reservation can be
# * `availability_X`: availability for the next X days (relative to the scraping date, June 8th, 2020)
# * `amenities`: a list of amenities provided by the listing. Note that each item is an iterable set
# 
# `ratings`: 
# * `review_scores_rating`: review score overall rating of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 100. 
# * `review_scores_accuracy`: review score based on accuracy of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# * `review_scores_cleanliness`: review score based on clealiness of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# * `review_scores_checkin`: review score based on check-in of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# * `review_scores_communication`: review score based on communication with host. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# * `review_scores_location`: review score based on location of listing. Note that on the Airbnb site the score is out of 5, but this value is out of 10. 
# 
# 
# `calendar`:
# * `listing_id`: ID of airbnb listing
# * `date`: date of the potential availability in question
# * `price`: price per night of listing in USD
# * `available`: true or false value representing whether the listing was available.
# 
# `evictions`:
# * `File Date`: date the eviction was reported 
# * `Neighborhood`: neighborhood in which the eviction occurred
# * `Longtitude` and `Latitude`: latitude and longitude of the listing
# * All other columns indicate the reason of the eviction. For example, if an eviction has `True` for the `Non Payment` column and `False` for all other columns, the eviction was due to non-payment. 

# ## Inspiration

# A variety of exploratory analyses, hypothesis tests, and predictions problems can be tackled with this data. Here are a few ideas to get
# you started:
# 
# 
# 1. Can we use Airbnb data to predict which neighborhoods in San Francisco are more gentrified and/or have more evictions?
# 2. Can we predict the overall rating of a listing from one or many of its 6 rating categories? Which of the 6 rating categories best predicts overall rating?  
# 3. Can we predict the average price of a listing (as determined by calendar prices) based on its location, number of bedrooms, or through other features?
# 
# Here are some articles we found to be interesting that may inspire you:
# 1. [Airbnb’s COVID-19 crisis could be a boon for affordable housing](https://www.fastcompany.com/90482662/airbnbs-covid-19-crisis-could-be-a-boon-for-affordable-housing)
# 2. [Identifying salient attributes of peer-to-peer accommodation experience](https://www.tandfonline.com/doi/full/10.1080/10548408.2016.1209153?src=recsys)
# 3. [Airbnb is seeing a surge in demand
# ](https://www.latimes.com/business/story/2020-06-07/airbnb-coronavirus-demand)
# 4. [Airbnb’s Coronavirus Crisis: Burning Cash, Angry Hosts and an Uncertain Future
# ](https://www.wsj.com/articles/airbnbs-coronavirus-crisis-burning-cash-angry-hosts-and-an-uncertain-future-11586365860)
# 5. [Research: When Airbnb Listings in a City Increase, So Do Rent Prices
# ](https://hbr.org/2019/04/research-when-airbnb-listings-in-a-city-increase-so-do-rent-prices)
# 6. [Airbnb is getting ripped apart for asking guests to donate money to hosts
# ](https://www.businessinsider.com/airbnb-asking-renters-to-donate-kindness-cards-backlash-2020-7?utm_source=reddit.com)
# 
# Don't forget to review the [Final Project Guidelines](https://docs.google.com/document/d/1NuHDYTdWGwhPNRov8Y3I8y6R7Rbyf-WDOfQwovD-gmw/edit?usp=sharing) for a complete list of requirements.

# ## Preview
# 
# The tables are loaded in the code cells below. Take some time to explore them!

# In[119]:


calender_2019 = Table().read_table("data/calendar_2019.csv")
calender_2019.show(5)


# In[120]:


# Load in the calendar table
calendar_2020 = Table().read_table("data/calendar.csv")
calendar_2020.show(5)


# <br>
# 
# # Research Report

# ## Introduction
# 
# 
# In this project, we are exploring the data of Inside Airbnb, an independent project that collects and hosts Airbnb data on 100 or more cities around the world. Thankfully, Inside Airbnb made their data public so we can explore their data freely. What we are using is the Airbnb data from San Francisco collected in June 2020 and 2019.  We are comparing two datasets in different year in order to see how much COVID-19 affected Airbnb in San Francisco. From the given datasets, we will be using ‘calendar_2019’ and ‘calender_2020’. Looking at our dataset ‘calendar’, this dataset consists of features, ‘listing_id’, ‘date’, ‘price’, and ‘available’. The ‘listing_id’ is used to distinguish the data points, ‘date’ gives the potential availability in question, ‘price’ gives price per night of listing in USD, and finally ‘available’ gives true or false value representing whether the listing was available. After reviewing our datasets’ features, we decided to focus on ‘listing_id’ and ‘available’ in order to check the availability before after COVID-19 occured. These variables will be the most important feature to our analysis since it directly shows whether availability changed between two years by true and false. We can count the number of true and false, draw a distribution chart, and get percentage chart in order to see the analysis this question in multiple views.

# ## Hypothesis Testing and Prediction Questions
# 
# 
# Our team (Yun and Andy) were both affected by COVID-19 earlier this year on shelter problems. We both had experience of searching on Airbnb to find a place for 2 weeks of self-isolation in our home country. This experience motivated us to analyze on how the Airbnb’s availability changed by COVID-19. Specifically, we are focusing on how the availability changed on year 2020, the year COVID-19 started, and before year 2020. We suspected that the percentage of Airbnb availability days went up after COVID-19 since there were less travelers. Our null hypothesis is the distributions of the available days for Airbnb did change after the COVID-19.
# 
# **Since we want to compare values from two year, 2019 and 2020, it is best to use A/B testing for our hypothesis testing.For the prediction part, our team is using the price mean of 2019 and availabilty of 2019 to create the regression line to predict the availability for 2020.**

# ## Exploratory Data Analysis
# 

# 
# This table shows the difference of available days for same place before and after COVID-19. <br />
# <br />
# The features of this table consists of:<br />
# listing_id: ID of airbnb listing<br />
# availability before COVID-19: the number of available days listed on the year 2019<br />
# availability after COVID-19: the number of available days listed on the year 2020<br />
# change: The difference between number of available days between the year 2019 and 2020.<br />
# price mean: The average of Airbnb price over the year <br />
# 
# <br />
# This table is significant since it shows trend of how the available days has increased after the COVID-19. 
# Therefore we suspected that the available days has increased due to COVID-19 suspecting due to lack of international travelers.
# 

# In[121]:


a = calender_2019.where('available', 't').group(['listing_id', 'available'], np.mean).select(['listing_id','price mean'])
b = calender_2019.where('available', 't').select(['listing_id', 'available']).group(['listing_id', 'available'])
pan_before = b.join('listing_id', a, 'listing_id').drop('available').relabel('count','availability before Covid-19')
pan_before.show(5)

c = calendar_2020.where('available', 't').group(['listing_id', 'available'], np.mean).select(['listing_id','price mean'])
d = calendar_2020.where('available', 't').select(['listing_id', 'available']).group(['listing_id', 'available'])
pan_after = d.join('listing_id', c, 'listing_id').drop('available').relabel('count','availability after Covid-19')
pan_after.show(5)

combine = pan_before.join('listing_id',pan_after,'listing_id')
combine= combine.select('listing_id','availability before Covid-19','availability after Covid-19')
combine= combine.with_column('Availability change',combine.column('availability after Covid-19')-combine.column('availability before Covid-19'))
combine.show(5)

aver_diff = np.mean(combine.column('Availability change'))


# This table shows the percentage difference of available days for same place before and after COVID-19. The features of this table consists of:<br/>
# listing_id: ID of airbnb listing<br/>
# % availability before COVID-19: the percentage of available days on the year 2019<br/>
# % availability after COVID-19: the percentage of available days on the year 2020<br/>
# change: The difference between percentage of available days between the year 2019 and 2020.<br/>

# In[122]:


per_combine=Table().with_column('id',combine.column('listing_id'))
per_combine= per_combine.with_column('% availability before Covid-19',combine.column('availability before Covid-19')/365*100)
per_combine= per_combine.with_column('% availability after Covid-19',combine.column('availability after Covid-19')/365*100)
per_combine= per_combine.with_column('change',per_combine.column('% availability after Covid-19')-per_combine.column('% availability before Covid-19'))
per_combine.show(5)
aver_per_diff=np.mean(per_combine.column('change'))


# **Quantitative Plot:**

# In[123]:


combine.hist('availability before Covid-19','availability after Covid-19')


# In[124]:


per_combine.hist('% availability before Covid-19','% availability after Covid-19')


# availability before Covid-19:the number of available days listed on the year 2019<br/>
# availability after Covid-19:the number of available days listed on the year 2020<br/>
# <br/>
# The histogram shows the availability before and after Covid-19. 

# ## Hypothesis Testing

# Looking at our dataset, we suspected there was an increase in days available for places in Airbnb. We wanted to check if this prediction is true or not which led to this analysis. **Our null hypothesis is the distributions of the available days for Airbnb did change after the COVID-19.**
# Our alternative hypothesis is that the distributions of the available days for Airbnb did not actually change after the COVID-19 which means COVID-19 impacted Airbnb’s job.
# We decide to check this with a 0.05 level of significance. Since we want to compare values from two year, 2019 and 2020, it is best to use A/B testing for our hypothesis testing.

# In[125]:


def difference_of_means(T, A, B):
    means_table = T.select(A, B).group(B, np.average)
    means = means_table.column(1)
    return means.item(0) - means.item(1)


# In[126]:


def one_simulated_difference(T, A, B):
    shuffled_labels = T.sample(with_replacement = False).column(B)
    shuffled_table = T.select(A).with_column('Shuffled Label', shuffled_labels)
    return difference_of_means(shuffled_table, A, 'Shuffled Label')   


# In[127]:


observed_difference=np.mean(combine.column('availability after Covid-19'))-np.mean(combine.column('availability before Covid-19'))
observed_difference


# In[128]:


# Use this cell to generate your qualitative plo# Use this cell to generate your qualitative plot

ava_sample= make_array()
repetitions =1000
for i in np.arange(repetitions):
    a=one_simulated_difference(combine,'availability before Covid-19','availability after Covid-19')
    ava_samples= np.append(samples,a)
    
    
Table().with_column('Difference Between before and after Covid-19', ava_samples).hist()
plots.title('Availability difference')
plots.scatter(observed_difference, 0, color='red', s=40)


# In[129]:


a=percentile(2.5,ava_samples)
b=percentile(97.5,ava_samples)
print('95% confidence interval range is [',a,',',b,']')


# In[130]:


observed_per_difference=np.mean(per_combine.column('% availability after Covid-19'))-np.mean(per_combine.column('% availability before Covid-19'))
observed_per_difference


# In[131]:


# Use this cell to generate your qualitative plo# Use this cell to generate your qualitative plot

per_samples= make_array()
repetitions =1000
for i in np.arange(repetitions):
    a=one_simulated_difference(per_combine,'% availability before Covid-19','% availability after Covid-19')
    per_samples= np.append(per_samples,a)
    
    
Table().with_column('% Difference Between before and after Covid-19', samples).hist()
plots.title('% availability difference')
plots.scatter(observed_per_difference, 0, color='red', s=40)


# In[132]:


a=percentile(2.5,per_samples)
b=percentile(97.5,per_samples)
print('95% confidence interval range is [',a,',',b,']')


# Looking at our observed difference, it is included in our 95% confidence interval. This means that our observed difference from the dataset exploration seems reasonable which leads to rejecting our null hypothesis. Therefore we reject our null hypothesis and approve our alternative hypothesis. The distributions of the available days in Airbnb did not change after COVID-19.

# ## Prediction
# 
# **Be sure to set a random seed so that your results are reproducible.**

# In[133]:


def su(a):
    return (a-np.mean(a))/np.std(a)


# In[134]:


def correlation(t, a, b):
    return np.mean(su(t.column(a))*su(t.column(b)))

def slope(t, a, b):
    r = correlation(t, a, b)
    return r*np.std(t.column(b))/np.std(t.column(a))

def intercept(t, a, b):
    return np.mean(t.column(b)) - slope(t, a, b)*np.mean(t.column(a))


# In[135]:


before_covid_r = correlation(pan_before,'price mean','availability before Covid-19')
print('correlation between the airbnb price and availability is',before_covid_r)
before_covid_s = slope(pan_before,'price mean','availability before Covid-19')
print('Slope between the airbnb price and availability between the airbnb price and availability is',before_covid_s)
before_covid_i = intercept(pan_before,'price mean','availability before Covid-19')
print('intercept between the airbnb price and availability is',before_covid_i)

print('Predition availability after Covid-19 =',before_covid_s,'*','price before Covid-19 +',before_covid_i)


# In[136]:


pred_before= pan_before.with_column('predict availability',before_covid_s*pan_before.column('price mean')+before_covid_i)
print(pred_before)

pred_after= pan_after.with_column('predict availability',before_covid_s*pan_after.column('price mean')+before_covid_i)
pred_after


# In[137]:


after_plot=pred_after.select('availability after Covid-19','price mean','predict availability')
after_plot.scatter('price mean')


# For the prediction part, our team is using the price mean of 2019 and availabilty of 2019 to create the regression line to predict the availability for 2020. 
# 
# There are many reasons why the data does not fits in the regression line, 
# 
# 1.Not all the airbnb is open for 365 which their maximum availalbe day is limited,
# Availability is showoing whether the airbnb is booked or not, but when the customer choosing the place, they will noramlly look in to the details like the overall rate, and those extra fee.
# 2.we are just taking average price for each airbnb, but we did not include the clean fee which could be real concern for many people while choosing the airbnb 
# 3.Overall rate
# 
# Best way to improve this prediction is to use the multiple regression line, but data 8 does not including it.

# ## Conclusion
# 
# In conclusion, we suspected that the distributions of the available days in Airbnb changed after COVID-19. Surprisingly, the distributions of the available days did not change after COVID-19 since the distribution is in our 95% confidence interval. If we had more data for years other than 2019 and 2020, we can see how the usual normal distribution is for each year, and get more accurate distribution on whether the change of distribution was big or not. We would have liked to perform analysis on how the availability is related to the price of the rent.

# ## Presentation
# 
# *In this section, you'll need to provide a link to your video presentation. If you've uploaded your presentation to YouTube,
# you can include the URL in the code below. We've provided an example to show you how to do this. Otherwise, provide the link
# in a markdown cell.*
# 
# **Link:** https://www.youtube.com/watch?v=05D-wDRwFRY

# # Submission
# 
# *Just as with the other assignments in this course, please submit your research notebook to Okpy. We suggest that you
# submit often so that your progress is saved.*

# In[143]:


# Run this line to submit your work
_ = ok.submit()


# In[ ]:




