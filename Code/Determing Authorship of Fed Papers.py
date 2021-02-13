#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:18:02 2021

@author: Owner
"""

# ----------------------------------------------------------------------------
#                                Purpose
# ----------------------------------------------------------------------------
# The purpose of this script is to pull the supposed authorship of the
# Federalist Papers, so we can correlate our analysis against the authors of
# the papers.

# Here's the most "readable" format I've found for the authorship of the papers:
# https://guides.loc.gov/federalist-papers/full-text


# ----------------------------------------------------------------------------
#                                Scrape Website
# ----------------------------------------------------------------------------
#%% We'll start by using a combination of the requests and pandas package to 
# parse the HTML of the site of interest to pull the table in as a dataframe.

import requests
import pandas as pd

# URL of the page we'd like to scrape
url = 'https://guides.loc.gov/federalist-papers/full-text'

# "Get" the request from the url and pull the html content from the page
html = requests.get(url).content

# Read the html from the page, which will automatically look for any tables
# so we can convert them to dataframes
authors_list = pd.read_html(html)

# We got a list of dataframes (in case there were multiple tables), but we really
# only have one dataframe in the list, so let's just pull the first item in the list
authors = authors_list[0]

# Check to make sure it looks good
print(authors)


# ----------------------------------------------------------------------------
#                                Clean DF
# ----------------------------------------------------------------------------
#%% Now that we have the dataframe we want, let's clean everything up for analysis
# purposes. In particular:
    # 1. Make the essay No. column of the form "Essay 1" instead of "1"
    # 2. For any papers that have "Hamilton or Madison" as authors, make these
    #   "Unknown"
    # 3. Turn the dates into a machine-readable format
    # 4. Replace blank ('--') publications with 'Unknown'
    
# Make a copy of the dataframe to work off and rename the 'No.' column
authors_clean = authors.copy().rename(columns = {'No.': 'Essay'})

# Convert "No." to an integer column to get rid of the '.0' and then to a string
# so we can append 'Essay ' to the beginning of it
authors_clean['Essay'] = "Essay " + authors_clean['Essay'].astype('int').astype('str')

# Replace authors for 'Hamilton or Madison' with 'Unknown'
authors_clean['Author'] = authors_clean['Author'].replace('Hamilton or Madison', "Unknown") 

# Convert dates into machine readable format. NAs were brought in as '--', so
# we'll have to replace those before proceeding.
authors_clean['Date'] = pd.to_datetime(authors_clean['Date'].replace('--', 'NaN'), format = "%A, %B %d, %Y")

# Replace blank ('--') publications with 'Unknown'
authors_clean['Publication'] = authors_clean['Publication'].replace('--', 'Unknown')
