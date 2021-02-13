#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:18:02 2021

@author: Owner
"""


# ----------------------------------------------------------------------------
#                                Purpose
# ----------------------------------------------------------------------------
# The purpose of this script is to load in all of the Federalist Papers essays,
# tokenize them, and join in associated metadata about the papers.
#%%
import pandas as pd
import nltk
nltk.download('punkt')
import os
import requests
# from spacy.lang.en.stop_words import stop_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop = stopwords.words('english')

# ----------------------------------------------------------------------------
#                       Combining all files together 
# ----------------------------------------------------------------------------
#%% Loop through our data folder (note: in Spyder you'll have to open a project
# in our working directory to do this) and read in all of our files
# Initialize an empty dataframe to hold all of our text
text_df = pd.DataFrame(columns = ["lines", "essay"])

# List all of the files in our Data folder
files = os.listdir("Data")

# Loop through all text files and read them into singular dataframe
for text_file in files:
    
    print(f"\n============================")
    print(f"About to load {text_file} in")
    print(f"============================")
        
    temp_df = pd.read_csv('Data/' + text_file, # Read the text file in our data folder
                          delimiter = '\n', # Looks like every line ends with \n
                          header = None, # Read from the first line
                          names = ["lines"], # Column header
                          error_bad_lines = False) # Skip bad lines (there are like 6 across all essays)
    
    # Add the file name in so we have the essay number for reference
    temp_df['essay'] = text_file
    
    # Append the temporary dataframe with all corresponding text into our master dataframe
    text_df = text_df.append(temp_df)

#%%% Take a look at our data
text_df.head(10)


# ----------------------------------------------------------------------------
#                                   Data Cleaning
# ----------------------------------------------------------------------------
#%% The essays come in in the format 'Essay22.txt', and we'd prefer if it just 
# said 'Essay 22'
# Let's start by saving our dataframe as a new object
cleaned_df = text_df.copy()

# Remove .txt first
cleaned_df['essay'] = cleaned_df['essay'].str.replace('.txt', '')

# Put a space in between 'essay' and the number of the essay
cleaned_df['essay'] = cleaned_df['essay'].str.replace('essay', 'Essay ')

# We may also have None (NA) types in the dataframe. Let's drop these
cleaned_df['lines'].dropna(how = 'any')

# Reset the index column for reference
cleaned_df.reset_index(drop = True, inplace = True)

# Because I use a Mac, some rows are .DS_Store, so let's filter those out
cleaned_df_filtered = cleaned_df[cleaned_df['essay'] != '.DS_Store']

cleaned_df_filtered.head(10)


#%% It's important to ensure that the text we analyze is clean. That is, no
# punctuation, everything lowercase, removal of stop words, etc.

cleaned_df_filtered['lines'] = cleaned_df_filtered['lines'].str.replace('[^A-z]', ' ').str.replace(' +', ' ').str.strip()
                                            
# TODO: remove stop words
cleaned_df_filtered['lines'] = cleaned_df_filtered['lines'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# TODO: make everything lower case                
cleaned_df_filtered['lines'] = cleaned_df_filtered['lines'].apply(lambda x: " ".join(x.lower() for x in x.split()))


## Removing puncutation 
# cleaned_df_filtered['lines'] = cleaned_df_filtered['lines'].str.replace('[^\w\s]','')


# ----------------------------------------------------------------------------
#                                Tokenization
# ----------------------------------------------------------------------------
#%% Now that the data's looking generally clean, let's tokenize our data. That is,
# our data is currently in the format such that each line in the text is a row
# in our dataframe. We'd prefer that every row is a word

# Start by creating a new copy of our dataframe
tokenized_df = cleaned_df_filtered.copy()

# Let's look at one tokenized output
word_tokenize(tokenized_df['lines'][1])

# Tokenize the words for every row
tokenized_df['word'] = tokenized_df['lines'].apply(nltk.word_tokenize)

# This puts the words in a given line as a list, but we still need this "exploded"
# into separate rows. Note: This will end up enlarging our dataframe from 16,000
# rows to almost 200,000!
tokenized_df = tokenized_df.explode('word')

tokenized_df.head(10)



# ----------------------------------------------------------------------------
#                           Joining Authorship Data
# ----------------------------------------------------------------------------
# The purpose of this section is to pull the supposed authorship of the
# Federalist Papers, so we can correlate our analysis against the authors of
# the papers.

# Here's the most "readable" format I've found for the authorship of the papers:
# https://guides.loc.gov/federalist-papers/full-text


# ----------------------------------------------------------------------------
#                                Scrape Website
# ----------------------------------------------------------------------------
#%% We'll start by using a combination of the requests and pandas package to 
# parse the HTML of the site of interest to pull the table in as a dataframe.

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
authors_clean = authors.copy().rename(columns = {'No.': 'essay'})

# Convert "No." to an integer column to get rid of the '.0' and then to a string
# so we can append 'Essay ' to the beginning of it
authors_clean['essay'] = "Essay " + authors_clean['essay'].astype('int').astype('str')

# Replace authors for 'Hamilton or Madison' with 'Unknown'
authors_clean['Author'] = authors_clean['Author'].replace('Hamilton or Madison', "Unknown") 

# Convert dates into machine readable format. NAs were brought in as '--', so
# we'll have to replace those before proceeding.
authors_clean['Date'] = pd.to_datetime(authors_clean['Date'].replace('--', 'NaN'), format = "%A, %B %d, %Y")

# Replace blank ('--') publications with 'Unknown'
authors_clean['Publication'] = authors_clean['Publication'].replace('--', 'Unknown')



# ----------------------------------------------------------------------------
#                           Join Authorship Data
# ----------------------------------------------------------------------------
#%% Using the work we prepared above, let's merge this with our actual data.
joined_fedpapers = tokenized_df.merge(authors_clean, on = ['essay', 'essay'])
