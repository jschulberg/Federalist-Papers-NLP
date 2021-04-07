#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:55:57 2021

@author: Owner
"""

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
import os
import requests
import string 

### NLTK Download
# Note: To download nltk products, you need to run the nltk downloader. If you 
# just want to run this quickly, uncomment the following line and run:
# nltk.download('popular')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



# ----------------------------------------------------------------------------
#                       Combining all files together 
# ----------------------------------------------------------------------------
#%% Loop through our data folder (note: in Spyder you'll have to open a project
# in our working directory to do this) and read in all of our files
# Initialize an empty dataframe to hold all of our text
text_df = pd.DataFrame(columns = ["lines", "essay"])

# Note that we need to go back one folder to the parent directory so that we can actually access the Data/ folder
parent_dir = os.path.realpath('')

# List all of the files in our Data folder
files = os.listdir(parent_dir + "/Data/")

# We may have non-text files in here, so let's remove these from our lists
txt_files = filter(lambda x: x[-5:] == '.txt', files)

txt_files = [x for x in files if x[-4:] == '.txt']
print(txt_files)  # only text files


# Loop through all text files and read them into singular dataframe
for text_file in txt_files:
    
    print(f"\n============================")
    print(f"About to load {text_file} in")
    print(f"============================")
        
    temp_df = pd.read_csv(parent_dir + '/Data/' + text_file, # Read the text file in our data folder
                          delimiter = '\n', # Looks like every line ends with \n
                          header = None, # Read from the first line
                          names = ["lines"], # Column header
                          error_bad_lines = False) # Skip bad lines (there are like 6 across all essays)
    
    # Add the file name in so we have the essay number for reference
    temp_df['essay'] = text_file
    
    # Append the temporary dataframe with all corresponding text into our master dataframe
    text_df = text_df.append(temp_df)

# Take a look at our data
print(text_df.head(10))


#%% Lemmatization
# Because a lot of the words are similar, but not exactly (like state and states), 
# we'll use a lemmatization method to find the canonical version of each.
lemmatizer = nltk.stem.WordNetLemmatizer()

# Initialize two dataframes:
    # 1. Error DataFrame | This will keep track of any words that have issues in
    #                       the lemmatizer
    # 2. Lemmatized DataFrame | This will be a copy of our original dataframe
error_df = pd.DataFrame(columns = ['idx', 'word', 'line', 'essay'])

#%%
# The following function would map the treebank tags to WordNet part of speech names
# This specifically helps with lemmatizing based on the word's part of speech
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    

#%% It's important to ensure that the text we keep is meaningful.
# To assist with this, we want to filter out any stop words, which don't
# mean much to us
stop = stopwords.words('english')
# additional stop words
stop.extend(['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 
              'also', 'ought', 'a', 'the', 'it', 'i', 'upon', 'but', 'if', 'in',
              'this', 'might', 'and', 'us', 'can', 'as', 'to', 'make', 'made',
             'much'])


#%% Define a word check that returns what we'd like
def lemmatize_words(word, pos):
    '''
    A function to lemmatize a word and, if not, return the lowercase version
    of the original word
    
    Parameters
    ----------
    word : string
        word to be lemmatized.
    pos : string
        part of speech of the word.

    Returns
    -------
    lemmatized_word : string
        lemmatized word.

    '''
    
    # Make everything lowercase and lemmatize based on pos                    
    try:
        # Lemmatize our word using the predefined function in the previous cell
        lemmatized_word = lemmatizer.lemmatize(word.lower(), pos = get_wordnet_pos(pos))
    # For some words, the lemmatizer isn't working. Let's log these and revisit
    # them later
    except:
        # If we hit an error, let's just keep the word as is
        lemmatized_word = word.lower()
    
    print(f"\t\t* Keeping\t{word} \t-->\t\t{lemmatized_word}")
        
    return lemmatized_word

# ----------------------------------------------------------------------------
#%%                              Part of Speech Tagging
# ----------------------------------------------------------------------------
# Next, we'll try to build out a dataframe, simultaneously taking out stop words
# and tagging the parts of speech.

# Initialize empty lists for each of the columns we'll want
index_list = []
essay_list = []
line_list = []
word_list = []
lemmatized_word_list = []
part_of_speech_list = []


# Start by converting our two columns to lists so they're easier to work with
lines = list(text_df['lines'])
essays = list(text_df['essay'])


# Let's zip the lists together so we can simultaneously iterate through them
for idx, item in enumerate(zip(lines, essays)):
    # if idx > 100:
    #     break
    print("\n\n===========================================")
    print(f"\t\t\t\tIndex {idx}:")
    print("===========================================")
    
    # Build a list of tokens for the given line
    tokens = nltk.word_tokenize(item[0])
        
    # Figure out the parts of speech for each word in our line
    pos_tags = nltk.pos_tag(tokens)
    print(f"{item}\n")
    
    # Loop through our words and parts of speech to lemmatize and remove stop words
    for word, pos in pos_tags:
        # Filter out any stop words (all lowercase) or punctuation
        if word.lower() in stop or word in string.punctuation or not word.isalpha():
            print(f"Removing '{word}'")
            continue
        else:
            lemmatized_word = lemmatize_words(word, pos)
        
        # Now we can append our results to our lists above
        index_list.append(idx)
        essay_list.append(item[1])
        line_list.append(item[0])
        word_list.append(word)
        lemmatized_word_list.append(lemmatized_word)
        part_of_speech_list.append(pos)


#%% Append results to dataframe
cleaned_df = pd.DataFrame(list(zip(index_list, essay_list, line_list, word_list, 
                      lemmatized_word_list, part_of_speech_list)),
               columns = ['line_index', 'essay', 'lines', 'word', 'lemmatized_word', 
                          'part_of_speech'])



# ----------------------------------------------------------------------------
#                                   Data Cleaning
# ----------------------------------------------------------------------------
#%% The essays come in in the format 'Essay22.txt', and we'd prefer if it just 
# said 'Essay 22'
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



print(cleaned_df_filtered.head(10))


# ----------------------------------------------------------------------------
#                                Tokenization
# ----------------------------------------------------------------------------
#%% Now that the data's looking generally clean, let's tokenize our data. That is,
# our data is currently in the format such that each line in the text is a row
# in our dataframe. We'd prefer that every row is a word

# Start by creating a new copy of our dataframe
tokenized_df = cleaned_df_filtered.copy()

#Renamed the column here to "Essay" so that we can perform an inner join. 
tokenized_df = tokenized_df.rename(columns = {'essay': 'Essay', 'word':'Word', 'lines': 'Lines'}) \
    .reset_index(drop = True)

#Rnamed all the Essays so that they would match up to the authors_clean dataframe when merging. 
tokenized_df = tokenized_df.replace("Essay 01", "Essay 1") \
    .replace("Essay 02", "Essay 2") \
    .replace("Essay 03", "Essay 3") \
    .replace("Essay 04", "Essay 4") \
    .replace("Essay 05", "Essay 5") \
    .replace("Essay 06", "Essay 6") \
    .replace("Essay 07", "Essay 8") \
    .replace("Essay 08", "Essay 8") \
    .replace("Essay 09", "Essay 9")



#%%
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
# We'll start by using a combination of the requests and pandas package to 
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

#Renaming the Publications
authors_clean['Publication'] = authors_clean['Publication'].replace('For the Independent Journal', 'Independent Journal')
authors_clean['Publication'] = authors_clean['Publication'].replace("Frm the New York Packet", "New York Packet")
authors_clean['Publication'] = authors_clean['Publication'].replace("From the New York Packet", "New York Packet")
authors_clean['Publication'] = authors_clean['Publication'].replace("From McLEAN's Edition, New York", "McLEAN's Edition")
authors_clean['Publication'] = authors_clean['Publication'].replace("From McLEAN's Edition", "McLEAN's Edition")
authors_clean['Publication'] = authors_clean['Publication'].replace("From the Daily Advertiser",  "Daily Advertiser")
authors_clean['Publication'] = authors_clean['Publication'].replace("From The New York Packet",  "New York Packet")


# ---------------------------------------------------------------------------- 
#                             Join Authorship Data
# ----------------------------------------------------------------------------
#%% Using the work we prepared above, let's merge this with our actual data.
joined_fedpapers = tokenized_df.merge(authors_clean, 
                                      left_on = 'Essay',
                                      right_on = 'Essay',
                                      how = 'inner')

print(joined_fedpapers.head(10))

# ----------------------------------------------------------------------------
#                                  Save Work
# ----------------------------------------------------------------------------
# Let's write our final dataframe out to a csv file so it's easier to do EDA.
joined_fedpapers.to_csv(parent_dir + "/Data/full_fedpapers.csv", index = False)

