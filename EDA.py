##EDA 


#%% Import libraries
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, sent_tokenize, corpus
nltk.download('punkt')
import os
import re
import spacy 
# from spacy.lang.en.stop_words import stop_words
from string import punctuation 
from nltk.tokenize import sent_tokenize #adding tockenize libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')


#%% Load Data
# df = pd.read_csv("C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay01.txt",sep='delimiter', header=None)
# Note that I created a Spyder prjoect in my federalist-papers-nlp folder so I
# can just reference the "Data" folder without all the stuff that comes before it.
df = pd.read_csv("Data/essay01.txt", sep='delimiter', header=None)


#%% Create columns and index
df.columns = ['Text']
df.index.names = ['Line']


#%% Tokenize
sentences = []
for sentence in df['Text']:
    sentences.append(sent_tokenize(sentence))

sentences = [y for x in sentences for y in x] 

#%% Finding the most common words
import re
# words = re.findall('\w+', open("C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay01.txt").read().lower())
Counter(words).most_common(100)

#%% Combining all files together 
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

#%% Data cleaning
# The essays come in in the format 'Essay22.txt', and we'd prefer if it just said 'Essay 22'
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

##EDA prior to Text Cleaning
tokenized_clean['word_count'] = tokenized_clean['lines'].apply(lambda x: len(str(x).split(" ")))
tokenized_clean[['lines','word_count', 'essay']].head()

#%% Text Cleaning
# It's important to ensure that the text we analyze is clean. That is, no
# punctuation, everything lowercase, removal of stop words, etc.
tokenized_clean = cleaned_df_filtered.copy()

tokenized_clean['lines'] = tokenized_clean['lines'].str.replace('[^A-z]', ' ').str.replace(' +', ' ').str.strip()
                                            
# TODO: remove stop words
tokenized_clean['lines'] = tokenized_clean['lines'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# TODO: make everything lower case                
tokenized_clean['lines'] = tokenized_clean['lines'].apply(lambda x: " ".join(x.lower() for x in x.split()))


##Removing puncutation 
##tokenized_clean['lines'] = tokenized_clean['lines'].str.replace('[^\w\s]','')



#%% Tokenization
# Now that the data's looking generally clean, let's tokenize our data. That is,
# our data is currently in the format such that each line in the text is a row
# in our dataframe. We'd prefer that every row is a word

# Start by creating a new copy of our dataframe
tokenized_df = tokenized_clean.copy()

# Let's look at one tokenized output
word_tokenize(tokenized_df['lines'][1])

# Tokenize the words for every row
tokenized_df['tokenized'] = tokenized_df['lines'].apply(nltk.word_tokenize)

# This puts the words in a given line as a list, but we still need this "exploded"
# into separate rows. Note: This will end up enlarging our dataframe from 16,000
# rows to almost 200,000!
tokenized_df = tokenized_df.explode('tokenized')



#%%
tokenized_df.head(10)


