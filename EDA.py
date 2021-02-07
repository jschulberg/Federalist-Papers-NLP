##EDA 


#%% Import libraries
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import os
import re
import spacy 
# from spacy.lang.en.stop_words import stop_words
from string import punctuation 


#%% Load Data
# df = pd.read_csv("C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay01.txt",sep='delimiter', header=None)
# Note that I created a Spyder prjoect in my federalist-papers-nlp folder so I
# can just reference the "Data" folder without all the stuff that comes before it.
df = pd.read_csv("Data/essay01.txt", sep='delimiter', header=None)


#%% Create columns and index
df.columns = ['Text']
df.index.names = ['Line']


#%% Tokenize
from nltk.tokenize import sent_tokenize
sentences = []
for sentence in df['Text']:
    sentences.append(sent_tokenize(sentence))

sentences = [y for x in sentences for y in x] 

#%% Finding the most common words
import re
# words = re.findall('\w+', open("C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay01.txt").read().lower())
Counter(words).most_common(100)

#%% Combining all files together 
import io

# Initialize an empty dataframe to hold all of our text
text_df = pd.DataFrame(columns = ["Lines", "Essay"])

# List all of the files in our Data folder
files = os.listdir("Data")

# TODO Loop through all text files and read them into singular dataframe
for text_file in files:
    
    print(f"\n============================")
    print(f"About to load {text_file} in")
    print(f"============================")
        
    temp_df = pd.read_csv('Data/' + text_file, # Read the text file in our data folder
                          delimiter = '\n', # Looks like every line ends with \n
                          header = None, # Read from the first line
                          names = ["Lines"], # Column header
                          error_bad_lines = False) # Skip bad lines (there are like 6 across all essays)
    
    # Add the file name in so we have the essay number for reference
    temp_df['Essay'] = text_file
    
    # Append the temporary dataframe with all corresponding text into our master dataframe
    text_df = text_df.append(temp_df)

#%%% Take a look at our data
text_df.head(10)

