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

# List all of the files in our Data folder
files = os.listdir("Data")

# Initialize an empty dataframe to hold all of our text
text_df = pd.DataFrame()

lines = io.StringIO()   

for file_dir in files:
    with open("Data/" + file_dir, 'r', encoding = "windows-1252") as file:
        print(file.read())
        lines.write(file.read())
        lines.write('\n')
        text_df.append(lines)

lines.seek(0)       
print(lines.read())