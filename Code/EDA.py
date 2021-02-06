##EDA 


# Import libraries
#%%
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import re
import spacy 
# from spacy.lang.en.stop_words import stop_words
from string import punctuation 


# Load Data
#%%
# df = pd.read_csv("C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay01.txt",sep='delimiter', header=None)
df = pd.read_csv("C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay01.txt",sep='delimiter', header=None)

# Create columns and index
#%%
df.columns = ['Text']
df.index.names = ['Line']

# Tokenize
from nltk.tokenize import sent_tokenize
sentences = []
for sentence in df['Text']:
    sentences.append(sent_tokenize(sentence))

sentences = [y for x in sentences for y in x] 

# Finding the most common words
#%%
import re
words = re.findall('\w+', open("C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay01.txt").read().lower())
Counter(words).most_common(100)

#Combining all files together 
#%%
import io

essays = ['C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay01.txt','C:\\Users\\sabri\\Desktop\\Federalist\\Data\\essay02.txt']
lines = io.StringIO()   
for file_dir in files:
    with open(file_dir, 'r') as file:
        lines.write(file.read())
        lines.write('\n')

lines.seek(0)       
print(lines.read())