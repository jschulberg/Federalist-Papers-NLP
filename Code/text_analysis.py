##Text Analysis - Cosine Similarity/Other Text Analysis

# ----------------------------------------------------------------------------
#                                Purpose
# ----------------------------------------------------------------------------
# The purpose of this script is do a text analysis of the 
# the 85 Essays in the Federalist Papers. By doing so we can figure out who wrote
# the unknown essays and how related these essays are to one another. 
#
# ## Import Libraries
import pandas as pd
import nltk
import os
import requests


# ----------------------------------------------------------------------------
#                       Combining all files together 
# ----------------------------------------------------------------------------
#%% Loop through our data folder (note: in Spyder you'll have to open a project
# in our working directory to do this) and read in all of our files
# Initialize an empty dataframe to hold all of our text

import os

text_df = pd.DataFrame(columns = ["lines", "essay"])
files = os.listdir("C:\\Users\\sabri\\Desktop\\Federalist\\Data\\")

# Loop through all text files and read them into singular dataframe
for text_file in files:
    
    print(f"\n============================")
    print(f"About to load {text_file} in")
    print(f"============================")
        
    temp_df = pd.read_csv('C:\\Users\\sabri\\Desktop\\Federalist\\Data\\' + text_file, # Read the text file in our data folder
                          delimiter = '\n', # Looks like every line ends with \n
                          header = None, # Read from the first line
                          names = ["lines"], # Column header
                          error_bad_lines = False) # Skip bad lines (there are like 6 across all essays)
    
    # Add the file name in so we have the essay number for reference
    temp_df['essay'] = text_file
    
    # Append the temporary dataframe with all corresponding text into our master dataframe
    text_df = text_df.append(temp_df)




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


#%% Text Cleaning
# It's important to ensure that the text we analyze is clean. That is, no
# punctuation, everything lowercase, removal of stop words, etc.

cleaned_df_filtered['lines'] = cleaned_df_filtered['lines'].str.replace('[^A-z]', ' ').str.replace(' +', ' ').str.strip()
                                            
# TODO: remove stop words

from nltk.corpus import stopwords
stop = stopwords.words('english')

cleaned_df_filtered['lines'] = cleaned_df_filtered['lines'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# TODO: make everything lower case                
cleaned_df_filtered['lines'] = cleaned_df_filtered['lines'].apply(lambda x: " ".join(x.lower() for x in x.split()))


##Removing puncutation 
cleaned_df_filtered['lines'] = cleaned_df_filtered['lines'].str.replace('[^\w\s]','')


fed_papers = cleaned_df_filtered.copy()

fed_papers.head()


# ----------------------------------------------------------------------------
#                           Cosine Similarity
# ----------------------------------------------------------------------------
# The purpose of this section is analyze the texts using cosine similarity.
# Cosine similarity is calculated by measuring the angle of the cosine between two vectors. 
# The smaller the angle, the higher the cosine similarity.
# Magnitute is not important in cosine similarity, only orientation. This is useful
# in text analysis because even if two documents have varying lengths, they could 
# still be related in terms of content. 
# This is what makes it more advantageous than other distance measures in text analysis. 


#Grouping all of the lines by essay number

fed_papers = fed_papers.groupby("essay")
fed_papers= fed_papers["lines"].agg(lambda column: "".join(column))

#Resetting the index 
fed_papers = fed_papers.reset_index(name="lines")


fed_papers.head()


#Converts lines into a vectorized TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

#Transforms the data
fed_transform = TfidfVectorizer().fit_transform(fed_papers['lines'])

fed_transform


#Slicing the matrix to get a submatrix in each row
#Let's take a look at the first Essay

fed_transform[0:1]


#Importing a linear kernal 
from sklearn.metrics.pairwise import linear_kernel



#Getting similarites through the linear kernal 
fed_cosine_similarities = linear_kernel(fed_transform[0:1],fed_transform).flatten()


fed_cosine_similarities


# This will show you the similarities between the first Essay and the other Essays. 
# The range is from 0-1. 1 meaning they are the most similar, 0 meaning they are the least similar. 
# As you can see the "1" in the first row and column is because it is comparing the first Essay to itself.



#Sorting the cosine similarities
#Let's look at the five most related Essays to the first Essay

related_essays = fed_cosine_similarities.argsort()[:-7:-1]
related_essays


#Cosine Similarity of the related essays
fed_cosine_similarities[related_essays]


# Essay 1 - Hamilton - For the Independent Journal - General Introduction
# 
# Excluding itself, the five most similar Essays to Essay 1 were the following: 
# 
# 1. Essay 84 - Hamilton - From McLEAN's Edition - Certain General and Miscellaneous Objections to the Constitution Considered and Answered
# 
# 
# 2. Essay 85 - Hamilton - 	From McLEAN's Edition - Concluding Remarks   
# 
# 
# 3. Essay 22 - Hamilton - From the New York Packet  - The Same Subject Continued: Other Defects of the Present Confederation - Friday, December 14, 1787
# 
# 
# 4. Essay 43 - Madison -  	For the Independent Journal - 	The Same Subject Continued: The Powers Conferred by the Constitution Further Considered
# 
# 
# 5.	Essay 40 - Madison - From the New York Packet - The Powers of the Convention to Form a Mixed Government Examined and Sustained - Friday, January 18, 1788

##----------------------------------------------------------------------------------------------

#Let's take a look at one of the unknown Essays - Essay 52
fed_transform[51:52]


#Getting similarites through the linear kernal 
fed_cosine_similarities = linear_kernel(fed_transform[51:52],fed_transform).flatten()


fed_cosine_similarities


#Sorting the cosine similarities
#Let's look at the five most related Essays to the 52nd Essay

related_essays = fed_cosine_similarities.argsort()[:-7:-1]
related_essays

#Cosine Similarity of the related essays
fed_cosine_similarities[related_essays]


#Essay 52 - Unknown - From the New York Packet - The House of Representatives - Friday, February 8, 1788


#Excluding itself, the five most similar Essays to Essay 49 were the following: 

#1. Essay 55 - Unknown - From the New York Packet - The Total Number of the House of Representatives - Friday, February 15, 1788


#2. Essay 63 - Unknown - For the Independent Journal - The Senate 


#3. Essay 53 - Unknown - From the New York Packet - The Same Subject Continued: The House of Representatives - Tuesday, February 12, 1788



#4. Essay 39 - Madison -  For the Independent Journal - Conformity of the Plan to Republican Principles


#5.Essay 59 - Hamilton - From the New York Packet Concerning the Power of Congress to Regulate the Election of Members - Friday, February 22, 1788

#-------------------------------------------------------------------------------------------------------
#To Do 
#Loop through the similarities, figure which Essays were related to each and put it into a dataframe. 
#Create a methodology where we sort which Essays were most similar to which author. 
#Example - In Essay 52, it was attributed once to Madison and once Hamilton in the top, the next author it is most similar to, 
#we attribute Essay 52 to that person. 
#Another methodology is to do a k-means clustering to group the Essays together. 

