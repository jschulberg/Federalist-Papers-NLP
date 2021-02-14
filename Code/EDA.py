##EDA 

# ----------------------------------------------------------------------------
#                                Purpose
# ----------------------------------------------------------------------------
# The purpose of this script is to conduct basic exploratory data analysis on 
# the 85 essays in the Federalist Papers.



#%% Load Data and import packages
import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

### NLTK Download
# Note: To download nltk products, you need to run the nltk downloader. If you 
# just want to run this quickly, uncomment the following line and run:
# nltk.download('popular')

# Note that I created a Spyder prjoect in my federalist-papers-nlp folder so I
# can just reference the "Data" folder without all the stuff that comes before it.
fed_papers = pd.read_csv("Data/full_fedpapers.csv")


# ----------------------------------------------------------------------------
#                             Viz 1: Top 20 Words
# ----------------------------------------------------------------------------
#%% Our first visualization constitutes the top 20 words across all documents.

# Start by creating a grouped dataframe of our word counts
word_counts = fed_papers.groupby(['word']) \
    .size() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

print(word_counts.head(10))
      
# Before we move on, there are a lot of unnecessary words here! Let's filter
# some of these (stop words) out.
stop_words = ['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 'also', 'ought']

words_nonstop = word_counts.copy()
words_nonstop = words_nonstop[~words_nonstop['word'].isin(stop_words)]

print(words_nonstop.head(10))

# It also looks like there are words that should be counted together (i.e. state
# and states). Let's use a lemmatizer to solve this.


# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz1 = sns.barplot(x = 'count',
            y = 'word',
            data = words_nonstop[:20],
            palette = "Purples_r")

# Set our labels
viz1.set(xlabel='Number of Appearances', ylabel='Word', title = 'Word Counts across all Federalist Papers')
plt.show()

# Save our plot to the Viz folder 
viz1.figure.savefig("Viz/Top 20 Words.png")

      
# ----------------------------------------------------------------------------
#                         Viz 2: Top 10 Words by Author
# ----------------------------------------------------------------------------
#%% TODO: Our second visualization constitutes a bar chart of the top 10 words  
# by word count of each author (John Jay, Alexander Hamilton, James Madison, or 
# Unknown).



# ----------------------------------------------------------------------------
#                      Viz 3: Word Count vs. Word Frequency
# ----------------------------------------------------------------------------
#%% TODO: Our third visualization constitues a scatter plot of all the words
# that could reasonably appear in our dataset, measuring the number of times
# each one appears as well as the number of documents it appears in.

# The hope here is to take a look at what will eventually be the TF-IDF of each
# word: that way we can filter out words that appear many times but only in very
# few documents (i.e. 'Constitution' appears 100 times in total but 95 times
# in Essay 100.)




# ----------------------------------------------------------------------------
#                          Viz 4: Document Lengths
# ----------------------------------------------------------------------------
#%% TODO: Our fourth visualization will look at the lengths of each document,
# as well as the average length of each one.
