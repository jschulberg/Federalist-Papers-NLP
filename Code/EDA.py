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

print(fed_papers.head())


# ----------------------------------------------------------------------------
#                             Viz 1: Top 20 Words
# ----------------------------------------------------------------------------
#%% Our first visualization counts the top 20 words across all documents.

# Start by creating a grouped dataframe of our word counts
word_counts = fed_papers.groupby(['word']) \
    .size() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

print(word_counts.head(10))
      
# Before we move on, there are a lot of unnecessary words here! Let's filter
# some of these (stop words) out.
stop_words = ['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 
              'also', 'ought']

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
#                          Viz 2: Document Lengths
# ----------------------------------------------------------------------------
#%% Our second visualization will look at the lengths of each document,
# as well as the average length of each one.
doc_lengths = fed_papers.groupby(['essay']) \
    .size() \
    .reset_index(name = 'length') \
    .sort_values('length', ascending = False) \
    .reset_index(drop = True)

viz2 = sns.violinplot(y = doc_lengths['length'], 
               color = "Slateblue")

# Set our labels
viz2.set(ylabel = 'Number of Words', title = 'Length of Federalist Papers ')
plt.show()

# Save our plot to the Viz folder 
viz2.figure.savefig("Viz/Document Lengths.png")


# ----------------------------------------------------------------------------
#                      Viz 3: Document Lengths by Author
# ----------------------------------------------------------------------------
#%% Our third visualization will look at the lengths of each document,
# as well as the average length of each one, disaggregated by author
doc_lengths = fed_papers.groupby(['essay', 'Author']) \
    .size() \
    .reset_index(name = 'length') \
    .sort_values('length', ascending = False) \
    .reset_index(drop = True)

viz3 = sns.catplot(x = 'Author',
                      y = 'length',
                      data = doc_lengths,
                      hue = 'Author',
                      palette = 'Purples_r')
                      # color = "Slateblue")

# Set our labels
viz3.set(xlabel = 'Author', ylabel = 'Number of Words', title = 'Length of Federalist Papers by Author')
plt.show()

# Save our plot to the Viz folder 
viz3.savefig("Viz/Document Lengths by Author.png")

