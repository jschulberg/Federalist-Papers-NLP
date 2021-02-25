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

#%%
# ----------------------------------------------------------------------------
#                             Data Prep
# ----------------------------------------------------------------------------
# First, let's create a few dataframes that can be used for analysis purposes later on
# Before we move on, there are a lot of unnecessary words here! Let's filter
# some of these (stop words) out.
stop_words = ['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 
              'also', 'ought', 'a', 'the', 'it', 'i', 'upon', 'but', 'if', 'in',
              'this', 'might', 'and', 'us', 'can', 'as', 'to']

fed_nonstop = fed_papers.copy()
fed_nonstop = fed_nonstop[~fed_nonstop['word'].isin(stop_words)]

# It also looks like there are words that should be counted together (i.e. state
# and states). Let's use a lemmatizer to solve this.



# Start by creating a grouped dataframe of our word counts
word_counts = fed_nonstop.groupby(['word']) \
    .size() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

print(word_counts.head(10))

# Now let's create a grouped dataframe that counts the number of documents
# a given word appears in (document frequency). This is important to help us identify
# words that may appear many times but in the same document. A word is considered
# more "important" if it is not just a frequently occuring word within a document, but a word that
# appears across many documents
doc_freq = fed_nonstop[['word', 'Essay']].drop_duplicates() \
    .groupby(['word']) \
    .size() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

print(doc_freq.head(10))


# ----------------------------------------------------------------------------
#                             Viz 1: Top 20 Words
# ----------------------------------------------------------------------------
#%% Our first visualization counts the top 20 words across all documents.
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

# ----------------------------------------------------------------------------
#                         Viz 4-7: Top 10 Words by Author
# ----------------------------------------------------------------------------
#%% TODO: Our fourth  through seventh  visualization constitutes a bar chart of the top 10 words  
# by word count of each author (John Jay, Alexander Hamilton, James Madison, or 
# Unknown).

#Hamilton - Visualization 4------------------------------------------------------

doc_lengths = fed_papers.groupby(['Author','word']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

Hamilton_words = doc_lengths.loc[doc_lengths.Author == 'Hamilton']
Hamilton_top_words = Hamilton_words.head(17)

#Stop Words
stop_words = ['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 
              'also', 'ought', 'the', 'it', 'i', 'upon']

Hamilton_top_words = Hamilton_top_words.copy()
Hamilton_top_words = Hamilton_top_words[~Hamilton_top_words['word'].isin(stop_words)]

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz4 = sns.barplot(x = 'word',
            y = 'count',
            data = Hamilton_top_words,
            palette = "flare")

#Rotate X tick labels
viz4.set_xticklabels(viz5.get_xticklabels(), rotation=45 )

# Set our labels
viz4.set(xlabel='word', ylabel='count', title = 'Hamilton Top Words')
plt.show()

# Save our plot to the Viz folder 
viz4.savefig("Viz/Hamilton Top Words.png")

#John Jay - Visualization 5--------------------------------------------------

doc_lengths = fed_papers.groupby(['Author','word']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

Jay_top = doc_lengths.loc[doc_lengths.Author == 'Jay']
Jay_top_words = Jay_top.head(17)

#Stop Words
stop_words = ['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 
              'also', 'ought', 'the', 'it', 'i', 'upon']

Jay_top_words = Jay_top_words.copy()
Jay_top_words = Jay_top_words[~Jay_top_words['word'].isin(stop_words)]

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz5 = sns.barplot(x = 'word',
            y = 'count',
            data = Jay_top_words,
            palette = "mako")

#Rotate X tick labels
viz5.set_xticklabels(viz6.get_xticklabels(), rotation=45 )

# Set our labels
viz5.set(xlabel='word', ylabel='count', title = 'Jay Top Words')
plt.show()

# Save our plot to the Viz folder 
viz5.savefig("Viz/Jay Top Words.png")

#Madison - Visualization 6-------------------------------------------------------

doc_lengths = fed_papers.groupby(['Author','word']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

Madison_top = doc_lengths.loc[doc_lengths.Author == 'Madison']
Madison_top_words = Madison_top.head(15)

#Stop Words
stop_words = ['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 
              'also', 'ought', 'the', 'it', 'i', 'upon']

Madison_top_words = Madison_top_words.copy()
Madison_top_words = Madison_top_words[~Madison_top_words['word'].isin(stop_words)]

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz6 = sns.barplot(x = 'word',
            y = 'count',
            data = Madison_top_words,
            palette = "coolwarm")

#Rotate X tick labels
viz6.set_xticklabels(viz7.get_xticklabels(), rotation=45 )

# Set our labels
viz6.set(xlabel='word', ylabel='count', title = 'Madison Top Words')
plt.show()

# Save our plot to the Viz folder 
viz6.savefig("Viz/Madison Top Words.png")


#Unknown - Visualization 7-------------------------------------------------------

doc_lengths = fed_papers.groupby(['Author','word']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

Unknown_top = doc_lengths.loc[doc_lengths.Author == 'Unknown']
Unknown_top_words = Unknown_top.head(19)

#Stop Words
stop_words = ['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 
              'also', 'ought', 'the', 'it', 'i', 'upon']

Unknown_top_words = Unknown_top_words.copy()
Unknown_top_words = Unknown_top_words[~Unknown_top_words['word'].isin(stop_words)]

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz7 = sns.barplot(x = 'word',
            y = 'count',
            data = Unknown_top_words,
            palette = "YlOrBr")

#Rotate X tick labels
viz7.set_xticklabels(viz8.get_xticklabels(), rotation=45 )

# Set our labels
viz7.set(xlabel='word', ylabel='count', title = 'Unknown Top Words')
plt.show()


# Save our plot to the Viz folder 
viz7.savefig("Viz/Unknown Top Words.png")

# ----------------------------------------------------------------------------
#                      Viz 8: Word Count vs. Word Frequency
# ----------------------------------------------------------------------------
#%% TODO: Our eighth visualization constitues a scatter plot of all the words
# that could reasonably appear in our dataset, measuring the number of times
# each one appears as well as the number of documents it appears in.

# The hope here is to take a look at what will eventually be the TF-IDF of each
# word: that way we can filter out words that appear many times but only in very
# few documents (i.e. 'Constitution' appears 100 times in total but 95 times
# in Essay 100.)


doc_lengths = fed_papers.groupby(['word','Essay']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

#Looking at which essays government and other words appears more frequently 
word_frequency= doc_lengths.loc[doc_lengths.word.isin([ 'one', 'government', 'people'])]

#If we wanted to look at the percentage in total papers
##word_frequency['% of total papers'] = word_frequency['count'] / word_frequency['count'].sum()

#Remove "Essay" from the Essay columns so we are only left with the number - just so we can fit everything into the graph
word_frequency['Essay'] = pd.to_numeric(word_frequency['Essay'].astype(str).str[5:], errors='coerce')

word_frequency.head()

#Resize the plot
plt.figure(figsize=(10,5))
viz8 = sns.scatterplot(data=word_frequency, x="Essay", y="count", hue = 'word')

#Redo the x axis ticks 

viz8.xaxis.set_major_locator(ticker.MultipleLocator(5))
viz8.xaxis.set_major_formatter(ticker.ScalarFormatter())
#Rotate X tick labels
plt.show()

# Save our plot to the Viz folder 
viz8.savefig("Viz/Word Counts in Essays.png")

####W should look for key words that would are unique to each author. Eventually do TF-IDF code here. 