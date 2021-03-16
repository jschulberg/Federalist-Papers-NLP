##EDA 

# ----------------------------------------------------------------------------
#                                Purpose
# ----------------------------------------------------------------------------
# The purpose of this script is to conduct basic exploratory data analysis on 
# the 85 Essays in the Federalist Papers.



#%% Load Data and import packages
import pandas as pd
import numpy as np
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
# Before we move on, there are a lot of unnecessary Words here! Let's filter
# some of these (stop Words) out.
stop_Words = ['would', 'may', 'yet', 'must', 'shall', 'not', 'still', 'let', 
              'also', 'ought', 'a', 'the', 'it', 'i', 'upon', 'but', 'if', 'in',
              'this', 'might', 'and', 'us', 'can', 'as', 'to', 'could']

fed_nonstop = fed_papers.copy()
fed_nonstop = fed_nonstop[~fed_nonstop['Word'].isin(stop_Words)]

# It also looks like there are Words that should be counted together (i.e. state
# and states). Let's use a lemmatizer to solve this.



# Start by creating a grouped dataframe of our Word counts
Word_counts = fed_nonstop.groupby(['Word']) \
    .size() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

print(Word_counts.head(10))


# ----------------------------------------------------------------------------
#                             Viz 1: Top 20 Words
# ----------------------------------------------------------------------------
#%% Our first visualization counts the top 20 Words across all documents.
# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz1 = sns.barplot(x = 'count',
            y = 'Word',
            data = Word_counts[:20],
            palette = "Purples_r")

# Set our labels
viz1.set(xlabel='Number of Appearances', ylabel='Word', title = 'Word Counts across all Federalist Papers')
plt.show()

# Save our plot to the Viz folder 
viz1.figure.savefig("Viz/Top_20_Words.png")


# ----------------------------------------------------------------------------
#                          Viz 2: Document Lengths
# ----------------------------------------------------------------------------
#%% Our second visualization will look at the lengths of each document,
# as well as the average length of each one.
doc_lengths = fed_nonstop.groupby(['Essay']) \
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
viz2.figure.savefig("Viz/Document_Lengths.png")


# ----------------------------------------------------------------------------
#                      Viz 3: Document Lengths by Author
# ----------------------------------------------------------------------------
#%% Our third visualization will look at the lengths of each document,
# as well as the average length of each one, disaggregated by author
doc_lengths = fed_papers.groupby(['Essay', 'Author']) \
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
viz3.savefig("Viz/Document_Lengths_by_Author.png")

# ----------------------------------------------------------------------------
#                         Viz 4-7: Top 10 Words by Author
# ----------------------------------------------------------------------------
#%% Our fourth  through seventh  visualization constitutes a bar chart of the top 10 Words  
# by Word count of each author (John Jay, Alexander Hamilton, James Madison, or 
# Unknown).

#Hamilton - Visualization 4------------------------------------------------------

doc_lengths = fed_nonstop.groupby(['Author','Word']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

Hamilton_Words = doc_lengths.loc[doc_lengths.Author == 'Hamilton']
Hamilton_top_Words = Hamilton_Words.head(17)


Hamilton_top_Words = Hamilton_top_Words.copy()
Hamilton_top_Words = Hamilton_top_Words[~Hamilton_top_Words['Word'].isin(stop_Words)]

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz4 = sns.barplot(x = 'Word',
            y = 'count',
            data = Hamilton_top_Words,
            palette = "Purples_r")

#Rotate X tick labels
viz4.set_xticklabels(viz4.get_xticklabels(), rotation=45 )

# Set our labels
viz4.set(xlabel='Word', ylabel='count', title = 'Hamilton Top Words')
plt.show()

# Save our plot to the Viz folder 
viz4.figure.savefig("Viz/Hamilton_Top_Words.png")


#%% John Jay - Visualization 5--------------------------------------------------

doc_lengths = fed_nonstop.groupby(['Author','Word']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

Jay_top = doc_lengths.loc[doc_lengths.Author == 'Jay']
Jay_top_Words = Jay_top.head(17)



Jay_top_Words = Jay_top_Words.copy()
Jay_top_Words = Jay_top_Words[~Jay_top_Words['Word'].isin(stop_Words)]

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz5 = sns.barplot(x = 'Word',
            y = 'count',
            data = Jay_top_Words,
            palette = "Purples_r")

#Rotate X tick labels
viz5.set_xticklabels(viz5.get_xticklabels(), rotation=45 )

# Set our labels
viz5.set(xlabel='Word', ylabel='count', title = 'Jay Top Words')
plt.show()

# Save our plot to the Viz folder 
viz5.figure.savefig("Viz/Jay_Top_Words.png")


#%% Madison - Visualization 6-------------------------------------------------------

doc_lengths = fed_nonstop.groupby(['Author','Word']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

Madison_top = doc_lengths.loc[doc_lengths.Author == 'Madison']
Madison_top_Words = Madison_top.head(15)

Madison_top_Words = Madison_top_Words.copy()
Madison_top_Words = Madison_top_Words[~Madison_top_Words['Word'].isin(stop_Words)]

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz6 = sns.barplot(x = 'Word',
            y = 'count',
            data = Madison_top_Words,
            palette = "Purples_r")

#Rotate X tick labels
viz6.set_xticklabels(viz6.get_xticklabels(), rotation=45 )

# Set our labels
viz6.set(xlabel='Word', ylabel='count', title = 'Madison Top Words')
plt.show()

# Save our plot to the Viz folder 
viz6.figure.savefig("Viz/Madison_Top_Words.png")


#%% Unknown - Visualization 7-------------------------------------------------------

doc_lengths = fed_nonstop.groupby(['Author','Word']) \
    .Essay.count() \
    .reset_index(name = 'count') \
    .sort_values('count', ascending = False) \
    .reset_index(drop = True)

Unknown_top = doc_lengths.loc[doc_lengths.Author == 'Unknown']
Unknown_top_Words = Unknown_top.head(19)


Unknown_top_Words = Unknown_top_Words.copy()
Unknown_top_Words = Unknown_top_Words[~Unknown_top_Words['Word'].isin(stop_Words)]

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz7 = sns.barplot(x = 'Word',
            y = 'count',
            data = Unknown_top_Words,
            palette = "Purples_r")

#Rotate X tick labels
viz7.set_xticklabels(viz7.get_xticklabels(), rotation=45 )

# Set our labels
viz7.set(xlabel='Word', ylabel='count', title = 'Unknown Top Words')
plt.show()


# Save our plot to the Viz folder 
viz7.figure.savefig("Viz/Unknown_Top_Words.png")

# ----------------------------------------------------------------------------
#                      Viz 8: Word Count vs. Word Frequency
# ----------------------------------------------------------------------------
#%% Our eighth visualization constitues a scatter plot of all the Words
# that could reasonably appear in our dataset, measuring the number of times
# each one appears as well as the number of documents it appears in.

# The hope here is to take a look at what will eventually be the TF-IDF of each
# Word: that way we can filter out Words that appear many times but only in very
# few documents (i.e. 'Constitution' appears 100 times in total but 95 times
# in Essay 100.)

# Now let's create a grouped dataframe that counts the number of documents
# a given Word appears in (document frequency). This is important to help us identify
# Words that may appear many times but in the same document. A Word is considered
# more "important" if it is not just a frequently occuring Word within a document, but a Word that
# appears across many documents

doc_lengths = fed_nonstop[['Word', 'Essay']].drop_duplicates() \
    .groupby(['Word']) \
    .size() \
    .reset_index(name = 'doc_count') \
    .sort_values('doc_count', ascending = False) \
    .reset_index(drop = True)
    
    
#Looking at which Essays government and other Words appears more frequently 
# Word_frequency = doc_lengths.loc[doc_lengths.Word.isin([ 'one', 'government', 'people'])]

#If we wanted to look at the percentage in total papers
##Word_frequency['% of total papers'] = Word_frequency['count'] / Word_frequency['count'].sum()

#Remove "Essay" from the Essay columns so we are only left with the number - just so we can fit everything into the graph
# Word_frequency['Essay'] = pd.to_numeric(Word_frequency['Essay'].astype(str).str[5:], errors='coerce')

# Word_frequency.head()

merged_counts = pd.merge(Word_counts, 
                         doc_lengths, 
                         left_on = 'Word', 
                         right_on = 'Word',
                         how = 'inner')

# Resize the plot
plt.figure(figsize=(10,5))
viz8 = sns.scatterplot(data = merged_counts, 
                       x = "doc_count", 
                       y = "count",
                       alpha = .3,
                       color = "slateblue")

# Set our labels
viz8.set(ylabel = 'Word Frequency', 
         xlabel = 'Document Frequency',
         title = 'Word Frequency by Document Frequency')

plt.show()

# Save our plot to the Viz folder 
viz8.figure.savefig("Viz/Word_Frequency_by_Document_Frequency.png")


#%%
# ----------------------------------------------------------------------------
#                                   TF-IDF
# ----------------------------------------------------------------------------
# Building on our analysis above, we'll now look into the TF-IDF for each Word.
####W should look for key Words that would are unique to each author. Eventually do TF-IDF code here. 

# Let's start by calculating term frequency. While we've mostly been looking at the Word counts
# across all documents, for term frequency, we care about the propoortion of times
# the Word appears in a given document. Ex: If a sentence is 10 Words long and 
# 'constitution' appears 3 times, its term frequency is .3 (30%).
fed_analysis = merged_counts.copy()

# Calculate the length of each Essay
doc_lengths = fed_nonstop.groupby(['Essay']) \
    .size() \
    .reset_index(name = 'doc_length') \
    .reset_index(drop = True)
    
# Now let's figure out how many times a Word appears in a given Essay
Word_frequency = fed_nonstop.groupby(['Word', 'Essay']) \
    .size() \
    .reset_index(name = 'Word_freq') \
    .sort_values('Word') \
    .reset_index(drop = True)

# With these two dataframes, we can bring them together to calculate our tf score
merged_tf = pd.merge(Word_frequency, 
                     doc_lengths, 
                     left_on = 'Essay',
                     right_on = 'Essay',
                     how = 'inner')

merged_tf['tf'] = merged_tf['Word_freq'] / merged_tf['doc_length']


# We can pull the inverse document frequency from our merged_counts dataframe above
fed_analysis['idf'] = np.log(85 / fed_analysis['doc_count'])

# Let's merge these (again) into one big dataframe
tf_idf_df = pd.merge(merged_tf,
                     fed_analysis,
                     left_on = 'Word',
                     right_on = 'Word',
                     how = 'inner')

tf_idf_df['tf_idf'] = tf_idf_df['tf'] * tf_idf_df['idf']


#%%
# ----------------------------------------------------------------------------
#                                Viz 9: Top TF-IDF
# ----------------------------------------------------------------------------
# Let's see which Words have the highest TF-IDF scores by author. This will
# help us identify the style of each author by looking at the Words that they
# use most uniquely.

authors = fed_nonstop[['Essay', 'Author']].drop_duplicates()

merged_df = tf_idf_df.merge(authors,
                            left_on = 'Essay',
                            right_on = 'Essay')

authors_tf = merged_df.groupby(['tf_idf', 'Author', 'Word']) \
    .size() \
    .reset_index(name = 'tfidf') \
    .sort_values('tf_idf', ascending = False) \
    .reset_index(drop = True)
    
# Find our top 10 Words for each author
authors_top_tf = authors_tf.groupby('Author')['tf_idf'] \
    .nlargest(10, keep = 'first') \
    .reset_index(name = "tf_idf")

# Unfortunately this drops the actual Word, so let's merge it back on
authors_top_tf = authors_top_tf.merge(authors_tf,
                                      left_on = ['Author', 'tf_idf'],
                                      right_on = ['Author', 'tf_idf'])

# Set the theme
sns.set_style('white')
sns.set_context('notebook')

# Build the visualization
viz9 = sns.FacetGrid(authors_top_tf, 
                     col = "Author", 
                     sharex = False, 
                     sharey = False)
viz9.map(sns.barplot, "tf_idf", "Word")


# Set our labels
# Set our labels
viz9.set(xlabel='tf_idf', ylabel='Word')
plt.show()

# Save our plot to the Viz folder 
viz9.savefig("Viz/Top_TF_IDF.png")

