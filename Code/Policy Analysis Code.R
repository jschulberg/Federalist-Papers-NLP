###########################################################################
###########################################################################

#######               U.S. Policy Fatalities Analysis               #######

###########################################################################
###########################################################################
# In this script, I will analyze the U.S. Policy Fatalities dataset included
# in the data folder of this repository. I will use a variety of exploratory
# and modeling techniques to do answer the following questions:

# How many people have been killed by police over time?
# What is the race/age/gender of these people?
# What are the demographics of the offending police officers?
# What are the demographics of the general population in which the incident
#   occurred? This includes the political affiliations of the mayor, city
#   council, and the presidential candidate that the city voted for in 2016.


###########################################################################
# Set Up ------------------------------------------------------------------
###########################################################################
# Bring in packages
suppressMessages(library("tidyverse")) # Used for data wrangling
suppressMessages(library("tidyr")) # Used for data cleaning
suppressMessages(library("ggplot2")) # Used for visualizations
suppressMessages(library("readxl")) # Used for loading excel files
suppressMessages(library("readr")) # Used for working with files
suppressMessages(library("pander")) # Used for pretty tables
suppressMessages(library("lubridate")) # Used for fixing dates
suppressMessages(library("praise")) # Used for positive reinforcement


# Bring in the data, taking advantage of the project structure
police_data_first <- readr::read_csv(here::here("Data/police_killings_2000-2015.csv"))
police_data_second <- readr::read_csv(here::here("Data/police_killings_2015-2020.csv"))
census_data <- readr::read_csv(here::here("Data/censusStatePopulations2014.csv"))
county_data <- readr::read_csv(here::here("Data/HighSchoolCompletionPovertyRate.csv"))
police_deaths <- readr::read_csv(here::here("Data/police_deaths.csv"))
race_demographics <- readr::read_csv(here::here("Data/race_demographics.csv"),
                                     # define our NAs
                                     na = c("<.01", "N/A"))

# Convert to a tibble, my preferred data structure
(police_data_first <- as_tibble(police_data_first))
(police_data_second <- as_tibble(police_data_second))
(census_data <- as_tibble(census_data))
(county_data <- as_tibble(county_data))
(police_deaths <- as_tibble(police_deaths))
(race_demographics <- as_tibble(race_demographics))

########################################################################
## Clean/Join Data -----------------------------------------------------
########################################################################
# Notice that I've brought in a few datasets. Different datasets include
# different ranges of time for police fatalities, with some overlap. On
# top of that, I have some census, incident, and subject data

# Start by cleaning our 2000-2015 data
police_data_first_date <- police_data_first %>%
  # Use lubridate to fix up the dates
  mutate(Date = mdy(Date),
         # Change the flee column to be a character
         Flee = as.character(if_else(T, "Flee", "Not fleeing")),
         # Make the weapon with which the victim was armed all lowercase
         Armed = tolower(Armed)) %>%
  # Order by Date
  arrange(Date) %>%
  # Make all the column names lower case
  janitor::clean_names()

# Repeat for our other dataset
police_data_second_date <- police_data_second %>%
  # Use lubridate to fix up the dates
  # In general, we need to fix up a bunch of the columns in this other
  # dataset so we can match it up with our first dataset
  mutate(date = mdy(date),
         # Clean up uor gender column
         gender = case_when(
           gender == "M" ~ "Male",
           gender == "F" ~ "Female",
           gender == "NA" ~ "Unknown"
         ),
         race = case_when(
           race == "W" ~ "White",
           race == "B" ~ "Black",
           race == "A" ~ "Asian",
           race == "N" ~ "Native",
           race == "H" ~ "Hispanic",
           race == "O" ~ "Other"
         ),
         manner_of_death = case_when(
           manner_of_death == "shot" ~ "Shot",
           manner_of_death == "shot and Tasered" ~ "Shot and Tasered"
         ),
         # Make the weapon with which the victim was armed all lowercase
         armed = tolower(armed)
         ) %>%
  # Rename our mental illness column
  rename(mental_illness = signs_of_mental_illness) %>%
  # Order by Date
  arrange(date)

# Before cleaning our demographics dataset, let's just find the USA totals
usa_demographics <- race_demographics %>%
  filter(Location == "United States")

# Clean our race demographics dataset
race_dem_clean <- race_demographics %>%
  janitor::clean_names() %>%
  # Get rid of unnecessary columns
  select(-c(total, footnotes)) %>%
  # Let's pivot the dataset so we can join on it properly later
  pivot_longer(cols = "white":"two_or_more_races",
               names_to = "race",
               values_to = "race_percent") %>%
  # Change the name of our races to match what we have above
  mutate(race = case_when(
    race == "white" ~ "White",
    race == "black" ~ "Black",
    race == "asian" ~ "Asian",
    race == "american_indian_alaska_native" ~ "Native",
    race == "hispanic" ~ "Hispanic",
    race == "two_or_more_races" ~ "Other",
    race == "native_hawaiian_other_pacific_islander" ~ "Other"
  )) %>%
  # Impute 0 into any NA's
  tidyr::replace_na(list(race_percent = 0)) %>%
  print()

# What's our range of dates?
range(police_data_first_date$date)
range(police_data_second_date$date)

# How much overlap is there between the two datasets?
table(police_data_first_date$name %in% police_data_second_date$name)

# Split out our names that are not in the first dataset
second_no_overlap <- police_data_second_date[!(police_data_second_date$name %in% police_data_first_date$name), ]

# Before trying to de-dupe, there are a number of names listed as either
# "Name withheld by police" or "TK TK". I'm assuming both of these mean
# "name unknown", so let's pull these out before de-duping
non_dupes <- police_data_first_date %>%
  filter(name == "Name withheld by police" | name == "TK TK") %>%
  print()

# Add in non_dupes from the second dataset
non_dupes <- police_data_second_date %>%
  filter(name == "Name withheld by police" | name == "TK TK") %>%
  bind_rows(non_dupes) %>%
  print()


# Join both datasets together
police_all_dates <- police_data_first_date %>%
  # There are a lot of duplicates, so let's try to remove these
  # We'll look for duplicative names + dates, since there are naturally
  # some repeat common names
  distinct(name, age, .keep_all = T) %>%
  # Bring back our non overlaps from the second dataset plus the rows
  # we specifically pulled out because they didn't have names
  bind_rows(second_no_overlap, non_dupes) %>%
  # Get rid of previous ID column and make our ID row unique
  select(-id) %>%
  mutate(uid = row_number()) %>%
  rename(id = uid) %>%
  distinct() %>%
  print()

police_compare <- police_data_first_date %>%
  # There are a lot of duplicates, so let's try to remove these
  distinct(name, .keep_all = T) %>%
  bind_rows(second_no_overlap) %>%
  # Get rid of previous ID column and make our ID row unique
  select(-id) %>%
  mutate(uid = row_number()) %>%
  rename(id = uid) %>%
  print()

# Where's the overlap?
compare <- police_all_dates[duplicated(police_all_dates$name), ]

### Clean county/census data
# Before joining in our county/census data, let's clean it up a bit
county_cleaned <- county_data %>%
  # Clean up our column names
  janitor::clean_names() %>%
  # Super annoying, but the dataset I imported from Kaggle has an extra word
  # after every single city name, so we need to split on the final space
  mutate(city = gsub(" [^ ]*$", "", city)) %>%
  print()

# We have our full dataset now! Let's bring in a few more columns
# that will eventually help us immensely.
police_joined <- police_all_dates %>%
  rename(stateCode = state) %>%
  # Join in our census data
  left_join(census_data) %>%
  # Join in our county data
  left_join(county_cleaned, by = c("city" = "city", "stateCode" = "geographic_area")) %>%
  # Join in our race demographic data on location and race
  left_join(race_dem_clean, by = c("state" = "location", "race" = "race")) %>%
  # Make our numeric columns not characters
  mutate(percent_completed_hs = as.numeric(percent_completed_hs),
         poverty_rate = as.numeric(poverty_rate)) %>%
  print()



########################################################################
## Analysis Time  ------------------------------------------------------
########################################################################
# Let's start by checking out the number of fatalities over time.

# Visualization time
police_joined %>%
  # Create our month and year variables
  mutate(month = month(date),
         year = year(date)) %>%
  # Start by grouping by month AND year of dates
  group_by(month, year) %>%
  # Count up our sums
  summarise(fatalities = n()) %>%
  # Bring our dates back together
  mutate(date = as.Date(paste(year, month, "01", sep = "-"))) %>%
  # run our visualization
  ggplot(aes(x = date, y = fatalities)) +
    # Let's make it a column graph and change the color
    geom_line(color = "slateblue") +
    # Change the theme to classic
    theme_classic() +
    # Let's change the names of the axes and title
    xlab("") +
    ylab("Number of Fatalities") +
    labs(title = "Number of Police-caused Fatalities over Time",
         subtitle = "Data runs from 2000-2020",
         caption = "Data is gathered from the Washington Post at\nhttps://github.com/washingtonpost/data-police-shootings") +
    # format our title and subtitle
    theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
          plot.subtitle = element_text(hjust = 0, color = "slateblue2", size = 10),
          plot.caption = element_text(color = "dark gray", face = "italic", size = 10))

# What's the breakout over time by race? Upon running this, I decided to
# pull in demographic information because it unfairly looked like white
# people were the primary target of police killings, which is true but is
# not proportionate to their population in the U.S.

# To do this, first we'll pivot our USA total demographic data
usa_dem_pivoted <- race_dem_clean %>%
  filter(location == "United States") %>%
  group_by(race) %>%
  # Do this to combine our two Other categories which are being counted differently
  summarise(race_percent = sum(race_percent)) %>%
  print()

# Ensure that the sum of each percentage adds up to 1
if (sum(usa_dem_pivoted$race_percent) == 1) {
  praise::praise()
}

# Save our dataset
race_killings <- police_joined %>%
  # Take out our null values
  filter(!is.na(race)) %>%
  # Start by grouping by state
  group_by(race) %>%
  # Count up our sums
  summarise(fatalities = n()) %>%
  # Bring in our country populations
  left_join(usa_dem_pivoted, by = "race") %>%
  # Calculate the proportion
  mutate(fatalities_perc = round(100*fatalities/sum(fatalities), 0),
         race_percent = 100*race_percent) %>%
  print()

race_killings %>%
  ggplot(aes(x = 100*race_percent, y = fatalities), label = race) +
  # Let's make it a column graph and change the color/transparency
  geom_point(color = "slateblue") +
  geom_text(aes(label = if_else(fatalities > 1000, race, "")), nudge_y = 400) +
  # Change the theme to classic
  theme_classic() +
  # Let's change the names of the axes and title
  xlab("Population (%)") +
  ylab("Number of Fatalities") +
  labs(title = "Number of Police-caused Fatalities by Race\nand Percent of U.S. Population",
       subtitle = "Data runs from 2000-2020",
       caption = "Data is gathered from the Washington Post at\nhttps://github.com/washingtonpost/data-police-shootings") +
  # format our title and subtitle
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(hjust = 0, color = "slateblue2", size = 10),
        plot.caption = element_text(color = "dark gray", size = 10, face = "italic"))

# Interestingly enough, more people with a white ethnicity have been killed
# by police than black people, at least according to the dataset. This seems
# to surprise me, especially with everything going on in the news lately.
# At least for now, I'll have to check on this to see if this data holds water.

# What's the breakout over time by race?
police_joined %>%
  # Create our month and year variables
  mutate(month = month(date),
         year = year(date)) %>%
  # Start by grouping by month AND year of dates
  group_by(month, year, race) %>%
  # Count up our sums
  summarise(fatalities = n()) %>%
  # Bring our dates back together
  mutate(date = as.Date(paste(year, month, "01", sep = "-"))) %>%
  # run our visualization
  ggplot(aes(x = date, y = fatalities)) +
  # Let's make it a column graph and change the color/transparency
  # geom_line(aes(color = race), lwd = .5, alpha = .7) +
  geom_line(color = "slateblue") +
  # Separate vizzes for each race
  facet_wrap(~ race) +
  # Change the theme to classic
  theme_classic() +
  # Let's change the names of the axes and title
  xlab("") +
  ylab("Number of Fatalities") +
  labs(title = "Number of Police-caused Fatalities over Time",
       subtitle = "Data runs from 2000-2020",
       caption = "Data is gathered from the Washington Post at\nhttps://github.com/washingtonpost/data-police-shootings") +
  # format our title and subtitle
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(hjust = 0, color = "slateblue2", size = 10),
        plot.caption = element_text(color = "dark gray", size = 10, face = "italic"))

# One thing I've noticed here is the spike in killings of white individuals
# in 2015. As a note, I merged two disparate datasets, one that runs from 2000-
# 2015 and the other that runs from 2015-2020. It's possible that something's
# amiss with the latter. The other facet that may be skewing the data is that
# 31% of the data on race has not been collected (i.e. is null). If most of
# these individuals were black, that would constitute another 3800 individuals.


# How old are people likely to be who are killed by police officers?
police_joined %>%
  # Take out any null values
  filter(!is.na(age)) %>%
  # run our visualization
  ggplot(aes(x = age, fill = race)) +
  # Let's make it a histogram with 25 bins, a slateblue filling,
  # and a white border
  # geom_histogram(bins = 25, color = "white") +
  geom_histogram(bins = 25, fill = "slateblue", color = "white") +
  # Change the theme to classic
  theme_classic() +
  # Let's change the names of the axes and title
  xlab("Age") +
  ylab("Number of Fatalities") +
  labs(title = "Age of Individuals Killed by Police",
       subtitle = "Data runs from 2000-2020",
       caption = "Data is gathered from the Washington Post at\nhttps://github.com/washingtonpost/data-police-shootings") +
  # format our title and subtitle
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(hjust = 0, color = "slateblue2", size = 10),
        plot.caption = element_text(color = "dark gray", size = 10, face = "italic"))

# Which states had the most fatalities?
police_joined %>%
  # Start by grouping by state
  group_by(state) %>%
  # Count up our sums
  summarise(fatalities = n()) %>%
  top_n(10, fatalities) %>%
  ggplot(aes(x = reorder(state, fatalities), y = fatalities), label = fatalities) +
  # Let's make it a column graph and change the color/transparency
  geom_col(fill = "slateblue") +
  # Add a label by recreating our data build from earlier
  geom_label(data = police_joined %>%
               # Start by grouping by state
               group_by(state) %>%
               # Count up our sums
               summarise(fatalities = n()) %>%
               top_n(10, fatalities),
             aes(label = fatalities),
             size = 2.5) +
  # Change the theme to classic
  theme_classic() +
  # Let's change the names of the axes and title
  xlab("") +
  ylab("Number of Fatalities") +
  labs(title = "Number of Police-caused Fatalities by State",
       subtitle = "Data runs from 2000-2020",
       caption = "Data is gathered from the Washington Post at\nhttps://github.com/washingtonpost/data-police-shootings") +
  # format our title and subtitle
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(hjust = 0, color = "slateblue2", size = 10),
        plot.caption = element_text(color = "dark gray", size = 10, face = "italic")) +
  # flip the axes
  coord_flip()

# This isn't helpful since the populations are so different. For example,
# some of our biggest states by population, like California and Texas, are
# showing up here. I have a hunch that if we normalize by the state population
# using some of our census data from 2014, we'll get a better sense of fatalities.


# Let's normalize by state population
police_joined %>%
  # Start by grouping by state
  group_by(state) %>%
  # Count up our sums
  summarise(fatalities = n()) %>%
  # To normalize by state population, let's rejoin this data in
  left_join(census_data, by = "state") %>%
  # Create our normalized data and multiply by 1,000,000 so we get the number
  # of fatalities per 1,000,000 people
  mutate(fatalities_normalized = 1000000*fatalities/popEst2014) %>%
  top_n(10, fatalities_normalized) %>%
  ggplot(aes(x = reorder(state, fatalities_normalized), y = fatalities_normalized), label = fatalities_normalized) +
  geom_col(fill = "slateblue") +
  # Add a label by recreating our data build from earlier
  geom_label(data = police_joined %>%
               # Start by grouping by state
               group_by(state) %>%
               # Count up our sums
               summarise(fatalities = n()) %>%
               # To normalize by state population, let's rejoin this data in
               left_join(census_data, by = "state") %>%
               # Create our normalized data and multiply by 1,000,000 so we get the number
               # of fatalities per 1,000,000 people
               mutate(fatalities_normalized = 1000000*fatalities/popEst2014) %>%
               top_n(10, fatalities_normalized),
             aes(label = round(fatalities_normalized, 0)),
             size = 2.5) +  # Change the theme to classic
  theme_classic() +
  # Let's change the names of the axes and title
  xlab("") +
  ylab("Number of Fatalities*") +
  labs(title = "Number of Police-caused Fatalities by State",
       subtitle = "*Fatalities per 1,000,000 population",
       caption = "Data is gathered from the Washington Post at\nhttps://github.com/washingtonpost/data-police-shootings") +
  # format our title and subtitle
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(hjust = 0, color = "slateblue2", size = 10),
        plot.caption = element_text(color = "dark gray", size = 10, face = "italic")) +
  # flip the axes
  coord_flip()

# Super interesting! We now have a lot of states in the new visualization,
# showing that just because a state, like Texas, has a lot of police-caused
# fatalities, it doesn't mean that it's as high as other states proportionately
# to state population


########################################################################
## High School Completion Rate Analysis  -------------------------------
########################################################################
# I'd like to analyze the number of shootings against the average city
# population which completed high school. My hypothesis here is that there
# will be more shootings in cities with low high school completion rates.

# The first thing to do is to fill out our data a bit more. There is a lot
# of data that didn't match. What I plan to do for imputation is find the
# average percent_completed_highschool for each state and impute that.

# Start by computing the average high school completion rates by state
hs_averages <- police_joined %>%
  group_by(state) %>%
  summarise(percent_completed_hsavg = mean(percent_completed_hs, na.rm = T))

# Split our dataset on those with NA values and those without
police_hs_na <- police_joined %>%
  # Only keep our nas for the first one
  filter(is.na(percent_completed_hs)) %>%
  left_join(hs_averages, by = "state") %>%
  # now get rid of the percent_completed_hs field and rename the other one to match
  select(-percent_completed_hs) %>%
  rename(percent_completed_hs = percent_completed_hsavg) %>%
  print()

police_hs_nona <- police_joined %>%
  # Only keep our non-nas for the second one
  filter(!(is.na(percent_completed_hs)))

# Now bring both datasets back together
police_hs <- bind_rows(police_hs_na, police_hs_nona)

# If the number of rows we ended with doesn't match the number of rows
# we started with, or if there are still any nulls in the percent_completed_hs
# field, throw an error message.
if (nrow(police_hs) != nrow(police_joined) | any(is.na(police_hs$percent_completed_hs))) {
  "ERROR! SOMETHING'S UP WITH THE police_hs dataframe."
} else {
  cat(praise(), "Let's keep going.")
}

# Let's look at a boxplot of the data
ggplot(police_hs, aes(y = percent_completed_hs)) +
  geom_boxplot(outlier.colour="slateblue4",
               outlier.size=2,
               color = "slateblue3") +
  theme_classic() +
  # Let's change the names of the axes and title
  labs(title = "High School Completion Rate by Geographic Area",
       subtitle = "Data is collected from 2014 Census data.",
       caption = "") +
  ylab("Percentage (%)") +
  # Center the title and format the subtitle/caption
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(color = "slateblue1", size = 10),
        plot.caption = element_text(hjust = 1, face = "italic", color = "dark gray"),
        # remove the x axis labels because they don't mean much for us
        axis.text.x = element_blank()) +
  # I thought the boxplot was too thick, so let's make it a little skinnier
  scale_x_discrete(limits=c("-.1", ".1"))

# It looks like most of the areas have pretty high high school completion
# rates, with a median of
median(police_hs$percent_completed_hs)
# and an average of
mean(police_hs$percent_completed_hs)

### Correlation
# Let's now look to see if there's any correlation between high school
# completion rate and number of police-related fatalities.
police_hs %>%
  # Start by grouping by state
  group_by(state) %>%
  # Count up our sums
  summarise(fatalities = n(),
            hs_completionavg = mean(percent_completed_hs)) %>%
  ggplot(aes(x = hs_completionavg, y = fatalities)) +
  # Make it a scatter plot
  geom_point(color = "slateblue", alpha = .8) +
  geom_text(aes(label = state), # label by state
            color = "slateblue", # Make our color match
            size = 3, # shrink the size
            alpha = .8, # add some transparency
            check_overlap = T, # avoid overlabelling
            nudge_y = 150) + # nudge the text a bit off center
  theme_classic() +
  # Let's change the names of the axes and title
  labs(title = "Police-caused Fatalities by High School Completion Rate",
       subtitle = "Data is broken out by state and uses 2014 Census data.",
       caption = "") +
  xlab("High School Completion Rate (%)") +
  ylab("Police-caused Fatalities") +
  # Center the title and format the subtitle/caption
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(color = "slateblue1", size = 10),
        plot.caption = element_text(hjust = 1, face = "italic", color = "dark gray"))


# Again, this doesn't account for normalizing our data by population. Let's
# see how that changes things
police_hs %>%
  # Start by grouping by state
  group_by(state) %>%
  # Count up our sums
  summarise(fatalities = n(),
            hs_completionavg = mean(percent_completed_hs)) %>%
  # To normalize by state population, let's rejoin this data in
  left_join(census_data, by = "state") %>%
  # Create our normalized data and multiply by 1,000,000 so we get the number
  # of fatalities per 1,000,000 people
  mutate(fatalities_normalized = 1000000*fatalities/popEst2014) %>%
  ggplot(aes(x = hs_completionavg, y = fatalities_normalized)) +
  # Make it a scatter plot
  geom_point(color = "slateblue", alpha = 1) +
  theme_classic() +
  # Let's change the names of the axes and title
  labs(title = "Normalized Police-caused Fatalities\nby High School Completion Rate",
       subtitle = "Police-caused fatalities are per 1,000,000 population\nusing 2014 Census data.",
       caption = "*per 1,000,000 population") +
  xlab("High School Completion Rate (%)") +
  ylab("Police-caused Fatalities*") +
  # Center the title and format the subtitle/caption
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(color = "slateblue1", size = 10),
        plot.caption = element_text(hjust = 1, face = "italic", color = "dark gray"))

# Between both of these graphs, I can't see any correlation in the data. It's
# reassuring to know that police-caused fatalities would not drastically increase
# in areas with lower high school completion rates.

#What's the actual correlation?
police_hs %>%
  # Start by grouping by state
  group_by(state) %>%
  # Count up our sums
  summarise(fatalities = n(),
            hs_completionavg = mean(percent_completed_hs)) %>%
  # To normalize by state population, let's rejoin this data in
  left_join(census_data) %>%
  # Create our normalized data and multiply by 1,000,000 so we get the number
  # of fatalities per 1,000,000 people
  mutate(fatalities_normalized = 1000000*fatalities/popEst2014) %>%
  select(-state, -stateCode) %>%
  cor() %>%
  pander()

# From this we can see that the correlation is highest with population,
# which makes sense. Once normalizing for population, the correlation
# drops to -.025, which is highly uncorrelated. Overall, I would say
# that there is no correlation between fatalities and high school
# completion rate.

########################################################################
## Poverty Rate Analysis  ----------------------------------------------
########################################################################
# I'd like to analyze the number of shootings against the average city
# poverty rate. My hypothesis here is that there will be more shootings
# in cities with high poverty rates, although my hypothesis earlier
# was debunked, so we will see!

# First, let's see if there's correlation between poverty rate and number
# of police-caused killings
police_hs %>%
  # First, filter out all of our NAs, which is 90% of our data
  filter(!is.na(poverty_rate)) %>%
  group_by(state) %>%
  summarise(fatalities = n(),
            avg_poverty = mean(poverty_rate)) %>%
  ungroup() %>%
  left_join(census_data) %>%
  mutate(fatalities_normalized = 1000000*fatalities/popEst2014) %>%
  # Get rid of unneeded columns
  select(-stateCode, -state, - popEst2014) %>%
  cor() %>%
  pander()

# There's a bit of correlation here (-.1845) and more than we saw with
# high school completion rate. The interesting thing to note is the negative
# in the correlation. This means that as poverty rates generally increase,
# the number of police-caused fatalities per 1,000,000 population goes down,
# which is exactly opposite of what I originally thought. Again, the negative
# correlation is rather weak, so I would not read too much into it.
police_hs %>%
  # First, filter out all of our NAs, which is 90% of our data
  filter(!is.na(poverty_rate)) %>%
  group_by(state) %>%
  summarise(fatalities = n(),
            avg_poverty = mean(poverty_rate)) %>%
  ungroup() %>%
  left_join(census_data) %>%
  mutate(fatalities_normalized = 1000000*fatalities/popEst2014) %>%
  ggplot(aes(x = avg_poverty, y = fatalities_normalized)) +
  geom_point(color = "slateblue", alpha = 1) +
  geom_smooth(method = "lm", se = F, color = "black") +
  theme_classic() +
  # Let's change the names of the axes and title
  labs(title = "Normalized Police-caused Fatalities\nby Poverty Rate",
       subtitle = "Police-caused fatalities are per 1,000,000 population\nusing 2014 Census data.",
       caption = "*per 1,000,000 population") +
  xlab("Poverty Rate (%)") +
  ylab("Police-caused Fatalities*") +
  # Center the title and format the subtitle/caption
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(color = "slateblue1", size = 10),
        plot.caption = element_text(hjust = 1, face = "italic", color = "dark gray"))


########################################################################
## Mental Health Analysis  ---------------------------------------------
########################################################################
# One thing that I often don't hear covered in the media is the presence
# of mental health issues amongst victims of police-caused fatalities.
# Let's take a look at what the data shows.

# Let's see where our overlap lies by creating a donut chart
donut_inputs <- round(prop.table(table(police_hs$mental_illness)), 2)*100
donut_inputs <- donut_inputs %>%
  as.data.frame() %>%
  # Create the cumulative percentages, which represent the top of each rectangle
  mutate(ymax = cumsum(Freq)) %>%
  # Create the bottom of each rectangle
  mutate(ymin = c(0, head(cumsum(Freq), n = -1))) %>%
  # Rename Var1 to Source
  rename("Source" = "Var1") %>%
  # Compute label position
  mutate(labelPosition = (ymax + ymin) / 2) %>%
  # Compute what our label will display
  mutate(label = paste0(Freq, "%", sep = ""),
         Source = if_else(Source == TRUE, "Mental Illness", "No Mental Illness"))

# Donut Chart
ggplot(donut_inputs, aes(ymax = ymax, ymin = ymin, xmax = 4, xmin = 3, fill = Source)) +
  geom_rect() +
  # A donut chart is just a rectangle charge pivoted to a polar-coordinate plane
  coord_polar(theta = "y") +
  # Add labels in
  geom_label(x = 3.5, aes(y = labelPosition, label = label), size = 3) +
  xlim(c(-1, 4)) +
  # Take out all the plot background which is cluttering the view
  theme_void() +
  # Let's change the names of the axes and title
  labs(title = "Mental Illness among Victims of\nPolice-caused Fatalities",
       subtitle = "Data runs from 2000-2020.",
       caption = "Data taken from Kaggle and Washington Post") +
  # Center the title and format the subtitle/caption
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(color = "slateblue1", size = 10),
        plot.caption = element_text(hjust = 1, face = "italic", color = "dark gray")) +
  # Change the colors we're working with
  scale_fill_manual(values = c("slateblue", "gray")) +
  # add text to the center of the donut, making the first one bold
  annotate(geom = 'text', x = -.25, y = 0, label = "Number of Fatalities:", size = 4.5, fontface = 2) +
  # and the second one normal font
  annotate("text", x = -1, y = 0, label = nrow(police_hs), size = 4)



# How does the prevalence of mental illness amongst victims of police-caused
# fatalities change over time?
police_hs %>%
  # Create our month and year variables
  mutate(year = year(date)) %>%
  # Start by grouping by month AND year of dates
  group_by(year, mental_illness) %>%
  # Count up our sums
  summarise(fatalities = n()) %>%
  mutate(mental_ill_prop = fatalities/sum(fatalities)) %>%
  # Create our proportions and bring our dates back together
  mutate(date = as.Date(paste(year, "01", "01", sep = "-"))) %>%
  # run our visualization
  ggplot(aes(x = date, y = mental_ill_prop, fill = mental_illness)) +
  # Let's make it a column graph and change the color
  geom_area(alpha = .9, color = "gray") +
  # Change the theme to classic
  theme_classic() +
  # Change the colors we're working with
  scale_fill_manual(values = c("gray", "slateblue")) +
  # Let's change the names of the axes and title
  xlab("") +
  ylab("Percentage of Fatalities (%)") +
  labs(title = "Number of Police-caused Fatalities over Time",
       subtitle = "Data runs from 2000-2020",
       caption = "Data is gathered from the Washington Post at\nhttps://github.com/washingtonpost/data-police-shootings") +
  # format our title and subtitle
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(hjust = 0, color = "slateblue2", size = 10),
        plot.caption = element_text(color = "dark gray", face = "italic", size = 10))

# Interestingly enough, it seems that there is a slight upward trend in
# the percentage of victims of police-caused fatalities who were struggling
# with mental health at the time. While the growth is not alarming, it
# does suggest that better training may be needed for police officers dealing
# with situations involving individuals with mental health issues.

police_hs %>%
  # Create our month and year variables
  mutate(month = month(date),
         year = year(date)) %>%
  # Only include victims with mental health issues
  filter(mental_illness == T) %>%
  # Start by grouping by month AND year of dates
  group_by(month, year) %>%
  # Count up our sums
  summarise(fatalities = n()) %>%
  # Calculate our proportions
  mutate(mental_ill_prop = fatalities/sum(fatalities)) %>%
  # Create our proportions and bring our dates back together
  mutate(date = as.Date(paste(year, month, "01", sep = "-"))) %>%
  # run our visualization
  ggplot(aes(x = date, y = mental_ill_prop)) +
  # Let's make it a column graph and change the color
  geom_line(color = "slateblue") +
  geom_smooth(method = "lm") +
  # Change the theme to classic
  theme_classic() +
  # Let's change the names of the axes and title
  xlab("") +
  ylab("Percentage of Fatalities (%)") +
  labs(title = "Number of Police-caused Fatalities over Time",
       subtitle = "Data runs from 2000-2020",
       caption = "Data is gathered from the Washington Post at\nhttps://github.com/washingtonpost/data-police-shootings") +
  # format our title and subtitle
  theme(plot.title = element_text(hjust = 0, color = "slateblue4"),
        plot.subtitle = element_text(hjust = 0, color = "slateblue2", size = 10),
        plot.caption = element_text(color = "dark gray", face = "italic", size = 10))

# Above we can see just the percentage of fatalities caused by police officers
# of individuals struggling with mental health as a proportion of all those
# killed by police officers. It looks like there's a strong case to say that
# every year, those dying at the hands of police officers with mental health
# issues is steadily increasing. By how much?

mental_health_victims <- police_hs %>%
  # Create our year variables
  mutate(year = year(date)) %>%
  # Only include victims with mental health issues
  filter(mental_illness == T) %>%
  # Start by grouping by year of dates
  group_by(year) %>%
  # Count up our sums
  summarise(fatalities = n()) %>%
  # Calculate our proportions
  mutate(mental_ill_prop = fatalities/sum(fatalities)) %>%
  # Create our proportions and bring our dates back together
  mutate(date = as.Date(paste(year, "01", "01", sep = "-")))

men_health_linreg <- lm(mental_ill_prop ~ date, data = mental_health_victims)
pander(summary(men_health_linreg))

# Although it's relatively small, we can see a strong trend between date
# and the proportion of victims who had mental health issues.
