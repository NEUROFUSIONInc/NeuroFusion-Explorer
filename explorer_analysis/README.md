Fusion Explorer Analysis

A set of scripts and resources for analyzing data recorded on [Fusion](https://usefusion.app)

eeg.py is a file that helps one analyze the data they have collected. Below is a description of what each of the functions in it do.

rnOccur: finds the nth occurrence of v in arr

extractBundledEEG: init: creates files to separte useful information by tags such as event or description addMeta: helps make data more understandable by adding categorical information to the csv file prune: filters out all events that do not have any eeg data to support them extractByID: finds the location of all the json files that have a specific event extractByTags: finds the location of all the json fiels that have a specific tag mergeTagsWithRegex: consolidates data by combining data that have the same tag mergeCategories: creates a new category with all the categories passed in

load_data: loads the eeg data to be used

get_signal_quality_summary: counts how many signals from each eeg channel is good and then returns the percentage of those that are good

load_session_epochs: I was a little confused about this one- i don't know what an epoch is

load_session_summery: computes a quantitive summary of eeg data including average power by band, average power by channel, etc

analysisEngine: init: Generates basic analytics for recording groups distributionVetting: generates graphs for each epoch wave basicComparisons: generates graphs that compare the accummulated power bands for each wave trendAnalysis: generates a graph with how a specific variable progressed with time

get_rolling_powerByBand: checks whether there is a difference in data depending on the time interval in which the data is collected

unix_to_period: converts the timestamps collected to one more readable to us

unix_to_localdate: converts the date collected to one more readable to us

load_sessions: loads the sessions

load_fileset: loads data