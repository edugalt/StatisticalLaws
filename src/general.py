import os
import gzip
import pandas as pd
import numpy as np
import ast

from modules_distributor import fit

###############################################################################################################################
# Function to read in data and output a dictionary with hashtag and count
# Input: time_format: "day" or "hour" or "minute" 

# In the day data sets, the header is "UserID" and "Hashtags" referring to hashtags and counts respectively
# In the hour/minute data sets, the header is "Hashtag" and "Frequency" which refer to hashtags and counts respectively

def get_data(path, filename, time_format):
    with gzip.open(path + filename) as f:
        features_train = pd.read_csv(f)
        
    # Convert panda data frame into 2 lists
    if time_format == "day":
        hashtags = features_train['UserID'].values.tolist()
        counts = features_train['Hashtags'].values.tolist()
    else:
        hashtags = features_train['Hashtag'].values.tolist()
        counts = features_train['Frequency'].values.tolist()
    
    # Convert 2 lists into dictionary
    data_dict = dict(zip(hashtags, counts))
    return(data_dict)

###############################################################################################################################
# Function to get all the zipped files in  a directory determined by path
# Input: path: name of directory

# Output: list of file names that are .gz files in the desired directory

def get_zipped_files(path):
    list_of_files = []

    for file in os.listdir(path):
        if file.endswith(".gz"):
            list_of_files.append(file)

    return(list_of_files)

###############################################################################################################################
# Function to fit the same model to many data sets
# Input: model = name of the model, eg: "simple", "shifted"

def rep_fit(model, nrep, path_in, filenames, time_format, path_out):
    fout = open(path_out + model + "-nrep" + str(nrep) + ".txt", "w")
    
    for i in range(len(filenames)):
        results_dict = get_data(path_in, filenames[i], time_format)
        counts = list(results_dict.values())
        res = fit(model = model, counts = counts, nrep = 1)
        fout.write(str(res) + "\n")
    fout.close()

###############################################################################################################################    
# Function that reads in fitted values from either fit() or rep_fit() calls.
# Saves time by not fitting again, just need to spend time to read in txt file.

# Output: List containing parameter estimates, negative log likelihood and nrep

def get_fitted_data(path_in, fitted_filename):
    f = open(path_in + fitted_filename, "r")
    contents = f.read()
    contents = contents.replace("inf", str(0)) # ast.literal_eval doesn't like inf or math.inf, set it to 0 instead
    f.close()
    
    results_list = []
    res_list_temp = contents.split("\n") # all elements of res_list are strings, not tuples or lists as they look like
    res_list_temp = res_list_temp[0:-1] # delete last element, an empty string
    
    for i in range(len(res_list_temp)):
        result_tuple_i = ast.literal_eval(res_list_temp[i]) #convert string into literal tuple
        results_list.append(result_tuple_i)
    
    return(results_list)
    
    
###############################################################################################################################
# Function that takes in files and outputs a dictionary of hashtags and counts
# Input: start/stop are indices, list_of_files are is a list of filenames of intrest, time_format refers to get_data() 
# Output: A dictionary of dictionaries. Example: {0: {"trump": 999, "Iran": 998}, 1 : {"Oscars": 1900}}
# The key refers to a specific file, the values of the dictionary are another dictionary of hashtags and counts.

def get_freq_dict(start, stop, list_of_files, path, time_format):
    hashtag_freq_dict_time = {}
    
    for i in range(start, stop):
        hashtag_freq_dict = {}
        results_dict_temp = get_data(path, list_of_files[i], time_format)
        
        hashtags = list(results_dict_temp.keys())
        counts = np.array(list(results_dict_temp.values()))
        freq = counts/np.sum(counts)

        for j in range(len(hashtags)):
            hashtag_freq_dict[hashtags[j]] = freq[j]
        
        hashtag_freq_dict_time[i] = hashtag_freq_dict
    return(hashtag_freq_dict_time)
