# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 06:44:13 2019

@author: Fabian Karst
Input:  all fillings.csvs (Collecting_Preprocessing_Analyzing_10K_files.py)
        bl_data_processed.csv, cf_data_processed.csv, is_data_processed.csv, lbl_data_processed.csv (Preprocessing_WIDS.py)
Output: filings_data_processed.csv (csv containing the filings with the same structure as all other data.
Purpose: Transform the filings data into a format which can be used for further processing
"""

import os
import csv
csv.field_size_limit(100000000)
import pandas as pd 
import numpy as np
import glob

# Set working directory
os.chdir('D:\Programming\Python\SmartDataAnalytics\Project\Preprocessing_filings')

df = [pd.read_csv("preprocesseddata//ratio//bl_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//cf_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//is_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//lbl_data_processed.csv", low_memory=False, index_col = 0)]    

df.append(df[0][["fyear", "cik", "state", "gsector"]])
df[4]["word_count"] = (np.nan * len(df[4]))
df[4]["positive_LM"] = (np.nan * len(df[4]))
df[4]["negative_LM"] = (np.nan * len(df[4]))
df[4]["polarity_LM"] = (np.nan * len(df[4]))
df[4]["positive_B"] = (np.nan * len(df[4]))
df[4]["negative_B"] = (np.nan * len(df[4]))
df[4]["polarity_B"] = (np.nan * len(df[4]))
df[4]["average_sentence_lenght"] = (np.nan * len(df[4]))

stats = pd.DataFrame([], columns=["year", "total", "found", "word_count", "positive_LM", "negative_LM", "polarity_LM", "positive_B", "negative_B", "polarity_B", "average_sentence_lenght"])

for file in glob.glob("filings/*.csv"):
    year = int(file[-8:-4])
    
    with open(r"filings\textual_analyis_2002.csv", encoding="utf-8") as file:
        csv_data = csv.reader(file, delimiter=',')
        next(csv_data)
        for row in csv_data:
            if not df[4][(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year)].empty:
                df[4].loc[(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year), "word_count"] = row[2]
                df[4].loc[(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year), "positive_LM"] = row[3]
                df[4].loc[(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year), "negative_LM"] = row[4]
                df[4].loc[(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year), "polarity_LM"] = row[5]
                df[4].loc[(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year), "positive_B"] = row[6]
                df[4].loc[(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year), "negative_B"] = row[7]
                df[4].loc[(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year), "polarity_B"] = row[8]
                df[4].loc[(df[4]['cik'] == float(row[0])) & (df[4]['fyear'] == year), "average_sentence_lenght"] = row[9]
            if sum((df[4]['cik'] == row[0]) & (df[4]['fyear'] == year)) > 1:
                print("Error, multiple matches for cik and year")
                print(df[4][(df[4]['cik'] == row[0]) & (df[4]['fyear'] == year)])
    
    print("Result: {} companies of {} found.".format(sum((df[4]['word_count'].notnull()) & (df[4]['fyear'] == year)), sum(df[4]['fyear'] == year)))
    
    word_count = np.asarray(df[4].loc[(df[4]["fyear"] == year) & (df[4]["word_count"].notnull()), "word_count"], dtype=np.float).mean()
    positive_LM = np.asarray(df[4].loc[(df[4]["fyear"] == year) & (df[4]["positive_LM"].notnull()), "positive_LM"], dtype=np.float).mean()
    negative_LM = np.asarray(df[4].loc[(df[4]["fyear"] == year) & (df[4]["negative_LM"].notnull()), "negative_LM"], dtype=np.float).mean()
    polarity_LM = np.asarray(df[4].loc[(df[4]["fyear"] == year) & (df[4]["polarity_LM"].notnull()), "polarity_LM"], dtype=np.float).mean()
    positive_B = np.asarray(df[4].loc[(df[4]["fyear"] == year) & (df[4]["positive_B"].notnull()), "positive_B"], dtype=np.float).mean()
    negative_B = np.asarray(df[4].loc[(df[4]["fyear"] == year) & (df[4]["negative_B"].notnull()), "negative_B"], dtype=np.float).mean()
    polarity_B = np.asarray(df[4].loc[(df[4]["fyear"] == year) & (df[4]["polarity_B"].notnull()), "polarity_B"], dtype=np.float).mean()
    average_sentence_lenght = np.asarray(df[4].loc[(df[4]["fyear"] == year) & (df[4]["average_sentence_lenght"].notnull()), "average_sentence_lenght"], dtype=np.float).mean()
    
    stats.append(pd.DataFrame([[year, sum((df[4]['word_count'].notnull()) & (df[4]['fyear'] == year)), sum(df[4]['fyear'] == year), word_count, positive_LM, negative_LM, polarity_LM, positive_B, negative_B, polarity_B, average_sentence_lenght]], columns=["year", "total", "found", "word_count", "positive_LM", "negative_LM", "polarity_LM", "positive_B", "negative_B", "polarity_B", "average_sentence_lenght"]))
    
    df[4].loc[(df[4]["word_count"].isnull()) & (df[4]['fyear'] == year), "word_count"] =  word_count
    df[4].loc[(df[4]["positive_LM"].isnull()) & (df[4]['fyear'] == year), "positive_LM"] = positive_LM 
    df[4].loc[(df[4]["negative_LM"].isnull()) & (df[4]['fyear'] == year), "negative_LM"] = negative_LM
    df[4].loc[(df[4]["polarity_LM"].isnull()) & (df[4]['fyear'] == year), "polarity_LM"] = polarity_LM
    df[4].loc[(df[4]["positive_B"].isnull()) & (df[4]['fyear'] == year), "positive_B"] = positive_B
    df[4].loc[(df[4]["negative_B"].isnull()) & (df[4]['fyear'] == year), "negative_B"] = negative_B
    df[4].loc[(df[4]["polarity_B"].isnull()) & (df[4]['fyear'] == year), "polarity_B"] = polarity_B
    df[4].loc[(df[4]["average_sentence_lenght"].isnull()) & (df[4]['fyear'] == year), "average_sentence_lenght"] = average_sentence_lenght
    
stats.to_csv("statistic_average_filings_by_year.csv")
df[4].to_csv("preprocesseddata//ratio//filings_data_processed.csv")
