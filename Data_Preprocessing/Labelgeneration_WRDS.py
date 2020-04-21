# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:56:39 2019

@author: Fabian Karst
Input:  bl_data_processed.csv, cf_data_processed.csv, is_data_processed.csv (from Preprocessing_WRDS.py)  
        WRDS Corporate Legal Bankruptcy data as .sas7bat
Output: lbl_data_processed.csv
Purpose: Identify the companies which disappeared during the observation period (label: disap) and identify which of them went bankrupt by Chapter 7 or 11 (label: bnkrpt)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Set working directory
os.chdir('D:\Programming\Python\SmartDataAnalytics\Project\Preprocessing_WRDS')

#import wids data
df_wids = [pd.read_csv("preprocesseddata//ratio//bl_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//cf_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//is_data_processed.csv", low_memory=False, index_col = 0)]

#import bankruptcy label data
df_sas = pd.read_sas(r"bankrupt.sas7bdat")
df_sas["fyear"] = df_sas["BANK_END_DATE"].dt.year
df_sas = df_sas[(df_sas.fyear > 2000) & (df_sas.fyear < 2019)]

#plot of bankrupt companies by year
dataByYear = df_sas.groupby("fyear")["COMPANY_FKEY"].nunique()    
print()
plt.title(r'Companies bankrupt by Year')
plt.bar(list(dataByYear.index), list(dataByYear))
plt.savefig("Descriptives/BankruptByYear.png", transparent=True, dpi=300)
plt.show()

#create disappered label data
cikGrouped = df_wids[0].groupby("cik")
df_disap = []

for group in cikGrouped:
    if len(group[1]) < 18:
        if max(group[1]["fyear"]) != 2018:
            df_disap.append([group[1]["cik"].iloc[-1], group[1]["conm"].iloc[-1], max(group[1]["fyear"])])

df_disap = pd.DataFrame(df_disap, columns=["cik", "conm", "fyear"])

#plot of disappeared companies by year
dataByYear = df_disap.groupby("fyear")["cik"].nunique()    
print(dataByYear)
plt.title(r'Companies out of Service by Year')
plt.bar(list(dataByYear.index), list(dataByYear))
plt.savefig("Descriptives/OutofServiceByYear.png", transparent=True, dpi=300)
plt.show()


#create wids supdataset for label
df_wids.append(df_wids[0][["fyear", "cik", "state", "gsector"]])
df_wids[3]["bnkrpt"] = ([False] * len(df_wids[3]))
df_wids[3]["disap"] = ([False] * len(df_wids[3]))


#transform disapperared label data into sudataset for disap label
matched = 0
unmatched = 0
for i in range(0, len(df_disap)):
    if not (df_wids[3][(df_wids[3]["fyear"] == (df_disap.iloc[i, :]["fyear"])) & (df_wids[3]["cik"] == float(df_disap.iloc[i, :]["cik"]))]).empty:
        df_wids[3].loc[(df_wids[3]["fyear"] == (df_disap.iloc[i, :]["fyear"])) & (df_wids[3]["cik"] == float(df_disap.iloc[i, :]["cik"])), "disap"] = True
        matched += 1
    else:
        unmatched += 1


#transform bankruptcy label data into sudataset for bnkrpt label
matched = 0
unmatched = 0
for i in range(0, len(df_sas)):
    if not (df_wids[3][(df_wids[3]["cik"] == float(df_sas.iloc[i, :]["COMPANY_FKEY"])) & (df_wids[3]["disap"] == True)]).empty:
        df_wids[3].loc[(df_wids[3]["cik"] == float(df_sas.iloc[i, :]["COMPANY_FKEY"])) & (df_wids[3]["disap"] == True), "bnkrpt"] = True
        matched += 1

#save the data
df_wids[3].to_csv("preprocesseddata//ratio//lbl_data_processed.csv")
