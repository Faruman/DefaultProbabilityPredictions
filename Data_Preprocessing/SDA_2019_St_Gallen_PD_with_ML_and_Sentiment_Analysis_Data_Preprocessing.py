# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:27:28 2019

@author: Fabian Karst
Input: data from WRDS as .csv
Output: bl_data_processed.csv, cf_data_processed.csv, is_data_processed.csv
Purpose: Select the relevant data (year: 2000 - 2019, company country: US, company type: Nonfinacial service provider), transform all the data into ratios and further drop the columns which contain NA for more than 99% of the rows NA
"""

import os
import pandas as pd

# Set working directory
os.chdir('D:\Programming\Python\SmartDataAnalytics\Project\Preprocessing_WRDS')

#import dataset
df = [pd.read_csv("Balance_Sheet.csv", low_memory=False), pd.read_csv("Cash_Flow.csv", low_memory=False), pd.read_csv("Income_Statement.csv", low_memory=False)]    

for i in range(len(df)):
    #keep only data from dates after 2000 and before 2019
    df[i] = df[i][(df[i].fyear > 2000) & (df[i].fyear < 2019)]
    #drop all non-US companies
    df[i] = df[i][df[i].curncd == "USD"]
    #drop all financial service providers (according to global industry classification standard)
    df[i] = df[i][df[i].gsector != 40]
    
    #drop all columns which only contain na
    df[i] = df[i].dropna(axis='columns', how='all')
    #drop columns which have the same values for all companies
    nunique = df[i].apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df[i].drop(cols_to_drop, axis=1)

#drop companies which has "at" = na / 0
dropDueToNoAT = list(~df[0]["cik"].isin(df[0][(df[0]["at"] == 0) | (pd.isnull(df[0]["at"]))]["cik"].unique()))
df[0] = df[0][dropDueToNoAT]
df[1] = df[1][dropDueToNoAT]
df[2] = df[2][dropDueToNoAT]

#replace zero/na revenue with small number to prevent division by zero
df[2]["revt"][df[2]["revt"] == 0] = 1e-4
df[2]["revt"][df[2]["revt"].isnull()] = 1e-4

#Uncomment the following lines if you do not want ratios
#Creating ratios by dividing everything in the balancesheet by total assets  
df[0].update(df[0].iloc[:, list(range(10, 371))].div(df[0]["at"], axis='index'))
#Creating ratios by dividing everything in the cashflow statment by total revenue
df[1].update(df[1].iloc[:, list(range(10, 67))].div(df[2]["revt"], axis='index'))
#Creating ratios by dividing everything in the income statement by total revenue
df[2].update(df[2].iloc[:, list(range(10, 335))].div(df[2]["revt"], axis='index'))

#drop all columns for which less than 1% of the data has entries
for i in range(len(df)):
    toDrop = df[i].isnull().sum(axis = 0)/len(df[i]) > 0.99
    df[i] = df[i].drop(toDrop[toDrop == True].index, axis=1)

#save the preprocessed data
df[0].to_csv("bl_data_processed.csv")
df[1].to_csv("cf_data_processed.csv")
df[2].to_csv("is_data_processed.csv")
