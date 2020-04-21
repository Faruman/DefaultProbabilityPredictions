# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:37:03 2019

@author: Jasmine

Input: ratios.csv, macrodata_simple series.csv, filings_data_processed.csv
Output: sumstat_ratios.xlsx, macro.png, fdi.png, polarLM.png, polarB.png
Purpose: Create summary statistics for financial ratios, make area-chart for FDI and lineplots for other macroeconomic indicators, boxplots for polarity scores

"""

import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

%matplotlib inline
plt.rcParams['figure.figsize'] = 10, 8


# financial ratios summary statistics
ratios = pd.read_csv('ratios.csv')
sumstat = pd.DataFrame(ratios.describe())
sumstat.to_excel('sumstat_ratios.xlsx')

# Macro indicators graphs
macro = pd.read_csv("macrodata_simple series.csv")
macro['fyear'] = macro['fyear'].astype('int64')

#Plot macro variables except FDI
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('years')
ax1.set_ylabel('%', color=color)
ax1.plot(macro["fyear"], macro["cpi"], '-g', label = 'CPI')
ax1.plot(macro["fyear"], macro["unemp"], '-r', label = 'Unemployment' )
ax1.plot(macro["fyear"], macro["gdp_growth"], '-b', label = 'GDP growth' )
ax1.tick_params(axis='y', labelcolor=color)
ax1.ticklabel_format(useOffset=False)
legend = ax1.legend(loc='lower right', shadow=True, fontsize='small', bbox_to_anchor=(0.34, 0.85))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('%', color=color)
ax2.plot(macro["fyear"], macro["stockp_volatility"], '--y', label = 'Stock Price Volatility (right axis)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.ticklabel_format(style='plain', axis='x')
legend = ax2.legend(loc='lower center', shadow=True, fontsize='small', bbox_to_anchor=(0.7, 1))

fig.tight_layout()  # otherwise the right y-label is slightly clipped

fig.savefig("macro.png", dpi=300, transparent=True)

#Plot FDI
macro["fdi"] = macro["fdi"]/1000000 # scale the data

plt.style.use("fivethirtyeight")

fig, ax2 = plt.subplots()
ax2.set_xlabel('years', fontsize= 'small')
ax2.set_ylabel('1000,000$', fontsize='small')
ax2.fill_between( macro["fyear"], macro["fdi"], color="green", alpha=0.2)
ax2.plot(macro["fyear"], macro["fdi"], color="Slateblue", alpha=0.6, label = "FDI")
legend = ax2.legend(loc='upper left', shadow=True, fontsize='medium')
ax2.ticklabel_format(style='plain', axis='y')
ax2.ticklabel_format(useOffset=False)
fig.savefig("fdi.png", dpi=300)

# Boxplots for polarity scores

polarS = pd.read_csv('filings_data_processed.csv')
polarS['fyear'] = polarS['fyear'].astype('int64')

sns.boxplot(x= "fyear", y = "polarity_LM",data=polarS)
plt.title("Polarity scores based on Loughran and McDonald library ")
plt.xticks(rotation=80)
plt.savefig("polarLM.png", dpi=300, transparent=True)

sns.boxplot(x= "fyear", y = "polarity_B",data=polarS)
plt.title("Polarity scores based on Bing Liu library")
plt.xticks(rotation=80)
plt.savefig("polarB.png", dpi=300, transparent=True)

