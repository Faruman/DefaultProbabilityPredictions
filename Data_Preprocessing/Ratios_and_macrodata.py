# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:25:08 2019

@author: Jasmine

Input: bl_data_processed.csv, cf_data_processed.csv, is_data_processed.csv
Output: ratios.csv, 20191114_macro_ind.csv, macrodata_simple series.csv 
Purpose: Collect macroeconomic data from WB database, match it with comapany panel dataset so that all companies have same macroindicator for a given year, calculate financial ratios.
"""

import pandas as pd 
import numpy as np
import wbdata
import datetime

# download macroenomic data from World Bank database 
# indicator ids can be found here: https://datacatalog.worldbank.org/
data_date = (datetime.datetime(2001, 1, 1), datetime.datetime(2018, 1, 1))
indicators = {"SL.UEM.TOTL.NE.ZS": "unemp", 
                   "NY.GDP.MKTP.KD.ZG" : "gdp_growth",
                  "FP.CPI.TOTL.ZG": "cpi", 
                  "GFDD.SM.01": "stockp_volatility", 
                  "BX.KLT.DINV.CD.WD": "fdi"}

usdata= wbdata.get_dataframe(indicators, country="USA", data_date=data_date)
usdata = usdata.reset_index()
usdata = usdata.rename(columns = {'index':'fyear'} )
usdata = usdata.astype(float)
usdata.to_csv('macrodata_simple series.csv')

#calculate finacial ratios for companies
# NB! there are companies with 0 sales value, thus some indicators cannot be calculated


balance_sheet = pd.read_csv(r"preprocesseddata\ratio\bl_data_processed.csv") 
income_state = pd.read_csv(r"preprocesseddata\ratio\is_data_processed.csv") 
cash_flow = pd.read_csv(r"preprocesseddata\ratio\cf_data_processed.csv") 

ratiosdf = pd.DataFrame(balance_sheet['Unnamed: 0'])
ratiosdf['cik'] = balance_sheet['cik']
ratiosdf['fyear'] = balance_sheet['fyear']

# Profitability ratios

ratiosdf['roa'] = income_state['ni']/balance_sheet['at']
ratiosdf['net_prof_margin'] = income_state['ni']/income_state['sale']
ratiosdf['oproa'] = income_state['oiadp']/balance_sheet['at']
ratiosdf['op_prof_margin'] = income_state['oiadp']/income_state['sale']
ratiosdf['ebit_ta'] = income_state['ebit']/balance_sheet['at']
ratiosdf['ebitda_ta'] = income_state['ebitda']/balance_sheet['at']
ratiosdf['ebit_sale'] = income_state['ebit']/income_state['sale']

# leverage ratios
ratiosdf['own_funds_ta_simple'] = (balance_sheet['at']-balance_sheet['lt'])/balance_sheet['at']
ratiosdf['own_funds_ta_adj'] = (balance_sheet['at']-balance_sheet['lt'])/(balance_sheet['at'] - balance_sheet['intan']- balance_sheet['ch'] - balance_sheet['ppent'])
ratiosdf['currentliab_ta'] = balance_sheet['lct']/balance_sheet['at']
ratiosdf['currentliab_cash_ta'] = (balance_sheet['lct']- balance_sheet['ch'])/balance_sheet['at']
ratiosdf['tl_ta'] = balance_sheet['lt']/balance_sheet['at']
ratiosdf['ebit_int'] = income_state['ebit']/income_state['xint']

#liquidity ratios
ratiosdf['cash_ta'] = balance_sheet['ch']/balance_sheet['at']
ratiosdf['cash_cl'] = balance_sheet['ch']/balance_sheet['lct']
ratiosdf['quick_assets'] = (balance_sheet['act']- balance_sheet['invt'])/balance_sheet['lct']
ratiosdf['ca_cl'] = balance_sheet['act']/balance_sheet['lct']
ratiosdf['work_cap'] = balance_sheet['wcap']/balance_sheet['at']
ratiosdf['cl_tl'] = balance_sheet['lct']/balance_sheet['lt']

#Activity ratios
ratiosdf['ta_sale'] = balance_sheet['at']/income_state['sale']
ratiosdf['invt_sale'] = balance_sheet['invt']/income_state['sale']
ratiosdf['rect_sale'] = balance_sheet['rect']/income_state['sale']
ratiosdf['ap_sale'] = balance_sheet['ap']/income_state['sale']

column_vec = ['roa', 'net_prof_margin', 'oproa', 'op_prof_margin', 'ebit_ta', 'ebitda_ta', 'ebit_sale', 'own_funds_ta_simple', 'own_funds_ta_adj', 'currentliab_ta', 'currentliab_cash_ta', 'tl_ta', 'ebit_int', 'cash_ta', 'cash_cl', 'quick_assets', 'ca_cl', 'work_cap', 'cl_tl', 'ta_sale', 'invt_sale', 'rect_sale', 'ap_sale']

for group in ratiosdf.groupby("fyear"):
    for ratio in column_vec:
        #ratiosdf.loc[((ratiosdf[ratio].isnull()) | (ratiosdf[ratio] == np.inf) | (ratiosdf[ratio] == -np.inf)) & (ratiosdf["fyear"] == group[0]), ratio] = group[1][ratio].loc[(group[1][ratio].notnull()) & ((group[1][ratio] != np.inf) & (group[1][ratio] != -np.inf))].mean()
        ratiosdf.loc[(ratiosdf[ratio].isnull()) & (ratiosdf["fyear"] == group[0]), ratio] = group[1][ratio].loc[(group[1][ratio].notnull()) & ((group[1][ratio] != np.inf) & (group[1][ratio] != -np.inf))].mean()
        ratiosdf.loc[(ratiosdf[ratio] == np.inf) & (ratiosdf["fyear"] == group[0]), ratio] = group[1][ratio].loc[(group[1][ratio].notnull()) & ((group[1][ratio] != np.inf) & (group[1][ratio] != -np.inf))].max()
        ratiosdf.loc[(ratiosdf[ratio] == -np.inf) & (ratiosdf["fyear"] == group[0]), ratio] = group[1][ratio].loc[(group[1][ratio].notnull()) & ((group[1][ratio] != np.inf) & (group[1][ratio] != -np.inf))].min()

#ratiosdf["net_prof_margin"] == -np.inf
#sum(np.isinf(np.array(ratiosdf)))
#investigate = []
#investigate.append(list(ratiosdf.columns))
#investigate.append(list(sum(np.isinf(np.array(ratiosdf)))))

#np.absolute(ratiosdf.loc[:, "net_prof_margin"])  == np.inf


ratiosdf.to_csv('ratios.csv')


# fit macro data to companies panel
# the panel data is unbalanced

ratiosdf["unemp"] = 1
ratiosdf["gdp_growth"] = 1
ratiosdf["cpi"] = 1
ratiosdf["stockp_volat"] = 1
ratiosdf["fdi"] = 1


'''this is a long loop :/ need to find more efficient way :/                
for i in range(len(ratiosdf['Unnamed: 0'])):
    for j in range(len(usdata['fyear'])):
        if ratiosdf['fyear'][i] == usdata['year'][j]:
            ratiosdf["unemp"][i] = usdata['unemp'][j]
            ratiosdf["gdp_growth"][i] = usdata['gdp_growth'][j]
            ratiosdf["cpi"][i] = usdata['cpi'][j]
            ratiosdf["stockp_volat"][i] = usdata['stockp_volatility'][j]
            ratiosdf["fdi"][i] = usdata['fdi'][j]
'''
#more efficient loop
for j in range(len(usdata['fyear'])):
    ratiosdf.loc[ratiosdf["fyear"] == usdata['fyear'][j], "unemp"] = usdata['unemp'][j]
    ratiosdf.loc[ratiosdf["fyear"] == usdata['fyear'][j], "gdp_growth"] = usdata['gdp_growth'][j]
    ratiosdf.loc[ratiosdf["fyear"] == usdata['fyear'][j], "cpi"] = usdata['cpi'][j]
    ratiosdf.loc[ratiosdf["fyear"] == usdata['fyear'][j], "stockp_volat"] = usdata['stockp_volatility'][j]
    ratiosdf.loc[ratiosdf["fyear"] == usdata['fyear'][j], "fdi"] = usdata['fdi'][j]


#fdi is not scaled yet
#ratiosdf.to_csv('combined_ratios_macro.csv') 

macrodf = ratiosdf[["fyear","cik","unemp","gdp_growth","cpi","stockp_volat","fdi"]]
macrodf.to_csv("20191114_macro_ind.csv")

