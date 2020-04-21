# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:15:07 2019

@author: Fabian Karst
Input:  bl_data_processed.csv, cf_data_processed.csv, is_data_processed.csv (Preprocessing_WIDS.py)
        ratios.csv ()
        makro.csv (Macroindicators_ratios_polarityscores_descp.py)
        lbl_data_processed.csv (Labelgeneration_WIDS.py)
        filings_data_processed.csv (Preprocessing_Textualanalysis.py)
Output: different train datasets with various oversampling methods applied, the test dataset (as csv)
Purpose: Select the 44 most important features of the dataset and apply oversampling to the training dataset.
"""


import os
import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier



# Set working directory
#os.chdir(r'C:\Users\KAT\Documents\Project\Oversampling')
df = [pd.read_csv("preprocesseddata//ratio//bl_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//cf_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//is_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//ratios.csv", low_memory=False, index_col = 0).drop("Unnamed: 0.1", axis = 1), pd.read_csv("preprocesseddata//ratio//filings_data_processed.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//macro_ind.csv", low_memory=False, index_col = 0), pd.read_csv("preprocesseddata//ratio//lbl_data_processed.csv", low_memory=False, index_col = 0)]    

#remove data from 2018
for i in range(len(df)):
    df[i] = df[i][(df[i]["fyear"] != 2018) & (df[i]["fyear"] != 2001)]


#check if the dataset only contains unique combinations of cik and year
def check_unique(df):
    data = df.copy()
    data["unique"] = data["cik"].apply(lambda x: str(x)) + data["fyear"].apply(lambda x: str(x))
    print("{} : {}".format(data["unique"].nunique(), len(data["unique"].index)))
    print("Contains only unique items: {}".format(data["unique"].nunique() == len(data["unique"].index)))

check_unique(df[1])
check_unique(df[2])
check_unique(df[3])
check_unique(df[4])
check_unique(df[5])
check_unique(df[6])


#merge Dataset into one table
mergeOnColumns = ['gvkey', 'datadate', 'fyear', 'indfmt', 'consol', 'popsrc', 'datafmt', 'conm', 'curcd', 'curncd', 'cik', 'costat', 'gsector', 'state']
y = df[6]["bnkrpt"].apply(lambda x: int(x))

X_all = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(df[0], df[1],  how="left", left_on=mergeOnColumns, right_on = mergeOnColumns), df[2],  how="left", left_on=mergeOnColumns, right_on = mergeOnColumns), df[3],  how="left", left_on=["fyear", "cik"], right_on = ["fyear", "cik"]), df[4],  how="left", left_on=["fyear", "cik", "state", "gsector"], right_on = ["fyear", "cik", "state", "gsector"]), df[5],  how="left", left_on=["fyear", "cik"], right_on = ["fyear", "cik"])
X_all = X_all.fillna(0)

X_WIDS_w_ratios = pd.merge(pd.merge(pd.merge(df[0], df[1],  how="left", left_on=mergeOnColumns, right_on = mergeOnColumns), df[2],  how="left", left_on=mergeOnColumns, right_on = mergeOnColumns), df[3],  how="left", left_on=["fyear", "cik"], right_on = ["fyear", "cik"])
X_WIDS_w_ratios = X_WIDS_w_ratios.fillna(0)

X_WIDS_wo_ratios = pd.merge(pd.merge(df[0], df[1],  how="left", left_on=mergeOnColumns, right_on = mergeOnColumns), df[2],  how="left", left_on=mergeOnColumns, right_on = mergeOnColumns)
X_WIDS_wo_ratios = X_WIDS_wo_ratios.fillna(0)

#check that the dataset do not contain nan or inf values
for i in range(0,6):
    inf = sum(sum(np.isinf(np.array(df[i].loc[:, list(set(X_all.columns)-set(mergeOnColumns))].fillna(0)))))
    nan = sum(sum(np.isnan(np.array(df[i].loc[:, list(set(X_all.columns)-set(mergeOnColumns))].fillna(0)))))
    print("df[{}] contains {} inf and {} null".format(i, inf, nan))


#reduce the components of the dataset
def reduce_components(X, X_name, y, top_n, do_plot):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X.loc[:, list(set(X.columns)-set(mergeOnColumns))], y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    topcols = [list(set(X.columns)-set(mergeOnColumns))[i] for i in list(indices[:top_n])]
    topimportances = [list(importances)[i] for i in list(indices[:top_n])]
    
    if do_plot:
        plt.figure(figsize=(21,7))
        plt.bar(topcols, topimportances)
        plt.xticks(rotation=90)
        plt.title("Top {} features".format(top_n))
        plt.savefig("{}_top{}_features.png".format(X_name, top_n))
        plt.show()
    
    X_top = X.loc[:, ["cik"] + ["fyear"] + topcols]
    #X_top =  X.iloc[:,[list(X_all.columns).index("fyear")] + [list(X_all.columns).index("cik")] + [list(X_all.columns).index("conm")] + list(indices[:top_n])]
    return(X_top, topcols, topimportances)

nFeatures = 44

X_all, topcols, topimportances = reduce_components(X_all, "X_all", y, nFeatures, True)
X_WIDS_w_ratios, topcols, topimportances = reduce_components(X_WIDS_w_ratios, "X_WIDS_w_ratios", y, nFeatures, True)
X_WIDS_wo_ratios, topcols, topimportances = reduce_components(X_WIDS_wo_ratios, "X_WIDS_wo_ratios", y, nFeatures, True)


#create testing and training sets
X_all_train, X_all_test, y_train, y_test, indices_train, indices_test = train_test_split(X_all, y, range(X_all.shape[0]), test_size=0.33, random_state=42)
X_WIDS_w_ratios_train = X_WIDS_w_ratios.iloc[indices_train, :]
X_WIDS_w_ratios_test = X_WIDS_w_ratios.iloc[indices_test, :]
X_WIDS_wo_ratios_train = X_WIDS_wo_ratios.iloc[indices_train, :]
X_WIDS_wo_ratios_test = X_WIDS_wo_ratios.iloc[indices_test, :]

#save the test sets
pd.DataFrame(y_test, columns = ["bnkrpt"]).to_csv("y_test.csv")
X_all_test.to_csv("X_all_test.csv")
pd.DataFrame(y_train, columns = ["bnkrpt"]).to_csv("y_train_wo_OS.csv")
X_all_train.to_csv("X_all_train_wo_OS.csv")
X_WIDS_w_ratios_test.to_csv("X_WIDS_w_ratios_test_{}.csv".format(nFeatures))
X_WIDS_wo_ratios_test.to_csv("X_WIDS_wo_ratios_test_{}.csv".format(nFeatures))


#plotting functions
def create_3Dplot(X, y, angle, name, transformer = []):
    if transformer == []:
        transformer = decomposition.PCA(n_components=3)
        transformer.fit(X)
    
    X_3d = transformer.transform(X)
    
    #Create 3D Plot
    y_col = y.replace(0, "green").replace(1, "red")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_col)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.view_init(30, angle)
    plt.savefig(name)
    plt.show()
    return(transformer)

def create_2Dplot(X, y, name, transformer = []):
    if transformer == []:
        transformer = decomposition.PCA(n_components=2)
        transformer.fit(X)
    
    X_2d = transformer.transform(X)
    
    #Create 3D Plot
    y_col = y.replace(0, "green").replace(1, "red")
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c = y_col)
    plt.savefig(name)
    plt.show()
    return(transformer)


dfs = [X_all_train, X_WIDS_w_ratios_train, X_WIDS_wo_ratios_train]
dfs_names = ["X_all_train", "X_WIDS_w_ratios_train", "X_WIDS_wo_ratios_train"]
angle = 45

#plot the number of defaults vs non-defaults in the training dataset
sns.countplot(y_train)
plt.title('Train: No Default (0) vs. Default (1)')
plt.savefig("train_nodefault_vs_default.png")
plt.show()

#plot the number of defaults vs non-defaults in the test dataset
sns.countplot(y_test)
plt.title('Test: No Default (0) vs. Default (1)')
plt.savefig("test_nodefault_vs_default.png")
plt.show()

#Run a loop to Oversample different format of the data
for i in range(len(dfs)):
    pca_transformer_3D = create_3Dplot(dfs[i], y_train, angle, "{}_3D_without_Oversampling.png".format(dfs_names[i]))
    pca_transformer_2D = create_2Dplot(dfs[i], y_train, "{}_2D_without_Oversampling.png".format(dfs_names[i]))

    #Random naive over-sampling
    from imblearn.over_sampling import RandomOverSampler

    X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(dfs[i], y_train)
    X_resampled = pd.DataFrame(X_resampled, columns = dfs[i].columns)
    y_resampled = pd.DataFrame(y_resampled, columns=["bnkrpt"])
    create_3Dplot(X_resampled, y_resampled.squeeze(), angle, "{}_3D_with_rand_naive_os.png".format(dfs_names[i]), pca_transformer_3D)
    create_2Dplot(X_resampled, y_resampled.squeeze(), "{}_2D_with_rand_naive_os.png".format(dfs_names[i]), pca_transformer_2D)
    
    X_resampled[X_all_train.columns].to_csv("{}_RNOS.csv".format(dfs_names[i]))
    y_resampled.to_csv("y{}_RNOS.csv".format(dfs_names[i][1:]))
    
    #Synthetic minority oversampling
    from imblearn.over_sampling import SMOTE
    
    X_resampled, y_resampled = SMOTE().fit_resample(dfs[i], y_train)
    X_resampled = pd.DataFrame(X_resampled, columns = dfs[i].columns)
    y_resampled = pd.DataFrame(y_resampled, columns=["bnkrpt"])
    create_3Dplot(X_resampled, y_resampled.squeeze(), angle, "{}_3D_with_smote_os.png".format(dfs_names[i]), pca_transformer_3D)
    create_2Dplot(X_resampled, y_resampled.squeeze(), "{}_2D_smote_os.png".format(dfs_names[i]), pca_transformer_2D)
    
    X_resampled[X_all_train.columns].to_csv("{}_SMOTEOS.csv".format(dfs_names[i]))
    y_resampled.to_csv("y{}_SMOTEOS.csv".format(dfs_names[i][1:]))
    
    #Adaptive synthetic
    from imblearn.over_sampling import ADASYN
    
    X_resampled, y_resampled = ADASYN().fit_resample(dfs[i], y_train)
    X_resampled = pd.DataFrame(X_resampled, columns = dfs[i].columns)
    y_resampled = pd.DataFrame(y_resampled, columns=["bnkrpt"])
    create_3Dplot(X_resampled, y_resampled.squeeze(), angle, "{}_3D_adasyn_os.png".format(dfs_names[i]), pca_transformer_3D)
    create_2Dplot(X_resampled, y_resampled.squeeze(), "{}_2D_adasyn_os.png".format(dfs_names[i]), pca_transformer_2D)
    
    X_resampled[X_all_train.columns].to_csv("{}_ADASYNOS.csv".format(dfs_names[i]))
    y_resampled.to_csv("y{}_ADASYNOS.csv".format(dfs_names[i][1:]))
