"""Created by Sasha on June 7th for Final Project
This .py file has code to:
Split into train/test
NA to median
Normalize
One hot encode
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split



def tt_split(df, rs):
    """Written by Sasha on May 11th
    Returns train/test split based on typical 80-20 split we've done in class
    For now just a wrapper for sklearn.train_test_split() but later will get more complicated"""
    return train_test_split(df, test_size=.2, random_state=rs)


def na_to_median(train, test, cont_feat):
    """Updated by Sasha on May 30th
    Takes a dataframe with one or more continuous features specified and
    replaces na with median value for those features
    Most recent change is to use median of train for both train and test
    Returns nothing, makes changes to df in place"""
    for f in cont_feat:
        train_median = train[f].median()
        train[f].fillna(train_median, inplace=True)
        test[f].fillna(train_median, inplace=True)


def normalize(df, feat_to_norm, my_scaler = None):
    """Written by Sasha on May 11th
    Takes a dataframe with one or more continuous features specified and
    adds column that is that feature normalized.
    If my_scaler is none then  fit and return a new StandardScaler object
    Returns list of scaler objects to normalize train data"""
    if not my_scaler:
        my_scaler = StandardScaler()
        my_scaler.fit(df[feat_to_norm])
    feat_norm = my_scaler.transform(df[feat_to_norm])
    norm_col = []
    for i in range(len(feat_to_norm)):
        norm =  feat_to_norm[i] + "_norm"
        df.loc[:, norm] = feat_norm[:,i].copy()
        norm_col.append(norm)
    return  my_scaler, norm_col

def one_hot(df, cat_feat, OH_encoder = None):
    """Re-written on May 13th to use sklearn's OneHotEncoder per Felipe's suggestion in the slack channel"""
    df.loc[:,cat_feat] = df[cat_feat].fillna("None").astype("str").copy()
    if not OH_encoder:
        OH_encoder = OneHotEncoder(handle_unknown = "ignore")
        OH_encoder.fit(df[cat_feat].values)
    oh_encoded = OH_encoder.transform(df[cat_feat].values).toarray()
    oh_encoded = pd.DataFrame(oh_encoded, columns=OH_encoder.get_feature_names())
    return OH_encoder, pd.concat([df.reset_index(drop = True),
                         oh_encoded.reset_index(drop=True)], axis= 1)

def limit_for_fit(df, target_col, cont_feat = [], OHE_feat = []):
    """Written by Sasha on May 13 to just take target attr and processed features to be passed to .fit
    Last edited by Sasha on June 3rd to final_col as list """
    final_col = [target_col]+ cont_feat +OHE_feat

    return df[final_col], final_col

def feat_target_split(df, target_col):
    Y = df[target_col].values.ravel()
    X_col = list(df.columns)
    X_col.remove(target_col)
    X = df[X_col].values
    if len(X_col) == 1:
        X = X.reshape(-1,1)
    return X, Y
