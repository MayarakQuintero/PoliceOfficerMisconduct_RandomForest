'''
Takes in a cleaned dataframe, list of vars to include.
Makes a model and runs it and stuff.
'''
import PreProcessing as PreProcessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import sklearn

log_est = LogisticRegression(max_iter = 1000,random_state=30)

log_params_dict = {'penalty':("l2", "none") ,
                 'C': [.01,.1,1,10,100]}
svc_params_dict = {'LinearSVC': [{'C': x, 'random_state': 15} \
                  for x in (0.01, 0.1, 1, 10, 100)]}
models_dict = {
    'LogisticRegression': LogisticRegression(max_iter = 1000),
    #'LinearSVC': SVC(kernel="linear", probability=True,random_state = 3),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=3),
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state =4)
}

big_params_dict = {
    'LogisticRegression': [{'penalty': ("l2", "none"), 'C': [.01,.1,1,10,100], 'random_state': (3,)}],
    'GaussianNB': [{'priors': (None,)}],
    'LinearSVC': [{'C': [.01,.1,1,10,100], 'random_state': (3,)}],
    "DecisionTree": [{"criterion":("gini", "entropy"),
                     "max_depth":[1,3,5],
                     "min_samples_split":[2,5,10]}],
    "RandomForest": [{"criterion":("gini", "entropy"),
                     "max_depth":[1,3,5],
                     "min_samples_split":[2,5,10],
                     "n_estimators": [100, 1000, 5000]}]
}
big_how_score = ["recall", "precision", "balanced_accuracy"]
log_how_score = ["recall", "precision", "balanced_accuracy"]

def log_model(df, target_col, cont_feat, cat_feat, refit):
    '''
    wrapper function for all of it
    '''
    #cont_feat.extend(bin_names) #bin_names comes from data prep
    train, test = PreProcessing.tt_split(df[[target_col] + cont_feat+cat_feat], 30)
    pare_df(df, cont_feat, cat_feat, target_col)
    train_X, train_Y, test_X, test_Y = pre_processing(target_col, train, test, cont_feat, cat_feat)
    grid = build_log_model(train_X, train_Y, refit=refit)
    best_log = eval_log_model(grid, test_X, test_Y)
    fixed_val_threshold(best_log, test_X, test_Y)
    return None

def pare_df(final_df, cont_feat, cat_feat, target_col):
    '''
    NOTE: bin_names is returned from the make_df function
    '''

    #cont_feat.extend(bin_names) moved this to log_model because it's passed into
        # tt split and preprocessing. If
    vars_to_include = cont_feat + cat_feat + [target_col]
    final_df = final_df[vars_to_include]
    na_to_zero = list(final_df.columns)
    if "start_date_timestamp" in na_to_zero:
        na_to_zero.remove("start_date_timestamp") # need to account for this not being there
    final_df["cleaned_rank"].fillna(value="Unknown", inplace=True)
    final_df[na_to_zero] = final_df[na_to_zero].fillna(value=0)



def build_log_model(train_X, train_Y, params_dict=log_params_dict, estimator=log_est, refit="recall"):
    grid = GridSearchCV(estimator=estimator, 
                        param_grid=params_dict, 
                        scoring=log_how_score, 
                        cv=5, 
                        refit=refit)
    grid.fit(train_X, train_Y)
    return grid

def build_model(train_X, train_Y, refit, model_type):
    estimator = models_dict[model_type]
    params = big_params_dict[model_type]
    grid = GridSearchCV(estimator=estimator,
                        param_grid=params,
                        scoring=log_how_score,
                        cv=5,
                        refit=refit)
    grid.fit(train_X, train_Y)
    return grid


def single_model(df, model_type, target_col, cont_feat, cat_feat, refit):
    """For decision tree refit can be one"""
    train, test = PreProcessing.tt_split(df[[target_col] + cont_feat+cat_feat], 30)
    train_X, train_Y, test_X, test_Y, labels = pre_processing(target_col, train, test, cont_feat, cat_feat)
    grid = build_model(train_X, train_Y, refit, model_type)
    best_model = eval_model(grid, test_X, test_Y, model_type)
    fixed_val_threshold(best_model, test_X, test_Y)
    return best_model, labels

def try_four_models(df, target_col, cont_feat, cat_feat, refit):
    """Copied from log_model to call big_grid_search instead of build_log_model"""
    train, test = PreProcessing.tt_split(df[[target_col] + cont_feat+cat_feat], 30)
    train_X, train_Y, test_X, test_Y, labels = pre_processing(target_col, train, test, cont_feat, cat_feat)
    big_grid_search(train_X, train_Y, test_X, test_Y, refit=refit)


def big_grid_search(train_X, train_Y, test_X, test_Y, refit="recall"):
    for k in models_dict.keys():
        print()
        print("got to ", k)
        estimator = models_dict[k]
        params = big_params_dict[k]
        print("parameter dictionary is", params)
        print()
        grid = GridSearchCV(estimator=estimator,
                            param_grid=params,
                            scoring=log_how_score,
                            cv=5,
                            refit=refit)
        grid.fit(train_X, train_Y)
        eval_model(grid, test_X, test_Y, k)

def eval_model(grid_search, test_X, test_Y, model_type):
    # display(pd.DataFrame(grid_search.cv_results_))
    best_model = grid_search.best_estimator_
    pred_Y = best_model.predict(test_X)
    print("best ", model_type, " metrics:")
    print(sklearn.metrics.classification_report(test_Y, pred_Y, output_dict=True)["True"])
    pred_Y_probs = best_model.predict_proba(test_X)[:,(best_model.classes_ ==True)]
    sklearn.metrics.precision_recall_curve(test_Y, pred_Y_probs)
    print(sklearn.metrics.plot_precision_recall_curve(best_model, test_X, test_Y))
    return best_model



def eval_log_model(grid_search, test_X, test_Y):
    display(pd.DataFrame(grid_search.cv_results_))
    best_logistic = grid_search.best_estimator_
    pred_Y = best_logistic.predict(test_X)
    print("best estimator metrics:")
    print(sklearn.metrics.classification_report(test_Y, pred_Y, output_dict=True)["True"])
    pred_Y_probs = best_logistic.predict_proba(test_X)
    sklearn.metrics.precision_recall_curve(test_Y, pred_Y_probs[:,(best_logistic.classes_ ==True)])
    print(sklearn.metrics.plot_precision_recall_curve(best_logistic, test_X, test_Y))
    return best_logistic

def fixed_val_threshold(best_model, test_X, test_Y, custom_cutoff = None):
    fixed_val_threshold_metrics = pd.DataFrame(columns = ["Cutoff", "precision", "recall", "f1-score","support"])
    pred_Y_prob_True = best_model.predict_proba(test_X)[:,1]
    if not custom_cutoff:
        cutoffs = [0, .00001, .01, .05, .1, .2, .5, .7, .9, .95]
    else:
        cutoffs = custom_cutoff
    for c in cutoffs:
        p_y = (pred_Y_prob_True > c)
        d = sklearn.metrics.classification_report(test_Y, p_y, output_dict = True)["True"]
        d["balanced_accuracy"] = sklearn.metrics.balanced_accuracy_score(test_Y, p_y)
        d["Cutoff"] = c
        fixed_val_threshold_metrics = fixed_val_threshold_metrics.append(d, ignore_index = True)
    fixed_val_threshold_metrics = fixed_val_threshold_metrics[["Cutoff", "precision", "recall", "balanced_accuracy"]]
    fixed_val_threshold_metrics = fixed_val_threshold_metrics.set_index("Cutoff")
    display("fixed val threshold metrics: ", fixed_val_threshold_metrics)
    return None

def pre_processing(target_col, train, test, cont_feat, cat_feat):
    '''
    Impute missing values, normalize continuous values, 
    one-hot encode categorical values, split up features and targets.
    '''

    # impute na to median
    PreProcessing.na_to_median(train, test, cont_feat)
    # normalize
    my_scaler, cont_feat_norm = PreProcessing.normalize(train, cont_feat)
    PreProcessing.normalize(test, cont_feat, my_scaler)
    # one-hot encode
    OHE, train = PreProcessing.one_hot(train, cat_feat)
    _, test = PreProcessing.one_hot(test, cat_feat, OHE)

    train_limited, labels = PreProcessing.limit_for_fit(train, target_col, cont_feat_norm, list(OHE.get_feature_names()))
    test_limited, labels = PreProcessing.limit_for_fit(test, target_col, cont_feat_norm, list(OHE.get_feature_names()))
    train_X, train_Y = PreProcessing.feat_target_split(train_limited, target_col)
    test_X, test_Y = PreProcessing.feat_target_split(test_limited, target_col)

    return train_X, train_Y, test_X, test_Y, labels


