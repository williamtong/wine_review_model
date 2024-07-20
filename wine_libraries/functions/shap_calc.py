import pandas as pd
import numpy as np
import re

import importlib
import matplotlib.pyplot as plt

import shap
shap.plots.initjs()


def create_shape_explanation(raw_model, X, y, columns_of_interest = None):  # None means all columns are of interest
    '''
    input:
    raw_model: trained xgboost or scikit-learn model (no wrapper)
    X (pandas dataframe): X data same format to train raw_model, usually the training data set.
    y (1-D pandas dataframe or np array): y data, same as X.
    columns_of_interest:  Note: None means all columns are of interest

    output: shap.explanation object
    '''
    if columns_of_interest is None:
        columns_of_interest = X.columns
        
    explainer = shap.TreeExplainer(raw_model)
    explanation = explainer(X = X,
                            y = y,
                            check_additivity=False 
                           )
    
    return explanation

def find_features_with_words(features, word):
    '''A filter function to select words of interest'''
    return [feature for feature in features if (word in feature[0])]

def plot_summary_plot(explanation, 
                      X, 
                      word_filter_list = [], word_filter_exclusivity = True, max_display = 20):
    '''
    This function takes plot the shap_values of the features that contain the word(s) in the word_filter_list.
    If the word_filter_list is [], it will just plot the top max_display features.

    explanation (shap object):  shap class object 
    X (Pandas dataframe): Samples (# samples x # features) on which to explain the model’s output.
    word_filter_list (list): Words of interests.
                             If [], shap values for all features will be plotted.
                             If len(word_filter_list) == 1 and word_filter_exclusivity = True, 
                                 plot only for feature == word verbatim.
                             If len(word_filter_list) != 0 and word_filter_exclusivity == True,
                                 plot features with any of the words in word_filter_list.
                             If len(word_filter_list) != 0 and word_filter_exclusivity == False,
                                 plot features with *all* of the words in word_filter_list.
    word_filter_exclusivity (boolean) : How to handle multiple elements in word_filter_list. 
                                See word_filter_list above.
    max_display = maximum number of top features to be plotted.
    '''
    full_feature_names = explanation.feature_names
    shap_values = explanation.values
    features_of_interest = [(feature.lower(), i) for i, feature in enumerate(full_feature_names)]
    if len(word_filter_list) == 1 and word_filter_exclusivity:
        word = word_filter_list[0]
        features_of_interest = [(feature.lower(), i) for i, feature in enumerate(full_feature_names)
                                if feature.split('_')[-1].lower() == word]
        
    elif len(word_filter_list) != 0 and word_filter_exclusivity:    
        features_of_interest = [(feature.lower(), i) for i, feature in enumerate(full_feature_names)]
        for word in word_filter_list:
            features_of_interest =  find_features_with_words(features_of_interest, word)

    elif len(word_filter_list) != 0 and word_filter_exclusivity == False:
        word_filter_list = [word.lower() for word in word_filter_list]
        features_of_interest = []
        for word in word_filter_list:
            features_of_interest.extend([(feature, i) for i, feature in enumerate(full_feature_names) 
                                if word in feature.lower()])
        features_of_interest = set(features_of_interest)

    features_of_interest_indices = [feature_w_word[1] for feature_w_word in features_of_interest]
    feature_names_wo_prefx =[name[0].split("_")[-1] for name in features_of_interest]

    # If all features are boolean, it is better to show with monochrome cmap
    # print(X.iloc[:, features_of_interest_indices].dtypes)
    if 'norm-points' not in feature_names_wo_prefx:
        cmap = 'Reds'
    else:
        cmap = 'bwr'
    shap.summary_plot(shap_values[:,features_of_interest_indices], 
                      X.iloc[:, features_of_interest_indices], 
                      feature_names=feature_names_wo_prefx,
                      show_values_in_legend = True,
                      cmap = cmap,
                      max_display = max_display
                     )