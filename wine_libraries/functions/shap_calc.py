import pandas as pd
import numpy as np
import re

import importlib
import matplotlib.pyplot as plt

import shap
shap.plots.initjs()

save_fig_dir = "wine_review_model/wine_libraries/price_model/images/shap_figs/"

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
    print("explainer done")
    explanation = explainer(X = X,
                            y = y,
                            check_additivity=False 
                           )
    
    return explanation


def find_features_with_words(features, word):
    '''A filter function to select words of interest'''
    return [feature for feature in features if (word in feature[0])]


def plot_bare_words(explanation,
                    X,
                    word_filter_list = [],
                    max_display = 50,
                    save_png_name = None):
    shape_values = explanation.values
    features_of_interest = []
    for word in word_filter_list:
        for i, feature in enumerate(X.columns):
            if word.lower() == feature.split('_')[-1].lower():
                features_of_interest.append((i, feature))
    features_of_interest_indices = [feature[0] for feature in features_of_interest]
    features_of_interest = [feature[1] for feature in features_of_interest]
    feature_names_wo_prefx = [feature.split('_')[-1] for feature in features_of_interest]
    cmap = 'Reds'
    shap.summary_plot(shape_values[:,features_of_interest_indices], 
                      X.iloc[:, features_of_interest_indices], 
                      feature_names=feature_names_wo_prefx,
                      show_values_in_legend = True,
                      cmap = cmap,
                      max_display = max_display,
                      show = False
                     )
    if save_png_name is not None:
        plt.savefig(save_fig_dir + save_png_name + ".png", format = "png", bbox_inches='tight')
    
def plot_summary_plot(explanation, 
                      X, 
                      word_filter_list = [], 
                      word_filter_exclusivity = 0, 
                      max_display = 20,
                      save_png_name = None):
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
    save_fig_name (str or None): If None, figure is not saved.  If str, then figured is saved as png file with the str as prefix.
    '''
    full_feature_names = explanation.feature_names
    shap_values = explanation.values
    features_of_interest = [(feature, i) for i, feature in enumerate(full_feature_names)]
    if len(word_filter_list) == 1 and word_filter_exclusivity == 1:
        word = word_filter_list[0]
        features_of_interest = [(feature, i) for i, feature in enumerate(full_feature_names)
                                if feature.split('_')[-1].lower() == word.lower()]
        
    elif len(word_filter_list) != 0 and word_filter_exclusivity == 1:    
        features_of_interest = [(feature, i) for i, feature in enumerate(full_feature_names)]
        for word in word_filter_list:
            features_of_interest =  find_features_with_words(features_of_interest, word)

    elif len(word_filter_list) != 0 and word_filter_exclusivity == False:
        word_filter_list = [word for word in word_filter_list]
        features_of_interest = []
        for word in word_filter_list:
            features_of_interest.extend([(feature, i) for i, feature in enumerate(full_feature_names) 
                                if word.lower() in feature.lower()])
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
                      max_display = max_display,
                      show = False
                     )
    if save_png_name is not None:
        plt.savefig(save_fig_dir + save_png_name + ".png", format = "png", bbox_inches='tight')
    plt.show()
                    


def plot_waterfall(explanation, 
                   X_all, 
                   df = None,  
                   sample_index = None,
                   y_all = None, 
                   max_display = 20, 
                   price_target = None,
                   save_png_name = None):
    '''
    This function plots the waterfall graph for a particular wine in the corpus.
    It shows how each feature influences the predicted price of the wine.
    If a sample index is provided it will use that wine.  Otherwise, it will randomly pick one from the corpus.

    Input:
    explanation (shap values): shap value object of the model.
    X_all (pandas dataframe): Featurized dataframe of the corpus, same as the one used to create the explanation.
    df (pandas dataframe): Original, UNPROCESSED dataframe of the corpus.  Make sure the indices match those of X_all.
    sample_index (int): index of the wine of interest.  If none is provide, the function is randomly pick a wine.
    y_all (pandas 1-D series): Actual prices of the wines.
    max_display:  Maximum number of features to display in the the waterfall plot.
    price_target (int or None): Find a wine within 10% of the price target.  In effect when sample_index == None only.

    Output:
    Shap waterfall plot.

    '''
    # If no sample_index provided, pick a random data point
    if sample_index == None:
        wine_price = -100
        while not price_target*0.9 <= wine_price  <= price_target*1.1:
            sample_number = np.random.randint(X_all.shape[0])
            sample_index = X_all.index[sample_number]
            wine_price = y_all.loc[sample_index]
    else:
        indices = list(X_all.index)
        sample_number = indices.index(sample_index)
    print(f'Sample number: {sample_number}, Sample index: {sample_index}')

    print("RAW DATA:")
    df_features = df.drop(["merged_info_text", "len", "description"], axis = 1).columns
    for feature in df_features:
        print(f'{feature.upper()}: {df.loc[sample_index, feature]}')
    print()

    if type(y_all) != type(None):
        print(f'THE ACTUAL PRICE IS ${y_all.loc[sample_index]}')
        
    print("Featurized data fed into the model:".upper())
    X = X_all.loc[sample_index, :]
    X_cat = X[X == True]
    feature_cat_dict = {feature.split("_")[-1]: feature.split("_")[0]  for feature in X_cat.index}
    features = feature_cat_dict.keys()
    categories = ["title", "winery", "region-1", "region-2", "province", "variety", "merged", "taster-name"]
    print(f'norm-points: {X["norm-points"]}')
    for category in categories:
        print(f'{category.upper()} feature:', end = ' ')
        for feature in features:
            if feature_cat_dict[feature] == category:
                print(f"'{feature}", end = "' ")
        print("")
    print(df.loc[sample_index, ["title"]].values[0])
    print(f'{str(df.loc[sample_index, ["title"]].values[0])}, Actual price = ${y_all.loc[sample_index]}')
    explanation.feature_names = [feature.split("_")[-1] for feature in explanation.feature_names]
    shap.plots.waterfall(explanation[sample_number, :], 
                         max_display=max_display,
                         show = False)
    if save_png_name is not None:
        if type(y_all) != type(None):
            plt.title(f'{str(df.loc[sample_index, ["title"]].values[0])}, Actual price = ${y_all.loc[sample_index]}')
        plt.savefig(save_fig_dir + save_png_name + ".png", format = "png", bbox_inches='tight')
    plt.show()
                    