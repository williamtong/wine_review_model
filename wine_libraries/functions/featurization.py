import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import  train_test_split

from sklearn.decomposition import PCA
from sklearn import preprocessing

import collections

import re
import csv

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
# import nltk
from nltk.tokenize import (word_tokenize,
                            sent_tokenize,
                            TreebankWordTokenizer,
                            wordpunct_tokenize,
                            TweetTokenizer,
                            MWETokenizer)
import string

import re
import itertools

import gc

saint_words = ['St. ' , 'Ste. ' , 'San ', 'Santa ', 'Saint ', 'Sainte ',  'St ', 'Ste ']
stop_words = ['la',  '&', 'le', 'les', 'the', 'da', 'della', 'del', 'de', 'al', 'alla',
              'dos', 'das', 'di', 'du', 'do', 'lo', 'of', 'and', 'with', 'to']
location_info_features = ["region-1", "region-2", "province", "country"]

def flatten_list(xss):
    return [x for xs in xss for x in xs]

class Preprocess_Data:
    def __init__(self,
                 columns_to_drop = ['taster-twitter-handle', 'description'],
                 columns_to_impute_unknown = ['designation', 
                                              'taster-name',
                                              'variety', 'region-1','region-2', 'province', 'country', 'winery']
                ):
        self.columns_to_drop = columns_to_drop
        self.columns_to_impute_unknown = columns_to_impute_unknown
        pass
        
    def fit(self, df, y = None):
        #check to make sure columns are presnt
        if len([col for col in self.columns_to_drop if col not in df.columns]) > 0:
            print(f'Some of the following columns {[col for col in self.columns_to_drop if col not in df.columns]} not in df')
            exit()
        if len([col for col in self.columns_to_impute_unknown if col not in df.columns]) > 0:
            print(f'Some of the following columns {[col for col in self.columns_to_impute_unknown if col not in df.columns]} not in df')
            exit()
        return self

    def transform(self, df, y = None):
        df = df.copy()
        df.drop(self.columns_to_drop, axis = 1, inplace = True)
        
        for col in self.columns_to_impute_unknown: 
            df[col] = df[col].fillna(col+"-Unknown")

        return df

def variety_tokenizer(df, feature):
    #G-S-M is a special case that would have been removed if the following step were not done
    df[feature] = df[feature].apply(lambda text: 'GSM' if text == 'G-S-M' else text)
    orig_multigram = list(df[feature].fillna('black_entry').apply(lambda text: text.replace("-", " ").replace("(", "").replace(")", "")).values)
    multigram_list = [text.split() for text in orig_multigram]
    # print(multigram_list)
    flattened_multigram_list = flatten_list(multigram_list)
    unique_multigrams = list(set([word for word in flattened_multigram_list if word not in stop_words and len(word)>1]))
    unique_multigrams = [unique_multigram for unique_multigram in unique_multigrams if unique_multigram not in stop_words]
    return multigram_list, unique_multigrams

def individual_text_processing(text):
    text = text.replace("-", " ")
 
    text = text[0].upper() + text[1:]
    
    # Add an extra '(' to the end of text if it doesn't have one.
    if '(' not in text:
        text = text + '('
    #Remove words after last open parenthesis because those are redundant location info.    
    text = text[:-text[::-1].index('(')-1]


    # Remove all other parenthesis
    text = text.replace('(', '').replace(')','')

    # Merge all saint words with following word
    for saint_word in saint_words:
        text = text.replace(saint_word, saint_word[:-1])
        
    #Replace all regular punctuations
    for char in [',', '.', ';', ':', '#', "'", '"', "_"]:
        text = text.replace(char, '')
        
    # Remove all words in lower case and stopwords, unless it looks like vintage (year) info.
    text = text.split()
    text = [word for word in text if word != word.lower() or
            ((word.isnumeric() and len(word) > 2) and (word[:2] == '19' or word[:2] == '20'))]
    text = [word for word in text if word.lower() not in stop_words]
    text = [word for word in text if len(word) > 1]

    text = ' '.join(text)
    
    #Convert all words to lower case
    text = text.lower()
    tokenizer = MWETokenizer(text,  separator='&')
    token_list = [token for token in tokenizer.tokenize(text.split())]
    bigram_list = list(bigrams(token_list))
    bigram_list = ["&".join(bigram) for bigram in bigram_list]
    token_list.extend(bigram_list)
    return token_list
    
def text_tokenizer(df, feature, training = True):
    token_list = df[feature].apply(individual_text_processing)
    if training:
        unique_multigrams = list(set(flatten_list(token_list)))
    else:
        unique_multigrams = None
    return token_list, unique_multigrams


class Create_Simple_Onehot():
    def __init__(self, feature = 'winery', threshold_frac = 0.0001):
        self.feature = feature
        self.threshold_frac = threshold_frac

    def fit(self, df, y = None):
        
        print(f'Creating simple onehot for feature = {self.feature}')
        df_column = pd.DataFrame(columns = [self.feature],
                                 data = df[self.feature].values,
                                 index = df.index)
        # onehotencoder = OneHotEncoder()
        min_frequency = int(self.threshold_frac*df.shape[0] + 0.5)
        # print(f'min_frequency = {min_frequency}')
        self.onehot = OneHotEncoder(min_frequency = min_frequency,
                                    sparse_output = False,
                                    handle_unknown = 'infrequent_if_exist',
                                    # handle_unknown = 'ignore'
                                   )
        self.onehot.fit(df_column)
        return self
    
    def transform(self, df, y = None):
        df_column = pd.DataFrame(columns = [self.feature],
                                 data = df[self.feature].values,
                                 index = df.index
                                )
        df_onehot = self.onehot.transform(df_column)
        df_onehot = pd.DataFrame(data = df_onehot, 
                                 index = df.index,
                                 columns= self.onehot.get_feature_names_out(), dtype='bool')
        # Last column is a catch all, so it is of no use.
        
        useless_column = df_onehot.columns[-1]
        df_onehot.drop(useless_column, axis = 1, inplace = True)
        
        df_output = pd.merge(df.drop(self.feature, axis = 1), df_onehot, left_index=True, right_index=True)
        print(f'feature onehotted = {self.feature}, output dataframe shape = {df_output.shape}')
        # Remove features with "None" in name
        df_output = df_output[[col for col in df_output.columns if "none" not in col.lower()]]
        return df_output
          

class Create_Multigrams():
    def __init__(self, tokenizer, feature = 'variety', threshold_frac = 0.01):
        self.feature = feature
        self.tokenizer = tokenizer
        self.threshold_frac = threshold_frac
        pass
        
    def fit(self, df, y = None):
        print(f'Creating multigrams for feature = {self.feature}')
        num_of_rows = df.shape[0]
        orig_multigram_list, self.unique_vocab  = self.tokenizer(df, self.feature)
        self.unique_vocab = sorted(self.unique_vocab)
        flattened_orig_multigram_list = flatten_list(orig_multigram_list)
        vocab_count_dict = collections.Counter(flattened_orig_multigram_list)

        #min threshold_count must be 1
        if self.threshold_frac * df.shape[0] < 1:
            self.threshold_frac = 20/df.shape[0]
        
        self.filtered_unique_vocab = []
        for unique_vocab in self.unique_vocab:
            frac_of_trues = vocab_count_dict[unique_vocab]/num_of_rows
            # if frac_of_trues is None:
            #     print(unique_vocab, frac_of_trues, self.threshold_frac, end = ', ')
                
            if frac_of_trues >= self.threshold_frac: 
                self.filtered_unique_vocab.append(unique_vocab)
                vocab_count_dict[unique_vocab] = frac_of_trues
            else:
                # print("")
                pass
        return self
        
    def transform(self, df, y = None):
        '''For each text entry, remove duplicated words and output a tuple of the split words'''
        length = df.shape[0]
        index_output = df.index
        multigram_list, _ = self.tokenizer(df, self.feature)
        filtered_multigram_list = [[monogram for monogram in multigram if monogram in self.filtered_unique_vocab] 
                                   for multigram in multigram_list]
        df_onehot_list = []
        for unique_word in self.filtered_unique_vocab[::]:
            unique_word_in_data = [x for x in map(lambda filtered_multigram: unique_word in filtered_multigram, filtered_multigram_list)]
            df_onehot_word = pd.DataFrame(columns = [f'{self.feature}_{unique_word}'],
                                          data = unique_word_in_data,
                                          index = index_output)
            df_onehot_list.append(df_onehot_word)
        
        df_onehot = pd.concat(df_onehot_list, axis = 1)
        df_output = pd.merge(df.drop(self.feature, axis = 1), df_onehot, left_index=True, right_index=True)
        print(f'feature tokenized = {self.feature}, output dataframe shape = {df_output.shape}')

        
        # Remove features with "None" in name
        df_output = df_output[[col for col in df_output.columns if "none" not in col.lower()]]
            
        return df_output 

class Merge_Similar_Columns():
    def __init__(self):
        pass

    def fit(self, df, y = None):
        self.df = df
        #suffices are actual feature names after the _
        suffices = [col.split("_")[-1] for col in df.columns]

        # Count for each suffix how many times it was deplicated
        num_appearances = [(suffix, np.sum([(suffix.lower() == col.lower()) for col in suffices]))
                           for suffix in suffices]
        self.duplicated_features = set([feature[0].lower() for feature in num_appearances if feature[1] > 1])
        print(f'Duplicated columns to be merged are {self.duplicated_features}.')
        return self

    def transform(self, df, y = None):
        print(f'Merging the following raw columns: {self.duplicated_features}')
        for duplicated_feature in self.duplicated_features:
            duplicated_feature_set = [col for col in df.columns 
                                      if duplicated_feature.lower() == col.lower().split("_")[-1]]
            print(f'Merging the following columns into one: {duplicated_feature_set}.')
            df.loc[:, f'merged-{string.capwords(duplicated_feature, sep = None) }'] = df[duplicated_feature_set].sum(1) > 0
            # print(df.loc[:, f'merged-{string.capwords(duplicated_feature, sep = None) }'])
            df.drop(duplicated_feature_set, axis = 1, inplace = True)
            print(f'Number of columns after merging = {df.shape[1]}')
            # print('')
            
            # Defragment pandas dataframe.
            df = df.copy()
        return df 


class Normalize_Points():
    def __init__(self, point_feature = 'points', groupby_feature = 'taster-name', drop_orig_data = True):
        self.point_feature = point_feature
        self.groupby_feature = groupby_feature
        self.drop_orig_data = drop_orig_data
        pass
        
    def fit(self, df, y = None):
        print(f'Fitting normalized_points for point_features = {self.point_feature}, groupby {self.groupby_feature}')
        df_raw = df[[self.point_feature, self.groupby_feature]]
        df_std = df_raw.groupby(self.groupby_feature).std()[self.point_feature]
  
        df_mean = df_raw.groupby(self.groupby_feature).mean()[self.point_feature]

        self.df_grouped = pd.merge(df_std, df_mean, left_index= True, right_index=True)
        
        self.df_grouped = pd.DataFrame(columns = ['stddev_points', 'mean_points'],
                                     data = self.df_grouped.values,
                                    index = self.df_grouped.index)
        return self
        
    def transform(self, df, y = None):
        '''For each text entry, remove duplicated words and output a tuple of the split words'''
        print('Calculating normalized point for each taster based on training data')

        df_output = pd.merge(df, self.df_grouped, left_on= "taster-name", right_index = True)
        df_output["norm-points"] = (df_output["points"] - df_output["mean_points"])/df_output["stddev_points"]
        df_output.drop(['stddev_points', 'mean_points'], axis = 1, inplace = True)
        if self.drop_orig_data:
            df_output.drop(['points'], axis = 1, inplace = True)
        return df_output 

