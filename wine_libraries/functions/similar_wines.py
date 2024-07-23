import pandas as pd
# pd.set_option('display.max_colwidth', None)

import numpy as np
import torch
import gzip
# import os
import pickle

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

info_features = ['title',  'designation', 'variety', 'region-1','region-2', 'province', 'country', 'winery']

# These are my personally tailored stop words that include foreign articles such as 'la', 'el', 'les', 'il'
from functions.featurization import stop_words
# About 150 typical English stopwords
nltk_words = list(stopwords.words('english'))
# Remove negative words from stop word list.  In other words, I want to keep them in the text.

# We want to KEEP negative words.  I believe those words are important in conveying negative sentiments.
nltk_words = [word for word in nltk_words if 'not' not in word and word[-3:] != "n't" and word[-2:] != "n'"][:141]
nltk_words.remove('nor')
nltk_words.remove('no')
nltk_words.remove('don')
nltk_words.remove('t')

# Add the nltk words to my original stop word list
stop_words.extend(nltk_words)

def find_similarities(df_all_wines_embeds, 
                      all_wines_prices, 
                      indiv_wine_embed, 
                      indiv_win_price, 
                      price_importance = 1, 
                      similarity_function = "cosine"
                     ):   
    '''This functions takes in the price and the vectorized description embeddings of a wine of interest, 
    and calculates and outputs the similarities in price and description between it and all wines in the data set.
    
    Input:
    df_all_wines_embeds (pandas dataframe): Embedings matrix of the description
    all_wines_prices (numpy vector or pandas series or dataframe): prices for all wines in the dataset.
    indiv_wine_embed (numpy vector or pandas series or dataframe): wine description embeddings.
    indiv_win_price (float or int): price of wine of interest.
    price_importance (0 ≤ float ≤ 1): importance of the price similarity compared to the description similarity.
    similarity_function (str, default = "cosine"):  "cosine" for cosine similarity, "distance" for euclidean distance.

    Output:
    df_similarities (pandas dataframe): the description, price, and overall similarities between the wine of interest and all wines.
    '''
    if similarity_function == "cosine":
        print(similarity_function)
        similarity_function = cosine_similarity
    elif similarity_function == "distance":
        print(similarity_function)
        similarity_function = euclidean_distances
    else:
        print
        raise Exception("Unrecognized similarity_function: Please enter either 'cosine' or 'distance'")
    indices = df_all_wines_embeds.index
    
    # Calculate description similarities between  wine of interest and all wines
    desc_sim = df_all_wines_embeds.apply(lambda X: similarity_function(indiv_wine_embed.reshape(1,-1), 
                                                                     X.values.reshape(1, -1) )[0][0], 
                                         axis = 1)
        
    # Calculate price similarities between  wine of interest and all wines
    # We use the rms % deviation from the wine_of_interest price to represent the price similarity.
    # A price_importance term is included to the exponential to allow for the weighting of this term.  
    # A larger price_importance term would make the distribution of price_sim broader, thereby increasing its importance in the 
    # overall_similarity calculation.
    price_sim = np.exp(-np.square(indiv_win_price - all_wines_prices)*price_importance/indiv_win_price)
    overall_similarity = desc_sim * price_sim
    
    df_similarities = pd.DataFrame(columns=["desc_sim", "price_sim", "overall_similarity"],
                                   data = zip(desc_sim, price_sim, overall_similarity),
                                   index = indices)
    return df_similarities


def find_your_wine(df_all, 
                   your_wine_text, 
                   df_all_wines_embeds, 
                   text_tranformer, 
                   your_price, 
                   price_importance = 1, 
                   similarity_function = "cosine",
                   output_verbose = False):
    '''This function allows you to create a descriptive text and a price for a wine you like, and it looks for the best match.

    Input:
    df_all:
    your_wine_text (str): description of the wine you like.
    df_all_wines_embeds: description embeddings of the wine data set.
    text_tranformer (transformer model): huggingface sentence_transformers.SentenceTransformer model (instatiated).
    your_price (float or int): the price of the wine you like.
    price_importance (0 ≤ float ≤ 1): importance of the price similarity compared to the description similarity.
    similarity_function (str, default = "cosine"):  "cosine" for cosine similarity, "distance" for euclidean distance.

    Output:
    df_merged (pandas dataframe): dataframe containing the modified description text and the three similarities.
    '''
    
    # Drop all non embed columns
    drop_columns = [col for col in df_all_wines_embeds.columns if 'embed' not in col]
    df_all_wines_embeds.drop(drop_columns, axis = 1, inplace = True)

    # Remove stop words from your_wine_text
    your_wine_text = " ".join( [w for w in " ".join([w for w in your_wine_text.split() if not w in stop_words]).split() if not w in stop_words])
    # Encode your_wine_text into embedding vector\
    desc_embeds = text_tranformer.encode(your_wine_text, convert_to_tensor=False)

    all_wines_prices = df_all["price"].values

    # Use find_similarity function to calculate the similarity values between your text/price 
    # and all other wines'.
    df_similarities = find_similarities(df_all_wines_embeds, 
                                        all_wines_prices, 
                                        desc_embeds, 
                                        your_price, 
                                        price_importance, 
                                        similarity_function = similarity_function
                                        )

    # Merge results with the modified description.  Sort by overall_similarity.
    if output_verbose == False:
        df_all = df_all.drop(info_features, axis =1)
    df_merged = pd.merge(df_all, 
                         df_similarities, 
                         left_index=True,
                         right_index=True).sort_values("overall_similarity", ascending=False)
    return df_merged
    

def most_similar_wine_within_dataset(df_all, 
                                     df_all_wines_embeds, 
                                     wine_of_interest_index, 
                                     price_importance = 1, 
                                     similarity_function = 'cosine',
                                     output_verbose = False):
    '''This function assumes that someone is interested in a certain wine in the data set.
    1.  Takes the index number of that wine, and calculates the embeddings for the description (review).
    2.  Calculates the cosine similarity of #1 with the pre-calculated description embeddings of the review corpus.
    3.  Calculates the prices similarity (the variance from the wine_of_interest price, with a importance factor for weighting).
    4.  Calculates the overall similarity from the product of #2 and #3.
    5.  Output the results.
    
    Inputs:
    df_all (pandas dataframe): data frame w/ merged text of description with other features, but w/o the embedding.
    df_all_wines_embeds (pandas dataframe)::  Data frame of the embeddings of the description.
    wine_of_interest_index (int):  Index of the wine of interesting on their table 
    price_importance (0 ≤ float ≤ 1): importance of the price similarity compared to the description similarity.
    similarity_function (str, default = "cosine"):  "cosine" for cosine similarity, "distance" for euclidean distance.

    Output:
    df_merged (pandas dataframe): data with with description and similarity values, sorted by overall similarity.
    '''
    # Drop all non embed columns
    drop_columns = [col for col in df_all_wines_embeds.columns if 'embed' not in col]
    df_all_wines_embeds.drop(drop_columns, axis = 1, inplace = True)

    # Get description embed vector of wine of interest
    wine_of_interest_embed = df_all_wines_embeds.loc[wine_of_interest_index,:].values
    print(wine_of_interest_index)

    wine_of_interest_embed.reshape(-1, 1)
    # Get price of wine_of_interest
    wine_of_interest_price = df_all.loc[wine_of_interest_index,:]["price"]
    
    all_wines_prices = df_all["price"].values

        
    df_similarities = find_similarities(df_all_wines_embeds, 
                                        all_wines_prices, 
                                        wine_of_interest_embed, 
                                        wine_of_interest_price, 
                                        price_importance, 
                                        similarity_function = similarity_function
                                        )

    if output_verbose == False:
        df_all = df_all.drop(info_features, axis =1)
    df_merged = pd.merge(df_all, 
                         df_similarities, 
                         left_index=True,
                         right_index=True
                        ).sort_values("overall_similarity", ascending = False)

    return df_merged