from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error #, mean_absolute_percentage_error

import importlib
import matplotlib.pyplot as plt

import numpy as np

rfm_params = {'n_estimators': 80, 
             'max_features': 0.5,
             'max_depth': 300,
             'bootstrap': True,
             'criterion': "squared_error",
             'verbose': 1,
             'oob_score': True,
             'n_jobs': 8
             }
xgboost_params = {'n_estimators' : 1000000,
                  'max_depth' : 6,
                  'verbosity' : 3,
                  'n_jobs' : 8, 
                  'eval_metric': 'rmse', 
                  'colsample_bytree': 0.5,
                  'colsample_bynode': 0.5,
                  'learning_rate': 0.0025,
                  'verbose': 1000, # Tree intervals to print eval errors
                  'early_stopping_rounds' : 20
                 }

class Tree_Model(): 
    ''' This function instantiate an RFM (sklearn) or gbm (xgboost) model and paraments,
    trains a model, and and outputs the training scores.
    '''

    def __init__(self, Model, params, price_max = 300
                 ):
        '''
        input:
        Model (sklearn.ensemble or xgboost class): a tree tree.
        params (dict):  parameters to be input into the model
        price_max (int or float): max for the output plot (only).  Not used in model and culling data.
        '''
    
        self.price_max = price_max,  # For plotting only
        self.params = params
        self.model = Model()
        self.model.set_params(**self.params)

    def fit(self,X_train, y_train_actual):
        '''
        input:
        X_train (pandas Dataframe): X data for training
        y_train_actual (pandas Dataframe or numpy array (1d): y data for training
        
        output:
        model: trained model
        '''
            
        self.price_max = self.price_max[0] # For plotting only
        self.y_train_actual_type = type(y_train_actual)
        y_train_actual = y_train_actual.values

        # if there is early stopping rounds, it is a gbm model.
        # if not, it is an rfm model.
        if 'early_stopping_rounds' in self.params.keys():
            self.model.fit(X_train, y_train_actual,
                           verbose = self.params['verbose'],
                           eval_set = self.params['eval_set'])
        else:
            self.model.fit(X_train, y_train_actual)

        
        #Calculate Training error.    
        print("Finished fitting.  Predicting X...")
        y_train_predict = self.model.predict(X_train)
        print("Finished predicting X.")
        self.MdAPE_train = np.median(np.abs(y_train_predict - y_train_actual)/y_train_actual)*100
        print(f'Training MdAPE is {self.MdAPE_train}% (not holdout).')
        self.feature_imp = sorted([feature for feature in zip(X_train.columns, self.model.feature_importances_)],
                                  key = lambda feature: feature[1], reverse = True)
        print("Finished training mdoel")

        error_calculation_output(y_train_actual, y_train_predict)

        #Plot training results
        plot_scatter_plot(y_train_actual, 
                          y_train_predict, 
                          "Training data set", 
                          price_max = self.price_max, 
                          color = 'r')

        return self.model
        
    def predict(self, X, y_actual = None):
        y_predict = self.model.predict(X)

        '''
        input:
        X (pandas Dataframe):  X data to be predicted.
                               If there's only one row, input in the following format.
                               X.iloc[row_number, row_number + 1, :]
        y (pandas Dataframe or np array:  y data

        output:
        y_predict
        '''
        # If y_actual is not None, then calculate the holdout errors and plot results
        y_actual_data_type = type(y_actual)
        if y_actual_data_type == self.y_train_actual_type:
            y_actual = y_actual.values
            
            error_calculation_output(y_actual, y_predict)

            #Ploting holdout results
            plot_scatter_plot(y_actual, 
                              y_predict, 
                              "Holdout data set", 
                              price_max = self.price_max, 
                              color = 'b')

        return y_predict

def plot_scatter_plot(y_actual, y_pred, title = "Holdout data set" , price_max = 300, color = 'r'):  
        print("Plotting scatter plot..")
        plt.figure(figsize = (12, 12))
        plt.title(title)
        plt.scatter(y_actual, y_pred, alpha = 0.01, color = color)
        plt.xlabel("Price actual ($)")
        plt.ylabel("Price predicted ($)")
        plt.grid()
        plt.xlim(0, price_max)
        plt.ylim(0, price_max)
        plt.show()

def error_calculation_output(y_actual, y_predict):
    # Calculate MdAPE
    MdAPE = np.median(np.abs(y_predict - y_actual)/y_actual)*100
    print(f'MdAPE is {MdAPE}%')
    rmse, mae, score = np.sqrt(mean_squared_error(y_actual, y_predict)), \
                                mean_absolute_error(y_actual, y_predict), \
                                r2_score(y_actual, y_predict)
    print("R2: %5.3f, RMSE: %5.3f, MAE: %5.3f" %(score, rmse, mae))
