from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error

import importlib
import matplotlib.pyplot as plt

import numpy as np

rfm_params = {'n_estimators': 80, 
             'max_features': '0.5',
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
                  'verbose': 100, # Tree intervals to print eval errors
                  'early_stopping_rounds' : 20
                 }

class Tree_Model(): 
    def __init__(self, Model, params, price_max = 300
                 ):
        self.price_max = price_max,  # For plotting only
        self.params = params
        self.model = Model()
        self.model.set_params(**self.params)

    def fit(self,X_train, y_train_actual):
            
        self.price_max = self.price_max[0] # For plotting only
        self.y_data_type = type(y_train_actual)
        y_train_actual = y_train_actual.values

        
        if 'early_stopping_rounds' in self.params.keys():
            self.model.fit(X_train, y_train_actual,
                           verbose = self.params['verbose'],
                           eval_set = self.params['eval_set'])
        else:
            self.model.fit(X_train, y_train_actual)
            
        print("Finished fitting.  Predicting X...")
        y_train_predict = self.model.predict(X_train)
        print("Finished predicting X.")
        self.MdAPE_train = np.median(np.abs(y_train_predict - y_train_actual)/y_train_actual)*100
        print(f'Training MdAPE is {self.MdAPE_train}% (not holdout).')
        self.feature_imp = sorted([feature for feature in zip(X_train.columns, self.model.feature_importances_)],
                                  key = lambda feature: feature[1], reverse = True)
        print("Finished training mdoel")

        print("Plotting scatter plot..")
        rmse, mae, score = np.sqrt(mean_squared_error(y_train_actual, y_train_predict)), \
                                    mean_absolute_error(y_train_actual, y_train_predict), \
                                    r2_score(y_train_actual, y_train_predict)
        print("R2: %5.3f, RMSE: %5.3f, MAE: %5.3f" %(score, rmse, mae))
        
        plt.figure(figsize = (12, 12))
        plt.title("Training data set")
        plt.scatter(y_train_actual, y_train_predict, alpha = 0.002, color = 'r')
        plt.xlabel("Price actual ($)")
        plt.ylabel("Price predicted ($)")
        plt.grid()
        plt.xlim(0, self.price_max)
        plt.ylim(0, self.price_max)
        plt.show()

        return self.model
        
    def predict(self, X, y_actual = None):
        print(type(y_actual))
        print(f'type(y_actual) = {type(y_actual)}')
        y_actual_data_type = type(y_actual)
        print(f"type(y_actual) = {y_actual_data_type}")
        print(f"self.y_data_type = {self.y_data_type}")
        y_predict = self.model.predict(X)

        # If y_actual is not None, then calculate the holdout errors and plot results
        if y_actual_data_type == self.y_data_type:
            
            y_actual = y_actual.values
            self.MdAPE = np.median(np.abs(y_predict - y_actual)/y_actual)*100
            print(f'MdAPE is {self.MdAPE}%')

        
            print("Plotting scatter plot..")
            rmse, mae, score = np.sqrt(mean_squared_error(y_actual, y_predict)), \
                                        mean_absolute_error(y_actual, y_predict), \
                                        r2_score(y_actual, y_predict)
            print("R2: %5.3f, RMSE: %5.3f, MAE: %5.3f" %(score, rmse, mae))
            
            plt.figure(figsize = (12, 12))
            plt.title("Holdout data set")
            plt.scatter(y_actual, y_predict, alpha = 0.01, color = 'blue')
            plt.xlabel("Price actual ($)")
            plt.ylabel("Price predicted ($)")
            plt.grid()
            plt.xlim(0, self.price_max)
            plt.ylim(0, self.price_max)
            plt.show()

        return y_predict
