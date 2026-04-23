# Basic Import
import os
import sys
import numpy as np
import pandas as pd

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.utils import evaluate_model,save_object
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Model training started")
            X_train,Y_train,X_test,Y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNN":KNeighborsRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "SVM":SVR(),
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso()
                
            }
            model_report:dict=evaluate_model(X_train,Y_train,X_test,Y_test,models)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best Model found",sys)
            
            logging.info("Model Training Done")

            save_object(self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(X_test)
            accuracy=r2_score(Y_test,predicted)
            return accuracy
            
        except Exception as e:
            raise CustomException(e,sys)
            

