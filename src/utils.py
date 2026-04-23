import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as fileObj:
            dill.dump(obj,fileObj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models,params):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():

            para = params[model_name]

            model_new = RandomizedSearchCV(
                estimator=model,
                param_distributions=para,
                n_iter=10,
                cv=3,
                scoring="r2",
                n_jobs=-1,
                verbose=1,
                random_state=42
            )

            model_new.fit(X_train, Y_train)

            best_model = model_new.best_estimator_

            y_test_pred = best_model.predict(X_test)
            score = r2_score(Y_test, y_test_pred)

            report[model_name] = score
            trained_models[model_name] = best_model

        return report, trained_models
    except Exception as e:
        raise CustomException(e,sys)
