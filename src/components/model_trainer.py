import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from dataclasses import dataclass
import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, X_train, y_train, X_test, y_test, model):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        roc_auc = roc_auc_score(y_test, y_pred_test)
        
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Initialize the model
            model = GradientBoostingClassifier()

            # Evaluate the model
            model_report = self.evaluate_model(X_train, y_train, X_test, y_test, model)
            print(model_report)
            print('\n' + '='*80 + '\n')
            logging.info(f'Model Report : {model_report}')

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            print(f'gradientboosting classifier Model Trained and Saved successfully')
            print('\n' + '='*80 + '\n')
            logging.info(f'gradientboosting Model Trained and Saved successfully')

        except Exception as e:
            logging.info('Exception occurred during Model Training')
            raise CustomException(e, sys)