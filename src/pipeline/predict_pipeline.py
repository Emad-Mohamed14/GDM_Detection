import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model, feature_names = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)

            # Ensure only the expected features are passed to the model
            features = features[feature_names]
            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e , sys)
        


class CustomData:
    def __init__(self,
        Age: int,
        No_of_Pregnancy: int,
        Gestation_in_previous_Pregnancy: int,
        BMI: int,
        HDL: int,
        Family_History: int,
        unexplained_prenetal_loss: int,
        Large_Child_or_Birth_Default: int,
        PCOS: int,
        Sys_BP: int,
        Dia_BP: int,
        OGTT: int,
        Hemoglobin: int,
        Sedentary_Lifestyle: int,
        Prediabetes: int):
            
        self.Age = Age
        self.No_of_Pregnancy = No_of_Pregnancy
        self.Gestation_in_previous_Pregnancy = Gestation_in_previous_Pregnancy
        self.BMI = BMI
        self.HDL = HDL
        self.Family_History = Family_History
        self.unexplained_prenetal_loss = unexplained_prenetal_loss
        self.Large_Child_or_Birth_Default = Large_Child_or_Birth_Default
        self.PCOS = PCOS
        self.Sys_BP = Sys_BP
        self.Dia_BP = Dia_BP
        self.OGTT = OGTT
        self.Hemoglobin = Hemoglobin
        self.Sedentary_Lifestyle = Sedentary_Lifestyle
        self.Prediabetes = Prediabetes
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "No_of_Pregnancy": [self.No_of_Pregnancy],
                "Gestation_in_previous_Pregnancy": [self.Gestation_in_previous_Pregnancy],
                "BMI": [self.BMI],
                "HDL": [self.HDL],
                "Family_History": [self.Family_History],
                "unexplained_prenetal_loss": [self.unexplained_prenetal_loss],
                "Large_Child_or_Birth_Default": [self.Large_Child_or_Birth_Default],
                "PCOS": [self.PCOS],
                "Sys_BP": [self.Sys_BP],
                "Dia_BP": [self.Dia_BP],
                "OGTT": [self.OGTT],
                "Hemoglobin": [self.Hemoglobin],
                "Sedentary_Lifestyle": [self.Sedentary_Lifestyle],
                "Prediabetes": [self.Prediabetes]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
