import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.logger import logging

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

#Route for a Home Page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('prediction.html',results=None)
    else:
        data=CustomData(
            Age= float(request.form.get('Age')),
            No_of_Pregnancy= float(request.form.get('No_of_Pregnancy')),
            Gestation_in_previous_Pregnancy= request.form.get('Gestation_in_Previous_Pregnancy'),
            BMI= float(request.form.get('BMI')),
            HDL= float(request.form.get('HDL')),
            Family_History= request.form.get('Family_History'),
            unexplained_prenetal_loss= request.form.get('unexplained_prenetal_loss'),
            Large_Child_or_Birth_Default= request.form.get('Large_Child_or_Birth_Default'),
            PCOS= request.form.get('PCOS'),
            Sys_BP= float(request.form.get('Sys_BP')),
            Dia_BP= float(request.form.get('Dia_BP')),
            OGTT= float(request.form.get('OGTT')),
            Hemoglobin= float(request.form.get('Hemoglobin')),
            Sedentary_Lifestyle= request.form.get('Sedentary_Lifestyle'),
            Prediabetes= request.form.get('Prediabetes')

        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        if(results[0]==0.0):
            msg = "No"
        else:
            msg = "Yes"
        return render_template('prediction.html',results=msg)


if __name__=="__main__":
    app.run(host="0.0.0.0")
