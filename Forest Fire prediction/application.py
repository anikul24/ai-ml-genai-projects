import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application

#import pickle files
model = pickle.load(open('models/ridge.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html') ## render_template looks for template folder


@app.route("/predictdata", methods=['GET','POST'])
def predict_datapoint():
    if request.method =='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data_scaled=scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        print(new_data_scaled.shape)

        result = model.predict(new_data_scaled)

        print("TYPE OF Model : ",type(model))

        print(hasattr(model, 'predict'))

        return render_template('home.html',result=result[0])
    

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")