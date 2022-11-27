from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import joblib



app = Flask(__name__)
model=joblib.load('XGBoost_Regressor_model.pkl')

@app.route('/')
def home():
   return render_template('csc.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        f_list=[request.form.get('cement'),request.form.get('blast_furnace'),request.form.get('fly_ash'),request.form.get('water'),request.form.get('superplasticizer'),request.form.get('coarse_aggregate'),
        request.form.get('fine_aggregate'),request.form.get('age')]


        final_features=np.array(f_list).reshape(-1,8)
      
        

        prediction=model.predict(final_features)
        result="%.2f" % round(prediction[0],2)

        return render_template('csc.html',prediction_text=f"Compresive Strength of Concrete : {result} MPa")


if __name__ == "__main__":
    app.run(debug=True)
