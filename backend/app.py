from flask import Flask,request,jsonify
import pandas as pd
import joblib
app = Flask(__name__)

model = joblib.load("xgb_model.joblib")

@app.route("/data",methods =["POST"])
def hello():
    data = request.get_json()
    prediction_data = pd.DataFrame(data)
    prediction= model.predict(prediction_data)
    prediction_list = prediction.tolist()
    return jsonify(prediction_list)
    

if __name__ == "__main__":
    app.run(debug=True)