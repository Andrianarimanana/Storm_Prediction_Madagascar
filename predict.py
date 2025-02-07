import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_xboost.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('storm')

@app.route('/predict', methods=['POST'])
def predict():
    meteo_data = request.get_json()

    X = dv.transform([meteo_data])
    y_pred = model.predict_proba(X)[0, 1]
    storm = y_pred 

    result = {
        # storm_probability
        'storm_probability': float(y_pred),
        'storm': bool(storm)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)