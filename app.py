import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('house_price.pkl', 'rb'))
iphone_model = pickle.load(open('linear_regression_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

# Navigated to home page
@app.route('/home.html', methods=['GET'])
def homee():
    return render_template('home.html')

# Navigated to House Price Prediction form:-
@app.route('/house.html', methods=['GET'])
def house():
    return render_template('house.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    area = int(request.form['area'])
    data = np.array([[area]])
    prediction = model.predict(data)

    return render_template('house.html', prediction_text='House Price could be Rs {}'.format(prediction))


# Navigated to Iphone Price Prediction form:-
@app.route('/phone.html', methods=['GET'])
def iphone():
    return render_template('phone.html')

@app.route('/predict_phone', methods=['POST'])
def predict_phone():
    '''
    For rendering results on HTML GUI
    '''
    version = int(request.form['version'])
    model_version = np.array([[version]])
    iphone_value = iphone_model.predict(model_version)

    return render_template('phone.html', prediction_iphone = 'Iphone Price could be Rs {}'.format(iphone_value))


if __name__ == "__main__":
    app.run(debug=True)
