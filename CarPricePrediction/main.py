from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('RandomForestRegressionModel.pkl', 'rb'))
car = pd.read_csv('carpricepred_cleaned.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    names = sorted(car['Name'].unique())
    car_models = sorted(car['Model'].unique())
    year = sorted(car['Year'].unique(),reverse=True)

    names.insert(0, 'Select Car Name')

    return render_template('index.html', names=names, car_models=car_models, years=year)


@app.route('/test', methods=['GET'])
@cross_origin()
def test():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    name = request.form.get('name')
    car_model = ' '.join(request.form.get('car_models').split()[1:])
    year = request.form.get('year')
    mileage = request.form.get('miles_driven')
    prediction = model.predict(pd.DataFrame(columns=['Model', 'Name', 'Year', 'Mileage'],
                                            data=np.array([car_model, name, year, mileage]).reshape(1, 4)))
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()