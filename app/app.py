

#OJOOOOOOOO Este solo es prueba local sin usar redis, pero para que corra en flask tiene que tener el nombre de "app.py"


from flask import Flask, render_template, request
import pickle
import numpy as np

app= Flask(__name__)

@app.route('/')
@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

#Prediction function
def Predictor(toPred_list):
    toPred = np.array(toPred_list).reshape(1,10)
    model = pickle.load(open("best_model.pkl", "rb")) #Loading the model
    result = model.predict(toPred)
    return result[0]

@app.route('/result', methods = ['POST'])
def result():
    if request.method=='POST':
        toPred_list = request.form.values()
        toPred_list = list(map(float, toPred_list))
        toPred_list[7] = np.log(toPred_list[7] + 1)
        toPred_list[8] = np.log(toPred_list[8] + 1)
        toPred_list[9] = np.log(toPred_list[9] + 1)
        result = Predictor(toPred_list)

        if result == 0:
            prediction='will not be approved'

        if result == 1:
            prediction='loan will be approved'

        return render_template("result.html", prediction=prediction, values=toPred_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
