from flask import Flask , render_template , jsonify, request
import numpy as np
import sklearn
import pickle


crop_recommendation_model_path = 'models\crop_recommendation.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

deficiency_prediction_model_path = 'models\Deficiency_prediction.pkl'
deficiency_prediction_model = pickle.load(
    open(deficiency_prediction_model_path, 'rb'))




app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/nd')
def first():
    return render_template("nutrition_deficiency.html")

@app.route('/crop-recc')
def second():
    return render_template("crop_recommendation.html")




#-------------------------------------------------------------

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
       if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        T = float(request.form['temprature'])
        H = float(request.form['humidity'])
        ph = float(request.form['ph'])
        
        
        data = np.array([[N, P, K, T, H, ph]])
        my_prediction =crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('recommend_result.html', prediction=final_prediction)
       

@ app.route('/crop-deficiency', methods=['POST'])
def crop_deficiency():
       if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        ph = float(request.form['ph'])
        ph = float(request.form['ph'])
        CT= object(request.form['cropname'])
        
        data = np.array([[N, P, K,  ph, CT]],dtype=object)
        my_prediction =deficiency_prediction_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('deficiency_result.html', prediction=final_prediction)
        
    
	
if __name__ == "__main__":
    app.run(port=5555)