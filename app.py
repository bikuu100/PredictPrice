from flask import Flask, render_template, request
import sys
sys.path.append("src/pipeline") 
from predict_pipeline import predict_pipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input data from the form
        input_data = {
            'screen_size': request.form['screen_size'],
            'main_camera_mp': request.form['main_camera_mp'],
            'selfie_camera_mp': request.form['selfie_camera_mp'],
            'int_memory': request.form['int_memory'],
            'ram': request.form['ram'],
            'battery': request.form['battery'],
            'weight': request.form['weight'],
            'release_year': request.form['release_year'],
            'days_used': request.form['days_used'],
            'new_price': request.form['new_price'],
            'brand_name': request.form['brand_name'],
            'os': request.form['os'],
            '4g': request.form['4g'],
            '5g': request.form['5g']
        }
        
        # Predict using predict_pipeline
        prediction = predict_pipeline(input_data)
        
        # Render prediction result template with the prediction
        return render_template('prediction_result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
