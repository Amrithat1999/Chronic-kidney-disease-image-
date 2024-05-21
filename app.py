from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'C:\Users\dell\Downloads\mediplus-lite\static\uploads'
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

@app.route("/")
def main():
    return render_template('contact.html')

@app.route('/overview')
def overview():
    return render_template('index.html')

@app.route('/overmore')
def overmore():
    return render_template('overmore.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/predictionpage')
def predictionpage():
    return render_template('predictionpage.html')

def predict_image(image_path):
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to the target size
    img = cv2.resize(img, (180, 180))

    img_array = np.array(img, dtype=np.float32) # Convert the array to float32
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1) # Add an extra dimension for the grayscale channel
    img_array /= 255.0 # Normalize the image

    # Load the model and make prediction
    model_path = 'KD_Model.h5'
    model = load_model(model_path)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    return predicted_label

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predicted_label = predict_image(file_path)
        pred = class_names[predicted_label]
        if pred=='Stone':
            con='Kidney stones are solid mineral deposits that form in the kidneys and can vary in size. Passing through the urinary tract, they cause intense pain.'
        elif pred=='Tumor':
            con='A kidney tumor, also known as a renal tumor, is an abnormal growth that develops in the kidney.'
        elif pred=='Cyst':
            con='A kidney cyst is a fluid-filled sac that can develop in the kidney. These cysts are typically non-cancerous and may not cause symptoms'
        else:
          con='Normal kidney health.' 
        return render_template('predictionpage.html', prediction=pred, image_loc=filename, con=con)

    return render_template('predictionpage.html')


if __name__ == '__main__':
    app.run(debug=True)
