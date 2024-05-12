import os,cnn

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
tomato_checker = cnn.TomatoChecker(
    model_path=os.path.join("model", "tomatoes.keras"),
    data_cat=['tomatoes_fresh', 'tomatoes_fresh_medium', 'tomatoes_rotten'],
    img_width=180,
    img_height=180
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['GET'])
def check():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        result = tomato_checker.predict_tomato_type(filename)
        
        for key in result:
            if isinstance(result[key], np.float32):
                result[key] = round(float(result[key])*100,4)
        result['image_url'] = url_for('uploaded_file', filename=file.filename)
        print(result)
        return jsonify(result)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == '__main__':
    app.run(debug=True)