import os,cnn

import numpy as np
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
tomato_checker = cnn.TomatoChecker(
    model_path=os.path.join("model", "tomatoes.keras"),
    data_cat=['tomatoes_fresh', 'tomatoes_fresh_medium', 'tomatoes_rotten'],
    img_width=180,
    img_height=180
)
@app.route('/', methods=['GET'])
def upload_file():
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
                result[key] = round(float(result[key]),5)
        
        print(result)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)