from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set up the upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/convert-to-grid', methods=['POST'])
def convert_to_grid():
    if 'image' not in request.files:
        return jsonify({ "error": "No image file found in the request" }), 400

    file = request.files['image']

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    grid = np.where(binary_image == 255, 0, 1)

    grid_list = grid.tolist()
    # return jsonify({"grid": grid_list})
    return jsonify({"shape": grid.shape}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
