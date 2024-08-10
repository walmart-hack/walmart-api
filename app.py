from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import uuid

from mongo_client import *

load_dotenv()

app = Flask(__name__)
fs = GridFS(db)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/convert-to-grid', methods=['POST'])
def convert_to_grid():
    if 'image' not in request.files:
        return jsonify({ "error": "No image file found in the request" }), 400

    file = request.files['image']

    unique_name = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    grid = np.where(binary_image == 255, 0, 1)

    # Compress and store the grid in GridFS
    grid_bytes = pickle.dumps(grid)
    file_id = fs.put(grid_bytes, filename="large_grid.pkl")

    metadata = {
        "image_name": unique_name,
        "dimensions": {"height": grid.shape[0], "width": grid.shape[1]},
        "grid_file_id": file_id
    }

    collection.insert_one(metadata)

    return jsonify({"file_name": unique_name, "shape": grid.shape}), 200


@app.route('/get-grid', methods=['GET'])
def get_grid():
    image_name = request.args.get('image_name')
    metadata = collection.find_one({"image_name": image_name})

    if metadata:
        file_id = metadata['grid_file_id']

        grid_file = fs.get(file_id)
        grid_bytes = grid_file.read()
        
        grid = pickle.loads(grid_bytes)

        return jsonify({"shape": grid.shape}), 200
    else:
        return jsonify({ "error": "Image not found" }), 404



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
