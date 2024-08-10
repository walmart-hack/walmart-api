from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import uuid
from dotenv import load_dotenv

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

    height, width = grid.shape
    image = np.zeros((height, width), dtype=np.uint8)
    image[grid == 0] = 255
    image[grid == 1] = 0 
    cv2.imwrite('./uploads/converted_image.png', image)

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


        # for i in range(grid.shape[0]):
        #     for j in range(grid.shape[1]):
        #         print(grid[i][j], end=' ')
        #     print()

        return jsonify({"shape": grid.shape}), 200
    else:
        return jsonify({ "error": "Image not found" }), 404


@app.route('/insert-coordinates', methods=['POST'])
def insert_coordinates():
    image_name = request.args.get('image_name')
    metadata = collection.find_one({"image_name": image_name})

    if metadata:
        file_id = metadata['grid_file_id']

        grid_file = fs.get(file_id)
        grid_bytes = grid_file.read()
        
        grid = pickle.loads(grid_bytes)
        
        num_labels, labels_im = cv2.connectedComponents(grid.astype(np.uint8))

        forbidden_boxes = []

        for label in range(1, num_labels):
            # Get the mask of the current label
            mask = (labels_im == label)
            
            coords = np.column_stack(np.where(mask))

            top_left = coords.min(axis=0)
            bottom_right = coords.max(axis=0)
            
            top_right = [top_left[0], bottom_right[1]]
            bottom_left = [bottom_right[0], top_left[1]]
            
            forbidden_boxes.append({
                'top_left': tuple(top_left),
                'top_right': tuple(top_right),
                'bottom_left': tuple(bottom_left),
                'bottom_right': tuple(bottom_right)
            })

        print(f"Number of forbidden boxes: {len(forbidden_boxes)}")   
        
        for i, box in enumerate(forbidden_boxes):
            print(f"Box {i + 1} corners: {box}")

        return jsonify({"shape": grid.shape}), 200
    
    else:
        return jsonify({ "error": "Image not found" }), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
