from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import uuid
from dotenv import load_dotenv
from helper import identify_forbidden_boxes, is_point_in_box

from mongo_client import *
from categorization import *

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
    '''
        Example post request json: 
        `{
            "image_name": "example.png",
            "points": [
                {"name": "point1", "coordinates": [3, 0]},
                {"name": "point2", "coordinates": [2, 2]}
            ]
        }`
    '''
    data = request.json
    image_name = data.get('image_name')
    points = data.get("points")
    if not points:
        return jsonify({"error": "No points provided"}), 400
    
    metadata = collection.find_one({"image_name": image_name})

    if metadata:
        file_id = metadata['grid_file_id']

        grid_file = fs.get(file_id)
        grid_bytes = grid_file.read()
        
        grid = pickle.loads(grid_bytes)
        
        forbidden_boxes = identify_forbidden_boxes(grid)

        point_to_box_map = {}
        
        for point in points:
            name = point['name']
            coordinates = point['coordinates']
            for i, box in enumerate(forbidden_boxes):
                if is_point_in_box(coordinates, box):
                    point_to_box_map[name] = i
                    break
            else:
                point_to_box_map[name] = None

        result = collection.update_one({"image_name": image_name}, {"$set": {"point_to_box_map": point_to_box_map, "forbidden_boxes": forbidden_boxes}})

        if result.matched_count == 0:
            return {"error": "Failed to update the document"}, 500
        
        return jsonify({"status": "success", "shape": grid.shape}), 200
       
    
    else:
        return jsonify({ "error": "Image not found" }), 404


@app.route('/customer-item-list', methods=['GET'])
def customer_item_list():
    customer_item_list = request.json.get('customer_item_list')
    
    prediction = predict_util(customer_item_list)

    return jsonify({"status": "success", "output": prediction}), 200


@app.route('/list-categories', methods=['GET'])
def list_categories():
    categories = list_categories_util()
    return jsonify({"status": "success", "categories": categories}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
