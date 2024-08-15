from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import uuid
from dotenv import load_dotenv
from helper import *

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
    filepath = app.config['UPLOAD_FOLDER'] + unique_name
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
        print(points)

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


@app.route('/customer-item-list', methods=['POST'])
def customer_item_list():
    customer_item_list = request.json.get('customer_item_list')
    
    prediction = predict_util(customer_item_list)
    print(prediction)

    return jsonify({"status": "success", "output": prediction}), 200


@app.route('/list-categories', methods=['GET'])
def list_categories():
    categories = list_categories_util()
    return jsonify({"status": "success", "categories": categories}), 200



@app.route('/test', methods=['POST'])
def generate_path():
    data = request.json
    image_name = data.get('image_name')
    categories_list = data.get("categories")
    start_location = data.get('start_location')

    if not categories_list or not image_name:
        return jsonify({"error": "No categories provided"}), 400
    
    metadata = collection.find_one({"image_name": image_name})

    if metadata:
        file_id = metadata['grid_file_id']

        grid_file = fs.get(file_id)
        grid_bytes = grid_file.read()
        
        # load the grid of image
        grid = pickle.loads(grid_bytes)
        
        # filter forbidden boxes
        forbidden_boxes = metadata['forbidden_boxes']
        mapped_points = metadata['point_to_box_map']
        filtered_forbidden_boxes = filter_forbidden_boxes(forbidden_boxes, mapped_points, categories_list, start_location)

        waypoints = [find_closest_free_point(grid, box) for box in filtered_forbidden_boxes if find_closest_free_point(grid, box) is not None]

        print(len(waypoints), waypoints[0])

        optimal_order = tsp_nearest_neighbor(waypoints, waypoints[-1])

        full_path = []
        for i in range(1, len(optimal_order)):
            segment_path = a_star(grid, optimal_order[i-1], optimal_order[i])
            full_path.extend(segment_path)

        # Load the original image
        image = cv2.imread(os.path.join(UPLOAD_FOLDER, image_name))

        # Draw the path
        draw_path_on_image(image, full_path)

        # Save the result
        cv2.imwrite('path_on_image.png', image)

        return jsonify({"status": "success", "shape": grid.shape}), 200
       
    
    else:
        return jsonify({ "error": "Image not found" }), 404


# @app.route('/test2', methods=['POST'])
# def upload_image():
#     data = request.json
#     image_name = data.get('image_name')
#     categories_list = data.get('categories')
#     start_location, end_location = data.get('start_location'), data.get('end_location')

#     if not categories_list or not image_name:
#         return jsonify({"error": "No categories or image name provided"}), 400

#     # Path to the image on your system
#     image_path = os.path.join(UPLOAD_FOLDER, image_name)
#     print(image_path)
    
#     if not os.path.exists(image_path):
#         return jsonify({"error": "Image not found"}), 404

#     # You can send the image as part of the response using `send_file`
#     return send_file(image_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
