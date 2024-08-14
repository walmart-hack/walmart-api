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

        print(forbidden_boxes)
        
        return jsonify({"status": "success", "shape": grid.shape}), 200
       
    
    else:
        return jsonify({ "error": "Image not found" }), 404


@app.route('/customer-item-list', methods=['POST'])
def customer_item_list():
    customer_item_list = request.json
    
    prediction = predict_util(customer_item_list)
    print(prediction)

    return jsonify({"status": "success", "output": prediction}), 200


@app.route('/list-categories', methods=['GET'])
def list_categories():
    categories = list_categories_util()
    return jsonify({"status": "success", "categories": categories}), 200


def find_closest_free_point(grid, box):
    h, w = grid.shape
    points_to_check = []

    # Check all border points around the box
    for i in range(box['top_left'][0], box['bottom_left'][0] + 1):
        if box['top_left'][1] - 1 >= 0:
            points_to_check.append((i, box['top_left'][1] - 1))
        if box['top_right'][1] + 1 < w:
            points_to_check.append((i, box['top_right'][1] + 1))

    for j in range(box['top_left'][1], box['top_right'][1] + 1):
        if box['top_left'][0] - 1 >= 0:
            points_to_check.append((box['top_left'][0] - 1, j))
        if box['bottom_left'][0] + 1 < h:
            points_to_check.append((box['bottom_left'][0] + 1, j))

    # Filter out the points that are within grid bounds and not forbidden
    free_points = [point for point in points_to_check if grid[point] == 0]

    if free_points:
        return free_points[0]  # Return the first free point (closest)
    return None


import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    h, w = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                if grid[neighbor] == 1:
                    continue

                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # No path found

import itertools

def tsp_nearest_neighbor(waypoints):
    visited = []
    unvisited = waypoints[:]
    current = unvisited.pop(0)
    visited.append(current)
    
    while unvisited:
        next_point = min(unvisited, key=lambda point: heuristic(current, point))
        unvisited.remove(next_point)
        visited.append(next_point)
        current = next_point
    
    return visited

def draw_path_on_image(image, path):
    for i in range(1, len(path)):
        cv2.line(image, (path[i-1][1], path[i-1][0]), (path[i][1], path[i][0]), (0, 255, 0), 2)

def filter_forbidden_boxes(forbidden_boxes, categories):
    return [box for box in forbidden_boxes if box['category'] in categories]


@app.route('/test', methods=['POST'])
def generate_path():
    data = request.json
    image_name = data.get('image_name')
    categories_list = data.get("categories") # list of categories
    if not categories_list or not image_name:
        return jsonify({"error": "No categories provided"}), 400
    
    metadata = collection.find_one({"image_name": image_name})

    if metadata:
        file_id = metadata['grid_file_id']

        grid_file = fs.get(file_id)
        grid_bytes = grid_file.read()
        
        grid = pickle.loads(grid_bytes)
        
        forbidden_boxes = identify_forbidden_boxes(grid)

        filtered_forbidden_boxes = filter_forbidden_boxes(forbidden_boxes, categories_list)

        waypoints = [find_closest_free_point(grid, box) for box in filtered_forbidden_boxes if find_closest_free_point(grid, box) is not None]


        optimal_order = tsp_nearest_neighbor(waypoints)

        full_path = []
        for i in range(1, len(optimal_order)):
            segment_path = a_star(grid, optimal_order[i-1], optimal_order[i])
            full_path.extend(segment_path)

        # Load the original image
        image = cv2.imread('your_image_path.png')

        # Draw the path
        draw_path_on_image(image, full_path)

        # Save the result
        cv2.imwrite('path_on_image.png', image)

        return jsonify({"status": "success", "shape": grid.shape}), 200
       
    
    else:
        return jsonify({ "error": "Image not found" }), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
