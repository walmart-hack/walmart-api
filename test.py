# import cv2
# import numpy as np

# # Load the image
# image_path = './uploads/image.jpg'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Thresholding to differentiate between obstacles and open areas
# _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# # Converting the binary image to a grid (1 for obstacles, 0 for open areas)
# # Assuming obstacles are represented by white in the thresholded image
# grid = np.where(binary_image == 255, 0, 1)

# # Print the grid or process it further as needed
# print(grid)
# print(grid.shape[0], grid.shape[1])

# # Assuming `grid` is the array you created earlier
# # Create an empty array with the same dimensions as the grid and data type uint8
# height, width = grid.shape
# image = np.zeros((height, width), dtype=np.uint8)

# # Map the grid values to pixel values
# # Open areas (0 in grid) -> 255 (white)
# # Obstacles (1 in grid) -> 0 (black)
# image[grid == 0] = 255  # Open areas to white
# image[grid == 1] = 0    # Obstacles to black

# # Save or display the image
# cv2.imwrite('./uploads/converted_image.png', image)

# # To display the image using OpenCV:
# # cv2.imshow('Grid Image', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()



from pymongo import MongoClient
from gridfs import GridFS
import pickle

# Set up MongoDB client and GridFS
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['image_database']
fs = GridFS(db)

# Retrieve the metadata document
metadata = db['floor_maps'].find_one({"image_name": "651c38cd-6fff-43b6-b1ff-a92e76b14367_image.jpg"})  # Replace with your specific query

if metadata:
    file_id = metadata['grid_file_id']  # Get the file ID from the metadata

    # Retrieve the grid file from GridFS
    grid_file = fs.get(file_id)
    grid_bytes = grid_file.read()  # Read the file's content
    
    # Deserialize the grid back into a NumPy array
    grid = pickle.loads(grid_bytes)

    
    # Now `grid` is a NumPy array that you can use
    print("Grid dimensions:", grid.shape)
    print("Grid content:\n", grid)
else:
    print("No metadata found for the specified image.")
