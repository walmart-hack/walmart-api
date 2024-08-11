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



# from pymongo import MongoClient
# from gridfs import GridFS
# import pickle

# # Set up MongoDB client and GridFS
# mongo_client = MongoClient('mongodb://localhost:27017/')
# db = mongo_client['image_database']
# fs = GridFS(db)

# # Retrieve the metadata document
# metadata = db['floor_maps'].find_one({"image_name": "651c38cd-6fff-43b6-b1ff-a92e76b14367_image.jpg"})  # Replace with your specific query

# if metadata:
#     file_id = metadata['grid_file_id']  # Get the file ID from the metadata

#     # Retrieve the grid file from GridFS
#     grid_file = fs.get(file_id)
#     grid_bytes = grid_file.read()  # Read the file's content
    
#     # Deserialize the grid back into a NumPy array
#     grid = pickle.loads(grid_bytes)

    
#     # Now `grid` is a NumPy array that you can use
#     print("Grid dimensions:", grid.shape)
#     print("Grid content:\n", grid)
# else:
#     print("No metadata found for the specified image.")


# import numpy as np
# import cv2

# grid = np.array([
#     [0, 0, 1, 1, 0],
#     [0, 0, 1, 1, 0],
#     [0, 0, 0, 0, 0],
#     [1, 1, 0, 0, 0],
#     [1, 1, 0, 0, 0],
# ])

# num_labels, labels_im = cv2.connectedComponents(grid.astype(np.uint8))

# forbidden_boxes = []

# for label in range(1, num_labels):
#     # Get the mask of the current label
#     mask = (labels_im == label)
    
#     coords = np.column_stack(np.where(mask))

#     top_left = coords.min(axis=0)
#     bottom_right = coords.max(axis=0)
    
#     top_right = [top_left[0], bottom_right[1]]
#     bottom_left = [bottom_right[0], top_left[1]]
    
#     forbidden_boxes.append({
#         'top_left': tuple(top_left),
#         'top_right': tuple(top_right),
#         'bottom_left': tuple(bottom_left),
#         'bottom_right': tuple(bottom_right)
#     })

# print(f"Number of forbidden boxes: {len(forbidden_boxes)}")
# for i, box in enumerate(forbidden_boxes):
#     print(f"Box {i + 1} corners: {box}")



import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL for Walmart
base_url = "https://www.walmart.com/"

# Function to get the HTML content of a page
def get_page_content(url):
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        return response.content
    else:
        return None
    
def get_categories(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    print(soup)
    # Find categories and subcategories
    categories = {}
    # Example: Assuming category links are in <a> tags with a certain class
    for cat in soup.find_all('a', class_='category-link-class'):
        cat_name = cat.text.strip()
        cat_url = base_url + cat['href']
        categories[cat_name] = cat_url
    return categories
    
def main():
    # categories_url = base_url + "browse/categories"
    categories_url = base_url + "/all-departments"
    categories_html = get_page_content(categories_url)
    categories = get_categories(categories_html)
    
    all_data = []
    for category, cat_url in categories.items():
        subcategories = get_categories(get_page_content(cat_url))
        # for subcategory, subcat_url in subcategories.items():
        #     products = get_products(subcat_url)
        #     for product in products:
        #         product['category'] = category
        #         product['subcategory'] = subcategory
        #         all_data.append(product)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_data)
    df.to_csv('walmart_products.csv', index=False)

if __name__ == "__main__":
    main()