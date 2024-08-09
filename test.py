import cv2
import numpy as np

# Load the image
image_path = './uploads/image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Thresholding to differentiate between obstacles and open areas
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Converting the binary image to a grid (1 for obstacles, 0 for open areas)
# Assuming obstacles are represented by white in the thresholded image
grid = np.where(binary_image == 255, 0, 1)

# Print the grid or process it further as needed
print(grid)
print(grid.shape[0], grid.shape[1])

# Assuming `grid` is the array you created earlier
# Create an empty array with the same dimensions as the grid and data type uint8
height, width = grid.shape
image = np.zeros((height, width), dtype=np.uint8)

# Map the grid values to pixel values
# Open areas (0 in grid) -> 255 (white)
# Obstacles (1 in grid) -> 0 (black)
image[grid == 0] = 255  # Open areas to white
image[grid == 1] = 0    # Obstacles to black

# Save or display the image
cv2.imwrite('./uploads/converted_image.png', image)

# To display the image using OpenCV:
# cv2.imshow('Grid Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
