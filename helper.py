import cv2
import numpy as np

def identify_forbidden_boxes(grid):
    num_labels, labels_im = cv2.connectedComponents(grid.astype(np.uint8))

    forbidden_boxes = []

    for label in range(1, num_labels):
        # Get the mask of the current label
        mask = (labels_im == label)
        
        coords = np.column_stack(np.where(mask))

        top_left = tuple(map(int, coords.min(axis=0)))
        bottom_right = tuple(map(int, coords.max(axis=0)))
        
        top_right = [top_left[0], bottom_right[1]]
        bottom_left = [bottom_right[0], top_left[1]]
        
        forbidden_boxes.append({
            'top_left': tuple(top_left),
            'top_right': tuple(top_right),
            'bottom_left': tuple(bottom_left),
            'bottom_right': tuple(bottom_right)
        })

    print(f"Number of forbidden boxes: {len(forbidden_boxes)}")
    
    return forbidden_boxes


def is_point_in_box(point, box):
    x, y = point
    top_left = box['top_left']
    bottom_right = box['bottom_right']
    return top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]