import cv2
import numpy as np
import heapq
import math

def identify_forbidden_boxes(grid):
    '''
    Identify forbidden boxes in the grid
    '''
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
    '''
    Check if a point is inside a box
    '''
    y, x = point
    top_left = box['top_left']
    bottom_right = box['bottom_right']
    return top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]


def hsv2rgb(h, s, v):
    '''
    Convert HSV to RGB
    '''
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


### Functions to find optimal path
def find_closest_free_point(grid, box):
    '''
    Find the closest free point to the edge centers of the box
    '''
    h, w = grid.shape
    points_to_check = []

    mid_top = (box['top_left'][0], (box['top_left'][1] + box['top_right'][1]) // 2)
    mid_bottom = (box['bottom_left'][0], (box['bottom_left'][1] + box['bottom_right'][1]) // 2)
    mid_left = ((box['top_left'][0] + box['bottom_left'][0]) // 2, box['top_left'][1])
    mid_right = ((box['top_right'][0] + box['bottom_right'][0]) // 2, box['top_right'][1])

    if mid_top[0] - 1 >= 0:
        points_to_check.append((mid_top[0] - 1, mid_top[1]))
    if mid_bottom[0] + 1 < h:
        points_to_check.append((mid_bottom[0] + 1, mid_bottom[1]))
    if mid_left[1] - 1 >= 0:
        points_to_check.append((mid_left[0], mid_left[1] - 1))
    if mid_right[1] + 1 < w:
        points_to_check.append((mid_right[0], mid_right[1] + 1))

    free_points = [point for point in points_to_check if grid[point] == 0]

    if free_points:
        return free_points[0]
    return None


def heuristic(a, b):
    '''
    Heuristic function to estimate the distance between two points
    '''
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    '''
    A* algorithm to find the optimal path between two points on the grid
    '''
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
    
    return []


def tsp_nearest_neighbor(waypoints, start_point=None):
    '''
    Given a list of waypoints, find the optimal order to visit them using the nearest neighbor algorithm.
    '''
    visited = []
    
    print(waypoints, start_point)

    if start_point is not None:
        # Find and remove the start point from unvisited waypoints
        if start_point in waypoints:
            current = start_point
            unvisited = waypoints[:]
            unvisited.remove(start_point)
        else:
            raise ValueError("The start point is not in the list of waypoints.")
    else:
        # If no start point is provided, start from the first point in waypoints
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
    '''
    Draw the path on the image
    '''
    color = hsv2rgb(0,57,85)
    for i in range(1, len(path)):
        cv2.line(image, (path[i-1][1], path[i-1][0]), (path[i][1], path[i][0]), color, 5)



def filter_forbidden_boxes(forbidden_boxes, mapped_points, categories, start_location):
    '''
    Filter forbidden boxes based on categories and start location
    '''
    result = []

    for i in categories:
        if mapped_points[i] is not None:
            result.append(forbidden_boxes[mapped_points[i]])

    if start_location is not None:
        result.append(forbidden_boxes[mapped_points[start_location]])

    return result