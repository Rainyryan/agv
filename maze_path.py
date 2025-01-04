import cv2
import numpy as np
from heapq import heappop, heappush
from skimage.morphology import skeletonize




def find_dot(image, lower_bound, upper_bound):
    """Find the centroid of the dot in the specified color range."""
    mask = cv2.inRange(image, lower_bound, upper_bound)
    # cv2.imshow("", mask)
    # cv2.waitKey(0)
    moments = cv2.moments(mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cy, cx  # Return in (row, col) format
    return None

def astar(maze, start, end):
    """A* pathfinding algorithm."""
    rows, cols = maze.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        _, current = heappop(open_set)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring cells
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor] == 1:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Load the maze image
maze_image = cv2.imread('maze2.jpg')
# print(maze_image.shape)
cv2.imshow("hi", maze_image)
cv2.waitKey(0)

# Find the start (red) and finish (green) dots
start = find_dot(maze_image, lower_bound=(0, 128, 0), upper_bound=(80, 255, 80))  # Green range
finish = find_dot(maze_image, lower_bound=(0, 0, 128), upper_bound=(80, 80, 255))  # Red range

if start is None or finish is None:
    print("Start or finish dot not found.")
    exit()
# print(start, finish)

# Convert the maze to grayscale and binary
gray_maze = cv2.cvtColor(maze_image, cv2.COLOR_BGR2GRAY)
_, binary_maze = cv2.threshold(gray_maze, 60, 255, cv2.THRESH_BINARY)

cv2.imshow("",binary_maze)
cv2.waitKey(0)

binary_maze = (binary_maze // 255).astype(np.uint8)  # Convert to 0-1 format

# cv2.imshow("",gray_maze)
# cv2.waitKey(0)



# Skeletonize the binary maze
skeleton = skeletonize(binary_maze).astype(np.uint8)
cv2.imshow("",skeleton*255)
cv2.waitKey(0)


# Adjust start and finish points to snap to the skeleton
def snap_to_skeleton(point, skeleton):
    y, x = point
    h, w = skeleton.shape
    max_radius = 50
    for radius in range(1, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Check if the current point lies on the circle's perimeter
                if dx**2 + dy**2 <= radius**2 and dx**2 + dy**2 > (radius - 1)**2:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:  # Ensure within bounds
                        if skeleton[ny, nx] == 1:  # Check if it's a skeleton point
                            return (ny, nx)


def thicken_skeleton(skeleton, iterations=1, kernel_size=(3, 3)):
    # Create a structuring element (kernel) for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Apply dilation iteratively
    thickened_skeleton = cv2.dilate(skeleton, kernel, iterations=iterations)
    return thickened_skeleton


skeleton = thicken_skeleton(skeleton)

start = snap_to_skeleton(start, skeleton)
finish = snap_to_skeleton(finish, skeleton)

# Run the A* algorithm
path = astar(skeleton, start, finish)

# Visualize the result
if path:
    for point in path:
        maze_image[point[0], point[1]] = [255, 0, 0]  # Draw path in gray
    cv2.imshow("Path", maze_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No path found.")
