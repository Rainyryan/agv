import cv2
import numpy as np

# Define chessboard parameters
chessboard_size = (12, 6)  # Number of internal corners (width, height)
square_size = 80  # Size of each square in mm (adjust as per your requirement)

# Create a blank white image
image_size = (720, 480)  # Size of the generated image (width, height)
board_image = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255

# Draw the chessboard pattern
for i in range(chessboard_size[1]):
    for j in range(chessboard_size[0]):
        if (i + j) % 2 == 0:
            top_left = (j * square_size, i * square_size)
            bottom_right = ((j + 1) * square_size, (i + 1) * square_size)
            cv2.rectangle(board_image, top_left, bottom_right, (0, 0, 0), -1)

# Save and display the chessboard image
cv2.imwrite("chessboard_pattern.png", board_image)
cv2.imshow("Chessboard Pattern", board_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
