import cv2
import numpy as np
import pygame

# Load the input video
cap = cv2.VideoCapture('input.mp4')

# Get the dimensions of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the Pygame display
display_size = (frame_width*1.5, frame_height*1.5)  # double the size of the input video
display = pygame.display.set_mode(display_size)

# Load the pawn images as Pygame surfaces
pawn_white = pygame.image.load('pawn_white.png').convert_alpha()
pawn_black = pygame.image.load('pawn_black.png').convert_alpha()

# Define the dimensions of the chessboard
rows = 60
cols = 60

# Create the chessboard array
chessboard = np.zeros((rows, cols))

# Initialize the background subtractor
bgsub = cv2.createBackgroundSubtractorMOG2()

# Define the size of the pawn images on the chessboard
pawn_size = (frame_width*1.5 // cols, frame_height*1.5 // rows)  # adjust the size of the pawns
cell_size = (frame_width*1.5 // cols, frame_height*1.5 // rows)

# Initialize the Pygame display
pygame.init()
pygame.mixer.init()
try:
    pygame.mixer.music.load('audio.mp3')
    pygame.mixer.music.play()
except pygame.error:
    print("Failed to load audio file")

# Loop over each frame of the video
while cap.isOpened():
    # Read the next frame of the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the colors to be segmented
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # Threshold the image to create binary masks for the white and black objects
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Apply background subtraction to separate the chessboard from the rest of the video
    fgmask = bgsub.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Find the contours of the white objects in the foreground mask
    white_contours, hierarchy = cv2.findContours(cv2.bitwise_and(fgmask, fgmask, mask=white_mask),
                                                 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each white object contour and extract the corresponding pawn
    for i, contour in enumerate(white_contours):
        # Use the centroid of the contour to determine the pawn's position
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Calculate the position of the pawn on the chessboard
        pawn_row = int((cy / frame_height) * rows)
        pawn_col = int((cx / frame_width) * cols)

        # Check if the pawn is within the bounds of the chessboard
        if pawn_row >= rows or pawn_col >= cols:
            continue

        # Update the chessboard with the position of the pawn
        if chessboard[pawn_row][pawn_col] == 0:
            chessboard[pawn_row][pawn_col] = 1  # mark the chessboard with a white pawn
        else:
            # If a pawn has already been placed on this cell, remove it
            chessboard[pawn_row][pawn_col] = 0

    # Find the contours of the black objects in the foreground mask
    black_contours, hierarchy = cv2.findContours(cv2.bitwise_and(fgmask, fgmask, mask=black_mask),
                                                 cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each black object contour and extract the corresponding pawn
    for i, contour in enumerate(black_contours):
        # Use the centroid of the contour to determine the pawn's position
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Calculate the position of the pawn on the chessboard
        pawn_row = int((cy / frame_height) * rows)
        pawn_col = int((cx / frame_width) * cols)

        # Check if the pawn is within the bounds of the chessboard
        if pawn_row >= rows or pawn_col >= cols:
            continue

        # Update the chessboard with the position of the pawn
        if chessboard[pawn_row][pawn_col] == 0:
            chessboard[pawn_row][pawn_col] = -1  # mark the chessboard with a black pawn
        else:
            # If a pawn has already been placed on this cell, remove it
            chessboard[pawn_row][pawn_col] = 0

    # Draw the chessboard on the Pygame display
    display.fill((255, 255, 255))
    for i in range(rows):
        for j in range(cols):
            if chessboard[i][j] == 1:
                display.blit(pawn_white, (j * pawn_size[0], i * pawn_size[1]))
            elif chessboard[i][j] == -1:
                display.blit(pawn_black, (j * pawn_size[0], i * pawn_size[1]))

            # Draw the grid lines
            pygame.draw.rect(display, (0, 0, 0),
                             pygame.Rect(j * pawn_size[0], i * pawn_size[1], cell_size[0], cell_size[1]), 1)

    # Update the Pygame display
    pygame.display.update()

    # Check for Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
cap.release()
cv2.destroyAllWindows()
