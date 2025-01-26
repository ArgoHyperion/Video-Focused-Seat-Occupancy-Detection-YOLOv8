import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    # Visualizing image with boxes of chairs
    def drawBoxes(self, img, rois, dataframe):
        for _, row in dataframe.iterrows():
            for idx, roi in enumerate(rois):
                if str(idx+1) == str(row['Chair Number']):
                    topLeft = (roi[2], roi[3])
                    bottomRight = (roi[0], roi[1])
                    color = (0, 255, 0) if row['Status'] == 0 else (0, 0, 255)
                    label = "empty" if row['Status'] == 0 else "occupied"

                    # Add rectangle
                    cv2.rectangle(img, topLeft, bottomRight, color, 2)

                    # Add the label
                    text_position = (roi[2] + 2, roi[3] - 5)
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    fontScale = 0.6
                    thickness = 1
                    textColor = color
                    cv2.putText(img, label, text_position, font, fontScale, textColor, thickness)

                    break
        return img

    # Visualizing Seat Map
    def drawMap(self, dataframe):
        mapSize = 200
        chairMap = np.zeros((mapSize, mapSize, 3), dtype=np.uint8) + 255  # White background
        radius = 30
        centers = [(50, 50), (50, 150), (150, 50), (150, 150)]  # Fixed positions for 4 chairs

        for idx, (x, y) in enumerate(centers):
            # Get chair status, default to 0 (empty) if not found
            status = dataframe[dataframe['Chair Number'] == idx + 1]['Status'].values
            if len(status) > 0:
                status = status[0]
            else:
                status = 0

            # Set color based on status
            color = (0, 255, 0) if status == 0 else (0, 0, 255)  # Green for empty, red for occupied
            cv2.circle(chairMap, (x, y), radius, color, -1)  # Draw filled circle
            label = f"Chair {idx + 1}"
            textPosition = (x - 20, y + radius + 15)
            cv2.putText(chairMap, label, textPosition, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        return chairMap

    # Combine the Seat Map and Processed Image
    def combine(self, annotatedFrame, seatMap):
        map_resized = cv2.resize(seatMap, (300, 300))
        combinedView = np.zeros((max(annotatedFrame.shape[0], map_resized.shape[0]),
                                  annotatedFrame.shape[1] + map_resized.shape[1], 3), dtype=np.uint8)

        combinedView[:annotatedFrame.shape[0], :annotatedFrame.shape[1]] = annotatedFrame
        combinedView[:map_resized.shape[0], annotatedFrame.shape[1]:] = map_resized

        return combinedView
    # Draw
    def draw(self, img, rois, dataframe):
        annotatedFrame = self.drawBoxes(img, rois, dataframe)
        seatMap = self.drawMap(dataframe)
        combinedView = self.combine(annotatedFrame, seatMap)
        return combinedView
