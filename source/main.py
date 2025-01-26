import cv2
import av
import pandas as pd
from ObjectsDetection import Detector
from DetectionSettings import *
from Visualizing import Visualizer

class Processor:
    # Initialize property
    def __init__(self, info):
        self._model, self._ROIs, self._smallerROIs, self._unimportantObjects = info.getProperty()
        self._detector = Detector(self._model)
        self._visualizer = Visualizer()
        self._room = None

    # Visualize Image
    def _visualize(self, image, rois, frameDF):
        return self._visualizer.draw(image, rois, frameDF)

    # Process function
    def process(self, room, video):
        self._room = room
        container = av.open(video)
        for frameIndex, frame in enumerate(container.decode(video=0)):
            frame = frame.to_ndarray(format="bgr24")

            # Resize image - ROIs are defined on this size
            frame = cv2.resize(frame, IMAGE_SIZE)

            # Process
            print(f'Frame Number: {frameIndex}')
            frameDF = self._processFrame(frame)
            print(f"Completed {frameIndex}\n")

            combinedView = self._visualize(frame, self._ROIs[self._room], frameDF)

            cv2.imshow("Test Window",  combinedView)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Ensure Chair Number as Int
            frameDF[['Chair Number', 'Status']] = frameDF[['Chair Number', 'Status']].astype(int)
            frameDF = frameDF.sort_values(by=['Chair Number'], ascending=True)

            # Save Frame Dataframe into a csv file
            frameDF.to_csv(f'../csvFolder/status_{frameIndex}.csv', index=False)
        cv2.destroyAllWindows()
        return 0

    # Check in the smaller ROI (only called when no object is detected or only chair is detected in the big ROI)
    def _processSmallerRoi(self, idx, img):
        zone = self._smallerROIs[self._room][idx]
        df = self._detector.detectObjects(
            img[zone[3]:zone[1], zone[2]:zone[0], :],
            confThreshold=0.3,
            nmsThreshold=0.5)
        status = 'empty'
        for item in df['ClassIds'].unique():
            if item not in self._unimportantObjects:
                status = 'occupied'
                break

        return status


    # Detect objects in the image
    def _processFrame(self, frame):
        frameDF = pd.DataFrame(columns=['Chair Number',
                                        'Status'])
        frameDF = frameDF.astype({'Chair Number': 'int',
                                  'Status': 'int'})


        # Select correct ROIs of chairs
        roi = self._ROIs[f'{self._room}']
        for idx, chair in enumerate(roi):
            print(f"ROI number : {idx + 1} is being processed.", end='\t')

            # Initialize status to default values for each chair
            status = 'empty'

            # Proceed detecting on selected ROI
            df = self._detector.detectObjects(frame[chair[3]:chair[1],
                                              chair[2]:chair[0], :],
                                              confThreshold=0.7,
                                              nmsThreshold=0.5)

            # Check if result from detecting is empty
            if df.empty:
                status = self._processSmallerRoi(idx, frame)
            else:
                if 0 in df['ClassIds'].values:
                    status = 'occupied'
                else:
                    uniqueVals = df['ClassIds'].unique()
                    for item in uniqueVals:
                        if item not in self._unimportantObjects:# Ignore chairs, and some others ( all these items have been mapped to chairs, manually by us )
                            status = 'occupied'
                            break

            # Add new row to Frame csv
            print(f"Status : {status}")
            newRow = pd.DataFrame([{
                       'Chair Number': idx + 1,
                       'Status': STATUS_DICT[status]
                      }])
            frameDF = pd.concat([frameDF, newRow], ignore_index=True)  # Use pd.concat to append
        return frameDF

def main():
    info = InfoContainer()
    processor = Processor(info)
    processor.process(ROOM, VIDEO_PATH)

if __name__ == '__main__':
    main()
