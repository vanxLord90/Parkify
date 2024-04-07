import joblib
from skimage.transform import resize
import numpy as np
import cv2

EMPTY=True
NOT_EMPTY = False

MODEL = joblib.load('C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\models\\parking_model5.pkl')

def is_car_there(spot_bgr):
    flat_data =[]
    try:
        img_resized = resize(spot_bgr,(20,20,3))
        
        # cv2.imshow('window2',img_resized)
        # cv2.resizeWindow('window2', 500,500)

        # if cv2.waitKey(10000) & 0xFF == ord('q'):
        #     return    
        flat_data.append(img_resized.flatten())
        
        flat_data = np.array(flat_data)
        # print(flat_data)
        y_pred = MODEL.predict(flat_data)
    
        # print(y_pred)
        # print(y_pred)
        if y_pred == 0:
            return EMPTY
        else:
            return NOT_EMPTY
    
    except Exception as e:
        print("Error occurred during resizing:", e)
        return NOT_EMPTY
    

def get_parking_spots(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots
