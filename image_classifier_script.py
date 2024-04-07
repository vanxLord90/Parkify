
#import needed libs
import cv2
from skimage import io, color
import sklearn
import pandas
import skimage
import matplotlib
from skimage.transform import resize
import matplotlib.pyplot as plt
import datetime
import os
import joblib
import schedule
import time
import sys

from util import  get_parking_spots, is_car_there

global counter
counter =0

def run_script():
    global counter

    mask = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\mask_folder\\mask_1920_1080.png'
    fdp = "C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\im_by_2_mins"
    files = os.listdir(fdp)
    if files:
        file_name = files[0]  # Get the name of the first file
        parking_image = os.path.join(fdp, file_name)
        parking_image_test1 = cv2.imread(parking_image)
        parking_image_test2 = resize(parking_image_test1,(1080,1920))
        mask2 = cv2.imread(mask)
        mask2_gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask2_gray, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 4, cv2.CV_32S)
        cc = (num_labels, labels, stats, centroids)
        parking_spaces = get_parking_spots(cc)
        open_space = 0
        closed_space =0
        frame = parking_image_test2
        
        for space in parking_spaces:
            x1,y1,w,h = space
            
            space_img = frame[y1:y1+h, x1:x1+w,:]

            space_status = is_car_there(space_img)
            
            if space_status:
                open_space+=1
                cv2.rectangle(frame,(x1+15,y1),(x1+w+17,y1+h),(0,255,0),2)   
            else:
                closed_space+=1
                cv2.rectangle(frame,(x1+15,y1),(x1+w+17,y1+h),(0,0,255),2)

        total_spaces = open_space+closed_space
        window_str = f"Number of Spots Available are : {open_space} / {total_spaces}"
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %I:%M %p")
        window_str2 = f"Time: {formatted_time}"
        org = (50,50)
        font = cv2.FONT_HERSHEY_DUPLEX
        fontScale = 1
        color_of_text = (255, 0, 0) 
        thickness = 2
        cv2.putText(frame, window_str, org, font,  
                        fontScale, color_of_text, thickness, cv2.LINE_AA)
        text_size = cv2.getTextSize(window_str, font, fontScale, thickness)[0]
        org_time = (org[0], org[1] + text_size[1] + 10)  # Adding 10 pixels for spacing
        cv2.putText(frame, window_str2, org_time, font, fontScale, color_of_text, thickness, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        fdp2 = "C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\classified_images"
        file_name = f"classified_lot__map_{file_name}.png"
        fp = os.path.join(fdp2, file_name)
        cv2.imwrite(fp, frame)
        cv2.waitKey(0)    
        cv2.destroyAllWindows()
        os.remove(parking_image)
        
    else:
        print("Waiting for new files")
        # global counter
        counter +=1
        print(counter)
        if counter>5:
            sys.exit()
def main():
    # if(counter>5):
    #     sys.exit()
    
    schedule.every(.1).minutes.do(run_script)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__=="__main__":
    main()
        


