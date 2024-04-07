import cv2
from skimage import io, color
import sklearn
import pandas
import skimage
import matplotlib
from skimage.transform import resize
import matplotlib.pyplot as plt

from util import  get_parking_spots, is_car_there

# mask = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\MASKOFPKLOTACROSSDREESE3.png'
# mask = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\mask_folder\\mask_folder\\north-french-field-house-parking_mask.png'
mask = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\mask_folder\\north-french-field-house-parking_mask.png'
# parking_image_test = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\pikl5test2.png'
# parking_image_test = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\input\\nffh2.png'
parking_image_test = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\input\\nffh2.png'

parking_image_test1 = cv2.imread(parking_image_test)

# parking_image_test2 = resize(parking_image_test1,(1172,1028))
# mask2 = io.imread(mask,0)
# parking_image = os.path.join(fdp, file_name)
# parking_image_test1 = cv2.imread(parking_image_test1)
# parking_image_test2 = resize(parking_image_test1,(1172,1028))
# mask2 = cv2.imread(mask)
# mask2_gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
# _, binary_mask = cv2.threshold(mask2_gray, 127, 255, cv2.THRESH_BINARY)
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 4, cv2.CV_32S)

# parking_image_test2 = cv2.resize(parking_image_test1, (1555,1802))
parking_image_test2 = cv2.resize(parking_image_test1, (2867,1335))


# Load the mask image and convert it to grayscale
mask2 = cv2.imread(mask)
mask2_gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

# Threshold the mask image to obtain a binary mask
_, binary_mask = cv2.threshold(mask2_gray, 127, 255, cv2.THRESH_BINARY)

# Find connected components in the binary mask
(num_labels, labels, stats, centroids)= cv2.connectedComponentsWithStats(binary_mask, 4, cv2.CV_8U)
cc = (num_labels, labels, stats, centroids)
parking_spaces = get_parking_spots(cc)
# Optionally, you may filter out small components or perform additional processing here

# Now, let's get the parking spots
# parking_spaces = get_parking_spots((num_labels, labels, stats, centroids))


''' now we will use something called connected components,
connected components. It is a mathematical component that says
that if a pixel and another neighboring pixel have the same value
then they are connected. Here we will use this concept to outline
parking lot spaces'''

# cc = cv2.connectedComponentsWithStats(mask2, 4, cv2.CV_32S)

# parking_spaces = get_parking_spots(cc)

frame = parking_image_test2
    
# park_test = parking_spaces[:4]

for space in parking_spaces:
    # cv2.imshow('frame',frame)

    x1,y1,w,h = space
    # print(space)
    # print(x1,y1,w,h)
    
    space_img = frame[y1:y1+h, x1:x1+w,:]

    space_status = is_car_there(space_img)
    
    # print(space_status)
    if space_status:
        cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
    else:
        cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)
        

cv2.imshow('frame',frame)
cv2.imwrite('nfh.png', frame)


cv2.waitKey(0)
cv2.waitKey(0)
    

    # ret = False
cv2.destroyAllWindows()
    


