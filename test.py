import cv2
from skimage import io, color
import sklearn
import pandas
import skimage
import matplotlib
from skimage.transform import resize
import matplotlib.pyplot as plt

parking_image_test = 'C:\\Users\\dasak\\HackAi\\ParkingLotAvailability\\parkifymodel\\\\ACROSSDREESE3\\crop2.png'
parking_image_test2 = cv2.imread(parking_image_test)
parking_image_test3 = resize(parking_image_test2,(20,20))

cv2.imshow('frame', parking_image_test3)
cv2.waitKey(0)
cv2.destroyAllWindows()

