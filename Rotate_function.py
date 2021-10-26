import sys
import os
import re
import cv2
import numpy as np
import math

script_path = os.path.dirname(__file__)
os.chdir(script_path)

def rotate(origin, point, radian):
    ox, oy = origin 
    px, py = point
    qx = ox + math.cos(radian) * (px - ox) - math.sin(radian) * (py - oy)
    qy = oy + math.sin(radian) * (px - ox) + math.cos(radian) * (py - oy)
    
    return round(qx), round(qy)

def rotate_box_dot(x_cen, y_cen, width, height, theta):
    x_min = x_cen-width/2
    y_min = y_cen-height/2
    rotated_x1,rotated_y1=rotate((x_cen,y_cen),(x_min,y_min),theta)
    rotated_x2,rotated_y2=rotate((x_cen,y_cen),(x_min,y_min+height),theta)
    rotated_x3,rotated_y3=rotate((x_cen,y_cen),(x_min+width,y_min+height),theta)
    rotated_x4,rotated_y4=rotate((x_cen,y_cen),(x_min+width,y_min),theta)

    return np.reshape([rotated_x1,rotated_y1, rotated_x2,rotated_y2, rotated_x3,rotated_y3, rotated_x4,rotated_y4], (4,2))