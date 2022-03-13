import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ex1_utils import rotate_image,  show_flow 
from of_methods import lucas_kanade,  horn_schunck

def read_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

def plot_optical_flow(im1_path, im2_path, read_images=True):
    if read_images:
        im1 = read_image(im1_path)
        im2 = read_image(im2_path)
    else:
        im1 = im1_path#
        im2 = im2_path
    
    U_lk, V_lk = lucas_kanade(im1, im2, 5, 0.45, 0.0)
    U_hs, V_hs = horn_schunck(im1, im2, 1000,  0.5)
    #U_hs, V_hs = lucas_kanade(im1, im2, 5,  3.0)
    
    fig, ((ax_11, ax_12), (ax_21, ax_22), (ax_31, ax_32)) = plt.subplots(3, 2)
    ax_11.imshow(im1)
    ax_12.imshow(im2)
    
    ax_22.set_title("LucasKanadeOpticalFlow")
    #ax_21.set_title("LK - without value correction")
    #ax_22.set_title("LK - with value correction")
    
    show_flow(U_lk, V_lk, ax_21, type = 'angle', set_aspect = True)
    show_flow(U_lk, V_lk, ax_22, type = 'field', set_aspect = True)
    
    #ax_31.set_title("HornSchunckOpticalFlow")
    ax_32.set_title("HornSchunckOpticalFlow")
    
    show_flow(U_hs, V_hs, ax_31, type = 'angle')
    show_flow(U_hs, V_hs, ax_32, type = 'field', set_aspect = True)
    
    fig.tight_layout()
    plt.show()
    
def optical_flow_time(im1_path, im2_path, speedUp = False):
    im1 = read_image(im1_path)
    im2 = read_image(im2_path)
    
    startTime = time.time()
    lucas_kanade(im1, im2, 3)
    lucasCanadeTime = time.time() - startTime
    
    startTime = time.time()
    if speedUp:
        horn_schunck(im1, im2, 1000, 0.5, 3)
    else:    
        horn_schunck(im1, im2, 100, 0.5)
        
    hornSchunckTime = time.time() - startTime
    
    return (lucasCanadeTime, hornSchunckTime)
    
im1 = np.random.rand(200, 200).astype(np.float32)
im2 = im1.copy()
im2 = rotate_image(im2, -1)

#plot_optical_flow(im1, im2, False)

plot_optical_flow('disparity/cporta_left.png', 'disparity/cporta_right.png')

plot_optical_flow('minePictures/waffle1.jpg', 'minePictures/waffle2.jpg')

plot_optical_flow('disparity/office2_left.png', 'disparity/office2_right.png')

plot_optical_flow('collision/00000115.jpg', 'collision/00000116.jpg')

(lucasCanadeTime, hornSchunckTime) = optical_flow_time(
    'disparity/office_left.png', 'disparity/office_right.png')

(lucasCanadeTimeS, hornSchunckTimeS) = optical_flow_time(
    'disparity/office_left.png', 'disparity/office_right.png', True)

print(f'''lucas canade time: {lucasCanadeTime}
          horn schunck time: {hornSchunckTime}
          lucas canade speed up time: {lucasCanadeTimeS}
          horn schunck speed up time: {hornSchunckTimeS}''')
