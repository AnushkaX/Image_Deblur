import cv2
import numpy as np

img = cv2.imread('assets/licenseplate_motion.jpg')                      #import image

if img is None:                                                         #check availability
    print('Failed to load image')
    sys.exit(1)


def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

img = np.float32(img)/255.0                                             #convert image
    cv2.imshow('input', img)

    img = blur_edge(img)                                                #blur image using the function
    IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)                    #fourier transformation