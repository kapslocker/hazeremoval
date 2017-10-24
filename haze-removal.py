import numpy as np
import cv2


def gen_dark_channel(img, window):
    '''get image size and num_channels'''
    R,C,D = img.shape
    '''pad image at the ends with a square of side window/2 to get minimum comparision at the ends'''
    pad_img = np.pad(img, ((window/2,window/2), (window/2, window/2), (0,0)), 'edge')
    print "Generating dark image..."
    #TODO: This is a slow operation to find minimum, devise a faster method that doesn't need iteration over all pixels explicitly
    sh = (R,C)
    channel_dark = np.zeros(sh)
    count = 0
    for r,c in np.ndindex(sh):
        channel_dark[r,c] = np.min(pad_img[r:r + window, c:c + window, :])
    return channel_dark


def atmosphere(img, channel_dark, thres_percent):
    R, C, D = img.shape
    # flatten dark to get top thres percentage of bright points. Paper uses thres_percent = 0.1
    flat_dark = channel_dark.ravel()
    req = int((R * C * thres_percent)/ 100)
    ''' find indices of top req intensites in dark channed'''
    indices = np.argpartition(flat_dark, -req)[-req:]

    '''flatten image and take max among these pixels '''
    flat_img = img.reshape(R * C,3)
    return np.max(flat_img.take(indices, axis = 0), axis = 0)

def eval_transmission(dark_div, param, min_thres):
    '''returns the estimated transmission'''
    transmission = 1 - param * dark_div
    return np.maximum(transmission, min_thres )

def depth_map(trans, beta):
    return -np.log(trans)/beta

def radiant_image(image, atmosphere, t):
    R,C,D = image.shape
    temp = np.empty(image.shape)
    for i in xrange(D):
        temp[:,:,i] = t
    return (image - atmosphere)/temp + atmosphere


window = 15
thres_percent = 0.2
omega = 0.95
beta = 1.0

orig_img = cv2.imread('manish.jpg')
img = np.asarray(orig_img, dtype = np.float64)
dark = gen_dark_channel(img, window)

A = atmosphere(img, dark, omega)

B = img / A

dark_div = gen_dark_channel(B, window)
t_estimate = eval_transmission(dark_div, omega, thres_percent)

unhazed = radiant_image(img, A, t_estimate)
# Outputs:
# dark_image = np.array(dark, dtype = np.uint8)
# cv2.imshow('dark_image', dark_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

temp = np.array(unhazed, dtype = np.uint8)
cv2.imshow('original', orig_img)
cv2.imwrite('unhazed.jpg',temp)
cv2.waitKey(0)
cv2.destroyAllWindows()
