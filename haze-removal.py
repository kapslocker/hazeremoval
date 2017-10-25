import numpy as np
import cv2


def gen_dark_channel(img, window):
    '''get image size and num_channels'''
    R,C,D = img.shape
    '''pad image at the ends with a square of side window/2 to get minimum comparision at the ends'''
    pad_img = np.pad(img, ((window/2,window/2), (window/2, window/2), (0,0)), 'edge')
    print "Generating dark image..."
    sh = (R,C)
    channel_dark = np.zeros(sh)
    count = 0
    for r,c in np.ndindex(sh):
        channel_dark[r,c] = np.min(pad_img[r:r + window, c:c + window, :])
    return channel_dark

def faster_dark_channel(img, kernel):
    print "Evaluating dark channel"
    temp = np.amin(img, axis = 2)
    return cv2.erode(temp, kernel)

def atmosphere(img, channel_dark, top_percent):
    R, C, D = img.shape
    # flatten dark to get top thres percentage of bright points. Paper uses thres_percent = 0.1
    flat_dark = channel_dark.ravel()
    req = int((R * C * top_percent)/ 100)
    ''' find indices of top req intensites in dark channed'''
    indices = np.argpartition(flat_dark, -req)[-req:]

    '''flatten image and take max among these pixels '''
    flat_img = img.reshape(R * C,3)
    return np.max(flat_img.take(indices, axis = 0), axis = 0)

def eval_transmission(dark_div, param):
    '''returns the estimated transmission'''
    transmission = 1 - param * dark_div
    return transmission

def depth_map(trans, beta):
    return -np.log(trans)/beta

def radiant_image(image, atmosphere, t, thres):
    R,C,D = image.shape
    temp = np.empty(image.shape)
    for i in xrange(D):
        temp[:,:,i] = t
    return (image - atmosphere)/np.maximum(temp, thres) + atmosphere


def automate(orig_img, window = 15, top_percent = 0.1, thres_haze = 0.1, omega = 0.95, beta = 1.0):
    img = np.asarray(orig_img, dtype = np.float64)
    #dark = gen_dark_channel(img, window)
    kernel = np.ones((window, window), np.float64)
    dark = faster_dark_channel(img,kernel)

    A = atmosphere(img, dark, top_percent)

    B = img / A

    #dark_div = gen_dark_channel(B, window)
    dark_div = faster_dark_channel(B, kernel)
    t_estimate = eval_transmission(dark_div, omega)
    unhazed = radiant_image(img, A, t_estimate, thres_haze)
    return [np.array(x, dtype = np.uint8) for x in [t_estimate * 255, unhazed] ]


img = cv2.imread('input.png')
#img = cv2.imread('foggy.jpg')
#img = cv2.imread('manish.jpg')
[trans, radiance] = automate(img)
cv2.imshow('original', img)
cv2.imshow('unhazed',radiance)
cv2.imshow('transmission', trans)
cv2.waitKey(0)
