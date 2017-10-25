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

def fast_guided_filter(img, p, r = 40.0, s = 3.0, eps = 0.0001):
    D = img.shape[2]
    temp = np.empty(img.shape)
    for i in xrange(D):
        temp[:,:,i] = p
    I_n = cv2.resize(img, fx = 1.0/s, fy = 1.0/s, dsize=(0,0), interpolation = cv2.INTER_LINEAR)
    p_n = cv2.resize(temp, fx = 1.0/s, fy = 1.0/s, dsize = (0,0), interpolation = cv2.INTER_LINEAR)
    r_n = int(r / s)
    k_size = (r_n, r_n)

    mean_I = cv2.blur(I_n, ksize = k_size)
    mean_p = cv2.blur(p_n, ksize = k_size)
    corr_I = cv2.blur(np.multiply(I_n, I_n), ksize = k_size)
    corr_I_p = cv2.blur(np.multiply(I_n, p_n), ksize = k_size)

    var_I = corr_I - np.multiply(mean_I, mean_I)
    cov_I_p = corr_I_p - np.multiply(mean_I, mean_p)

    ep = np.full((1,1,3), eps)
    a = np.divide(cov_I_p, var_I + ep)
    b = mean_p - np.multiply(a, mean_I)

    mean_a = cv2.blur(a, ksize = k_size)
    mean_b = cv2.blur(b, ksize = k_size)

    mean_a = cv2.resize(mean_a, fx = s, fy = s, dsize = (0,0), interpolation = cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_b, fx = s, fy = s, dsize = (0,0), interpolation = cv2.INTER_LINEAR)

    q = np.multiply(mean_a, img) + mean_b
    return q

def depth_map(trans, beta):
    return -np.log(trans)/beta

def radiant_image(image, atmosphere, t, thres):
    R,C,D = image.shape
    temp = np.empty(image.shape)
    for i in xrange(D):
        temp[:,:,i] = t
    return (image - atmosphere)/np.maximum(temp, thres) + atmosphere


def automate(orig_img, window = 15, top_percent = 0.1, thres_haze = 0.1, omega = 0.95, beta = 1.0, radius = 40, scale = 3.0, eps = 0.0001):
    img = np.asarray(orig_img, dtype = np.float64)
    img_norm = (img - img.min())/(img.max() - img.min())
    kernel = np.ones((window, window), np.float64)
    dark = faster_dark_channel(img,kernel)

    A = atmosphere(img, dark, top_percent)

    B = img / A

    dark_div = faster_dark_channel(B, kernel)
    t_estimate = eval_transmission(dark_div, omega)

    R,C,_ = img.shape
    scale_param = scale
    for i in xrange(10):
        if(R % (i + scale) == 0 and C % (i + scale) == 0):
            scale_param = i + scale
    t_refined = fast_guided_filter(img_norm, t_estimate, radius, scale_param, eps)
    unhazed = radiant_image(img, A, t_refined[:,:,0], thres_haze)
    return [np.array(x, dtype = np.uint8) for x in [t_estimate * 255, t_refined * 255, unhazed] ]


img = cv2.imread('input.png')
#img = cv2.imread('foggy.jpg')
#img = cv2.imread('manish.jpg')
[trans, trans_refined, radiance] = automate(img)
cv2.imshow('original', img)
cv2.imshow('unhazed',radiance)
#cv2.imshow('transmission', trans)
cv2.imshow('refined transmission', trans_refined)
cv2.waitKey(0)
