import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def ReadImage(path):
    # Returns an image object from a path
    return cv.imread(path)


def Gray(path):
    # Returns an image in grayscale
    im = cv.imread(path)    
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY)


def ShowImage(path):
    # Displays image 
    im = ReadImage(path)
    plt.imshow(im)


def ImageDimensions(path):
    # Returns image dimensions
    im = ReadImage(path)
    return im.shape


def CropImage(image_path):
    # Crops image to size of color object inside image
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    
    result = cv.bitwise_and(image, image, mask=thresh)
    result[thresh==0] = [255,255,255] 
    (x, y, z_) = np.where(result > 0)
    mnx = (np.min(x))
    mxx = (np.max(x))
    mny = (np.min(y))
    mxy = (np.max(y))
    
    crop_img = image[mnx:mxx,mny:mxy,:]
    return crop_img


def CropAndResize(image_path,resize_x,resize_y):

    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV)[1]
    result = cv.bitwise_and(image, image, mask=thresh)
    result[thresh==0] = [255,255,255] 
    (x, y, z_) = np.where(result > 0)
    minx = (np.min(x))
    maxx = (np.max(x))
    miny = (np.min(y))
    maxy = (np.max(y))
    crop_img = image[minx:maxx,miny:maxy,:]
    
    vborder = 0
    hborder = 0
    if (resize_y/resize_x) >= (crop_img.shape[0]/crop_img.shape[1]):
        vborder= int((((resize_y/resize_x)*crop_img.shape[1])-crop_img.shape[0])/2)
    else:
        hborder = int((((resize_y/resize_x)*crop_img.shape[0])-crop_img.shape[1])/2)
    
    crop_img = cv.copyMakeBorder(crop_img, vborder, vborder,
                                 hborder, hborder, cv.BORDER_CONSTANT, 0)
    resized_im = cv.resize(crop_img, (resize_x, resize_y))
    
    return resized_im


def MixedCropper(image_path,new_x,new_y):
    # Fits image into a larger image of specified dimensions to have uniform sizes through out
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV)[1]
    result = cv.bitwise_and(image, image, mask=thresh)
    result[thresh==0] = [255,255,255] 
    (x, y, z_) = np.where(result > 0)
    minx = (np.min(x))
    maxx = (np.max(x))
    miny = (np.min(y))
    maxy = (np.max(y))
    crop_img = image[minx:maxx,miny:maxy,:]
    
    bg_color = [0,0,0]
    
    new_image = np.ones((new_y, new_x, 3), dtype=np.uint8) * bg_color
    x_offset = (new_x - crop_img.shape[1]) // 2
    y_offset = (new_y - crop_img.shape[0]) // 2
    
    new_image[y_offset:y_offset + crop_img.shape[0],
                  x_offset:x_offset + crop_img.shape[1]] = crop_img
    
    return new_image


def MaxDimensions(paths):
    # Returns list containing the hieght and width of 
    # the largest image in the dataframe paths
    heights = []
    widths = []
    for i in paths['Image Paths']:
        im = CropImage(i)

        x = im.shape[1]
        y = im.shape[0]
        heights.append(y)
        widths.append(x)

    h_w = [np.max(heights),np.max(widths)]
    return h_w


