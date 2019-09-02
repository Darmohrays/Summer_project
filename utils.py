import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd


def get_words(img):
    """
    Function returns bboxes for each word and for chars in it.
    
    Parameters
    ----------
    img: np.array
    Thresholded grayscale image.
    
    Returns
    -------
    words: list
    Array of words bboxes in the form (x, y, w, h) where (x, y) - lower left point, h - height of bbox, w - width
    
    chars: list of lists
    Array which contains arrays that represent coordinates of each  character in word

    """
    
    # here we take maximum elements by word so we can detect rows with words
    # if there is a word the value would be 255, if not - 0
    maxs = np.max(img, axis=1) # maximum element by row
    indxs_y = np.where(maxs[:-1] != maxs[1:])[0] # indexes of elements where is transition from 255 to 0 or vice versa
    words = [] # list for storing words bboxes
    chars = [] # list for storing characters bboxes

    # In this loop we do the same operation but get max values by column so we can know
    # Where the character ends and starts
    for indx_y in range(0, len(indxs_y)-1, 2):

        distances = [] # list to store distances between 
        cordinates = []
        row = img[indxs_y[indx_y]:indxs_y[indx_y+1]] # take row of the text
        maxs1 = np.max(row, axis=0)
        indxs_x = np.where(maxs1[:-1] != maxs1[1:])[0]
        last_x = indxs_x[0]
        for indx_x in range(0, len(indxs_x)-1, 2):
            cordinates.append((
                indxs_x[indx_x],
                indxs_y[indx_y],
                indxs_x[indx_x+1] - indxs_x[indx_x],
                indxs_y[indx_y+1] - indxs_y[indx_y]
            ))
            distances.append(indxs_x[indx_x] - last_x)
            last_x = indxs_x[indx_x+1]

        temp_words, temp_chars = get_words_cordinates(cordinates, distances)
        words += temp_words
        chars += temp_chars
        
    return words, chars


def get_words_cordinates(cordinates, distances):
    """
    Parameters
    ----------
    cordinates: tuple or list which contains bboxes for each character in form - (x, y, w, h), where (x, y) - lower left point, h -       height of bbox, w - width
    
    """
    hist, bins = np.histogram(distances) # get histogram of distances between characters
        
    divider = find_divider(hist, bins) # get value by which we will decide if character belong to current word or starts next
    words = []
    chars = []
    x, y, w, h = cordinates[0]
    
    if w*h < 25: # weeding out anomalies
        return []
    
    if check_for_one_word_in_line(distances): # check is it only one word in row so we can treat this case different
        for i, (x_t, y_t, w_t, h_t) in enumerate(cordinates[1:]):
            y = min(y, y_t)
            h = max(h, h_t)
            w += w_t + distances[i]
        
        words.append((x, y, w+distances[i+1], h))
        chars = cordinates
    else:
        start = 0 # start index for current word
        end = 0 # end index for current word
        for i, item in enumerate(distances[1:]):
            if item < divider:
                x_t, y_t, w_t, h_t = cordinates[i+1]
                y = min(y, y_t)
                w += w_t + item
                h = max(h, h_t)
                end += 1
            else:
                words.append((x, y, w, h))
                x, y, w, h = cordinates[i+1]
                end += 1
                chars.append(cordinates[start:end])
                start = end

        chars.append(cordinates[end:])
        words.append((x, y, w, h))
        
    return words, chars

def find_divider(hist, bins):
    """
    Function finds distance value.
    
    If distance between separate characters more than this value, characters are in the same word and if distance lower -                 characters belong to different words
    """
    left = 0
    right = len(hist)-1
    
    while hist[left] == 0:
        left += 1
        
    while hist[right] == 0:
        right -= 1
    
    divider = bins[left]*0.5 + bins[right]*0.5
    
    return divider
        
def check_for_one_word_in_line(distances):
    """
    Function chechs whether it is only one word in line
    """
    mean = np.mean(distances)
    var = np.var(distances)
    
    if mean > var:
        return True

    return False        


def gen_alphabet():
    """
    Generates alphabet to decode predictions
    """
    alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
    symbols = ['.', '-', ',', '!', '?', ':', ';', '>', 
                   '<', '=', '@', '#', '$', '%', '^',
                   '&', '*', '(', ')',
                  '+', '\\', '/', '[', ']', '{', '}']
    numbers = [chr(i) for i in range(ord('0'), ord('9')+1)]
    for symbol in symbols:
        alphabet.append(symbol)
    for number in numbers:
        alphabet.append(number)
    return alphabet

def resize(img, cordinates):
    """
    Resizes images to (28, 28)
    """
    [x, y, w, h] = cordinates
    temp_img = img[y:y+h, x:x+w]
    
    max_ = max(w, h)
    temp_img = cv2.copyMakeBorder(temp_img, (max_-h)//2, (max_-h)//2, (max_-w)//2, (max_-w)//2,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    resized = cv2.resize(temp_img, (28, 28))
    
    return resized

def normalize_images(images):
    """
    Normalizes images
    """
    H, W = 28, 28
    images = np.reshape(images, (-1, H * W))
    numerator = images - np.expand_dims(np.mean(images, 1), 1)
    denominator = np.expand_dims(np.std(images, 1), 1)
    
    return np.reshape(numerator / (denominator + 1e-7), (-1, H, W))

# def normalize_images(images):
#     return np.array(images, np.float) / 255.0

def save_img(path, img, img_name):
    """
    Save the image by a given path
    """
    path_arr = path.split('/')
    cur_path = ''
    for p in path_arr:
        cur_path += p + '/'
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)
    plt.imsave(path + img_name, img, cmap='gray')