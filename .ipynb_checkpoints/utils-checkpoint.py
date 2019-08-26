import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_words(img):
    maxs = np.max(img, axis=1) # Take 
    indxs_y = np.where(maxs[:-1] != maxs[1:])[0]
    cordinates = [] # (x, y, w, h)
    for indx_y in range(0, len(indxs_y)-1, 2):
        row = img[indxs_y[indx_y]:indxs_y[indx_y+1]]
        kernel = np.ones((2, 9), np.uint8)
        dilated = cv2.dilate(row, kernel, iterations=1)
        maxs1 = np.max(dilated, axis=0)
        indxs_x = np.where(maxs1[:-1] != maxs1[1:])[0]
        for indx_x in range(0, len(indxs_x)-1, 2):
            cordinates.append((
                indxs_x[indx_x],
                indxs_y[indx_y],
                indxs_x[indx_x+1] - indxs_x[indx_x],
                indxs_y[indx_y+1] - indxs_y[indx_y]
            ))
            
    return cordinates


def get_chars(img, words):
    characters = {}
    for i, (x, y, w, h) in enumerate(words):
        cordinates = []
        word = img[y:y+h, x:x+w]
        maxs = np.max(word, axis=0)
        indxs_x = np.where(maxs[:-1] != maxs[1:])[0]
        for indx_x in range(0, len(indxs_x)-1, 2):
            cordinates.append((
                x+indxs_x[indx_x],
                y,
                indxs_x[indx_x+1] - indxs_x[indx_x],
                h
            ))
            
        characters[i] = {'cordinates': cordinates, 'word_bbox': (x, y, w, h)}
    return characters

def gen_alphabet():
    alphabet = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    symbols = ['point', 'dash', 'comma', '!', '?', 'colon', 'semicolon', '>', 
                   '<', 'equals', 'ampersant', 'hash', 'dollar', 'percent', '^',
                   'and', 'asterics', 'round_open', 'round_close',
                  'plus', 'backslash', 'slash', 'square_open', 'square_close', 'curly_open', 'curly_close']
    numbers = range(10)
    for symbol in symbols:
        alphabet.append(symbol)
    for number in numbers:
        alphabet.append(number)
    return alphabet

def resize(img, cordinates):
    [x, y, w, h] = cordinates
    temp_img = img[y:y+h, x:x+w]
    
    max_ = max(w, h)
    temp_img = cv2.copyMakeBorder(temp_img, (max_-h)//2, (max_-h)//2, (max_-w)//2, (max_-w)//2,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    resized = cv2.resize(temp_img, (28, 28), interpolation=cv2.INTER_CUBIC)
    return resized

def normalize_images(images):
    H, W = 28, 28
    images = np.reshape(images, (-1, H * W))
    numerator = images - np.expand_dims(np.mean(images, 1), 1)
    denominator = np.expand_dims(np.std(images, 1), 1)
    return np.reshape(numerator / (denominator + 1e-7), (-1, H, W))

def save_img(path, img, img_name):
    path_arr = path.split('/')
    cur_path = ''
    for p in path_arr:
        cur_path += p + '/'
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)
    plt.imsave(path + img_name, img, cmap='gray')


# def count_words():
#     res = {}
#     path = 'test_data/test_y/test_y_digit.txt'
#     file = open(path, 'r')
#     cur_x = 0
#     for line in file:
#         if len(line) == 2 or len(line) == 3:
#             cur_x += 1
#             res[cur_x] = {'words': 0, 'characters': 0}
#             continue
            
#         words = len(line.split(' '))
#         characters = len(line.replace(' ', ''))        

#         res[cur_x]['words'] += words
#         res[cur_x]['characters'] += characters
    
#     df = pd.DataFrame(res.values())
#     df.to_csv('words_count.csv', index=False)
    
# def check_words():
#     path_word = 'test/x_{}/words/'
#     df = pd.read_csv('words_count.csv')
#     mismatches = 0
    
#     for i in range(1, 41):
#         words = os.listdir(path_word.format(i))
#         if len(words) != df.loc[i-1]['words'] and check_words:
#             print('Mismatch in word in {} sample'.format(i))
#             print('Expected: {}'.format(df.loc[i-1]['words']))
#             print('Got: {}'.format(len(words)))
#             print('')
#             mismatches += 1
            
#     return mismatches

