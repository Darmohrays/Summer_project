import numpy as np
import cv2
from utils import gen_alphabet, get_words, get_chars, resize, normalize_images
from keras.models import load_model
from collections import Counter 


class HighlightWords:
    def __init__(self):
        self._model = load_model('course_project/Models/model_more_classes2.h5');
        self._orig_img = None
        self._text = ''
        self._bboxes = None
        self._most_frequent_word = None
        self._alphabet = gen_alphabet()
#         self._alphabet = alphabet2
#         print('Initialized')
        
    def _highlight_most_frequent(self):
        self._highlighted_img = self._orig_img.copy()
        words = self._text.split()
        counter = Counter(words)
        self._most_frequent_word = counter.most_common(1)[0]
        indxs = []
        for i, word in enumerate(words):
            if word == self._most_frequent_word:
                indxs.append(i)
                (x, y, w, h) = self._bboxes[i]['word_bbox']
                alpha = 3
                beta = 100
                self._highlighted_img[y:y+h, x:x+w] = np.clip(alpha*self._highlighted_img[y:y+h, x:x+w] + beta, 0, 255)
        
                
    def _get_bboxes(self):
        words = get_words(self._thresholded_img)
        chars = get_chars(self._thresholded_img, words)
        self._bboxes = chars

    def _get_prediction(self):
        for i, word in enumerate(self._bboxes.values()):
#             print('Word: {}'.format(i))
            for j, character_bbox in enumerate(word['cordinates']):
                img = resize(self._thresholded_img, character_bbox)
                img = normalize_images(img)

                img = img.reshape((1, 28, 28, 1))
                prediction = self._model.predict(img)
                decoded = self._alphabet[np.argmax(prediction)]
                
                self._text += str(decoded)[0]
                
            self._text += ' '
            
            
    def fit(self, img):
        self._orig_img = img
        self._text = ''
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)[1]
        self._thresholded_img = img
        self._get_bboxes()
        self._get_prediction()
        self._highlight_most_frequent()