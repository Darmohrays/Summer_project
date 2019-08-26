import numpy as np
import cv2
from utils import gen_alphabet, get_words, get_chars, resize, normalize_images
from keras.models import load_model

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
        
    def _find_most_frequent(self):
        pass
    
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
                
                self._text += str(decoded)
                
            self._text += ' '
            
            
    def fit(self, img):
        self._orig_img = img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)[1]
        self._thresholded_img = img
        self._get_bboxes()
        self._get_prediction()