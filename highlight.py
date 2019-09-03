import numpy as np
import cv2
from utils import gen_alphabet, get_words, resize, normalize_images
from keras.models import load_model
from collections import Counter 
import pytesseract
from keras.backend import set_learning_phase
from textblob import TextBlob


class HighlightWords:
    def __init__(self, img, model=None):
        if model == None:
            self._model = load_model('course_project/Models/main_model.h5')
        else:
            self._model = model
        set_learning_phase(0)
        self._text = ''
        self._most_frequent_word = None
        self._alphabet = gen_alphabet()
        self._text_tesseract = ''
        
        self.__fit(img)
        
    def _highlight_most_frequent(self):
        self._highlighted_img = self._orig_img.copy()
        words = self._text.split()
        counter = Counter(words)
        if not counter.most_common(1):
            return
        self._most_frequent_word = counter.most_common(1)[0][0]
        for i, word in enumerate(words):
            if word == self._most_frequent_word:
                (x, y, w, h) = self._bboxes_words[i]
                temp_area = self._highlighted_img[y:y+h, x:x+w].copy()
                mask = self._thresholded_img[y:y+h, x:x+w].copy()
                temp_area[mask == 255] = [255, 0, 0]
                
                self._highlighted_img[y:y+h, x:x+w] = temp_area 
        
                
    def _get_bboxes(self):
        words, chars = get_words(self._thresholded_img)
        self._bboxes_chars = chars
        self._bboxes_words = words

    def _get_prediction(self):
        for i, word in enumerate(self._bboxes_chars):
            for j, character_bbox in enumerate(word):
                img = resize(self._thresholded_img, character_bbox)
                img = normalize_images(img)
                img = img.reshape((1, 28, 28, 1))
                
                prediction = self._model.predict(img)
                decoded = self._alphabet[np.argmax(prediction)]
                self._text += str(decoded)
                
            self._text += ' '
    
    def _get_tesseract_predictions(self):
        self._text_tesseract = pytesseract.image_to_string(self._thresholded_img, lang='eng')
            
    def _correct_text(self):
        blob = TextBlob(self._text)
        self._corrected_text = str(blob.correct())
    
    def __fit(self, img):
        self._orig_img = img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)[1]
        self._thresholded_img = img
        self._get_bboxes()
        self._get_prediction()
        self._correct_text()
        self._highlight_most_frequent()
        self._get_tesseract_predictions()