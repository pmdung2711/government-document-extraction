import re
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import argparse
from utils import rotate_box, get_idx, align_box, merge_box_by_distance, merge_box_by_iou
from modules.text_detection.predict import test_net,net,refine_net,poly
from PIL import Image
import torch
import imutils	
import math
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import joblib 
import sys

#Loading Text Recognition module

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



'''
Get one image and extract all paragraph from the document
---> Use CRAFT to extract each line
---> Merge all boxes if they are closed or overlap other boxes.
---> Join the texts corresponding to merged boxes
---> 
'''
def export_text(image, text_recognizer):

    #Make a copy of orginal image
    image_copy = image.copy()

    #Find all bounding boxes
    bboxes, polys, score_text = test_net(net, image_copy, 0.7, 0.6, 0.5, True, poly, refine_net)
    
    #if bounding boxes exist:
    if bboxes != []:

        #Store bouding box in xmin, xmax, ymin, ymax format
        bboxes_xxyy = []

        #Store all texts and probabilities 
        texts = []
        probs = []

        #Align the image
        image_copy, check = align_box(image_copy, bboxes, skew_threshold=0.5, top_box=3)

        #Get all the bounding boxes again
        bboxes, polys, score_text = test_net(net, image_copy, 0.7, 0.2, 0.5, True, poly, refine_net)

        #Get orginal shape of image
        h,w,c = image_copy.shape

        '''
        For every bounding, we convert the 8 clock wise coordinates
        to xmin, xmax, ymin and ymax. The new coordinates then will be
        stored in bboxes_xxyy. 
        For text recognition, use the text_recognizer to predict the text inside 
        the bounding box. 
        '''
        for i, box in enumerate(bboxes):
            x_min = max(int(min(box, key=lambda x: x[0])[0]), 1)
            x_max = min(int(max(box, key=lambda x: x[0])[0]), w-1)
            y_min = max(int(min(box, key=lambda x: x[1])[1]), 3)
            y_max = min(int(max(box, key=lambda x: x[1])[1]),h-2)
            bboxes_xxyy.append([x_min-1, x_max,y_min-1, y_max])
            cropped_image = image_copy[y_min:y_max, x_min:x_max,:]
            img = Image.fromarray(cropped_image)
            text = text_recognizer.predict(img, return_prob = False)
            texts.append(text)

        '''
        Merge all boxes together by iou and distance
        Default:
        y_dist_limit: 10 (Maximum distance by y coordinate to merge two boxes)
        x_dist_limit: 40 (Maximum distance by x coordinate to merge two boxes)
        iou_limit = 0.001
        
        '''

        need_merging = True 
        while need_merging:
            need_merging, texts, bboxes_xxyy = merge_box_by_iou(texts, bboxes_xxyy)

        need_merging = True 
        while need_merging:
            need_merging, texts, bboxes_xxyy = merge_box_by_distance(texts, bboxes_xxyy)

        return texts

if __name__ == "__main__":
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'weights/transformerocr.pth'
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False
    text_recognizer = Predictor(config)
    test_image_path="test_data/Công văn 641_UBND-NC PDF.pdf.jpg"
    image = cv2.imread(test_image_path)
    detected_texts = export_text(image, text_recognizer)
    print(detected_texts)




