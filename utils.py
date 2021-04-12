import cv2
import numpy as np
import imutils	
import math

def align_box(image, bboxes, skew_threshold=5, top_box=3):

    """
    box - [[l, t], [r, t], [r, b], [l, b]]
    """
    bboxes = bboxes[::-1] #Only use the box from the bottom
    vertical_vector = [0, -1] #Horizontial vector 

    top_box = np.argpartition([box[1][0]- box[0][0] for box in bboxes], -top_box)[-top_box:]
    avg_angle = 0
    for idx in top_box:
        skew_vector = bboxes[idx][0] - bboxes[idx][3]
        angle = np.math.atan2(np.linalg.det([vertical_vector,skew_vector]),np.dot(vertical_vector,skew_vector))
        avg_angle += math.degrees(angle)/3

    if abs(avg_angle) < skew_threshold:
        return image,0
    return imutils.rotate(image, avg_angle),1
    

def rotate_box(img, bboxes, degree, rotate_90, flip):
    h,w = img.shape[:2]
    if degree:
        new_bboxes = [[[h - i[1], i[0]] for i in bbox] for bbox in bboxes]
        new_img = cv2.rotate(img, degree)
        return new_img, np.array(new_bboxes)
    if rotate_90:
        new_bboxes = [[[h - i[1], i[0]] for i in bbox] for bbox in bboxes]
        new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return new_img, np.array(new_bboxes)

    if flip:
        new_bboxes = [[[w-i[0], h-i[1]] for i in bbox] for bbox in bboxes]
        new_img = cv2.rotate(img, cv2.ROTATE_180) 
        return new_img, np.array(new_bboxes)
    return img, bboxes

def get_idx(out, score, label):
    rs_idx = None
    m = max(score, key=lambda x: x[0])[0]
    for idx in range(len(out)):
        if  out[idx] == label and score[idx][0] >= m:
            rs_idx = idx
    return rs_idx


#Get images module
def get_images(image_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(image_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files





def calc_sim(first_box, second_box, isFirstBox, y_dist_limit, x_dist_limit):
    
    first_xmin, first_xmax, first_ymin, first_ymax = first_box
    second_xmin, second_xmax, second_ymin, second_ymax = second_box

    x_dist = min(abs(first_xmin-second_xmin), abs(first_xmin-second_xmax), abs(first_xmax-second_xmin), abs(first_xmax-second_xmax))
    y_dist = min(abs(first_ymin-second_ymin), abs(first_ymin-second_ymax), abs(first_ymax-second_ymin), abs(first_ymax-second_ymax))

    flag = False 

    if y_dist < y_dist_limit:
      if (first_ymax  < second_ymin) and (first_xmin < second_xmax)  and (first_xmax > second_xmin):
        return True 
      if x_dist < x_dist_limit:
        if not isFirstBox:
          return True 

      
    return flag 


def bb_intersection_over_union(boxA, boxB):
  ixmin = max(boxA[0], boxB[0])
  ixmax = min(boxA[1], boxB[1])
  iymin = max(boxA[2], boxB[2])
  iymax = min(boxA[3], boxB[3])

  iw = np.maximum(ixmax-ixmin+1., 0.)
  ih = np.maximum(iymax-iymin+1., 0.)

  inters = iw*ih

  uni = ((boxA[1]-boxA[0]+1.) * (boxA[3]-boxA[2]+1.) +
           (boxB[1] - boxB[0] + 1.) * (boxB[3] - boxB[2] + 1.) -
           inters)

  iou = inters / uni

  return iou


def merge_boxes(box1, box2):   #-> xmin, xmax, ymin, ymax 
    return [min(box1[0], box2[0]), 
         max(box1[1], box2[1]),    
         min(box1[2], box2[2]),    
         max(box1[3], box2[3])]    


def merge_box_by_distance(texts, boxes, y_dist_limit=10, x_dist_limit=30):
  for i, (text_1, text_box_1) in enumerate(zip(texts, boxes)):
    for j, (text_2, text_box_2) in enumerate(zip(texts, boxes)):
      #if j is smaller than i, skip 
      if j <= i:
        continue 
      #Create a new box if a distance between two boxes is less than a limit
      if calc_sim(text_box_1, text_box_2, i==0, y_dist_limit, x_dist_limit):

        #merge two boxes
        new_box = merge_boxes(text_box_1, text_box_2)
        #merge two strings
        new_text = text_1 + ' ' + text_2

        #store new string and delete the second one
        texts[i] = new_text
        del texts[j]

        #store new box and delete previous text box
        boxes[i] = new_box
        del boxes[j]

        return True, texts, boxes
    
  return False, texts, boxes 


def merge_box_by_iou(texts, boxes, iou_limit=0.0001):
  for i, (text_1, text_box_1) in enumerate(zip(texts, boxes)):
    for j, (text_2, text_box_2) in enumerate(zip(texts, boxes)):
      #if j is smaller than i, skip 
      if j ==i:
        continue 
      #Create a new box if a iou between two boxes is less than a limit
      if bb_intersection_over_union(text_box_1, text_box_2) > iou_limit:
        #merge two boxes
        new_box = merge_boxes(text_box_1, text_box_2)
        #merge two strings
        new_text = text_1 + ' ' + text_2

        #store new string and delete the second one
        texts[i] = new_text
        del texts[j]

        #store new box and delete previous text box
        boxes[i] = new_box
        del boxes[j]
        
        return True, texts, boxes
    
  return False, texts, boxes     