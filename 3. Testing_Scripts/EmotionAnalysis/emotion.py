from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array
import config as cfg

class emotion:
    
    def __init__(self, target):
        self.target = target
        return None
    

    def face_detector(img):
        face_classifier = cv2.CascadeClassifier(cfg.HAAR_CASCADE)
        gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces is ():
            return (0,0,0,0), np.zeros((48,48), np.uint8), img
        
        allfaces = []   
        rects = []

        for (x,y,w,h) in faces:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
            allfaces.append(roi_gray)
            #rects.append((x,w,y,h))
            rects.append((x,y,w,h))
            
        return rects, allfaces, img

    #VGG16 support RGB, so need to pass back 3 channel for VGG16
    def analyse_emotionVGG16(rects, faces, image, model,flag):
        i = 0
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        
        for face in faces:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
             # make a prediction on the ROI, then lookup the class
            
            ###Convert to RGB (for VGG16 only)
            gray_three = cv2.merge([roi[0],roi[0],roi[0]])
            gray_three = gray_three.reshape(1,48,48,3)
            
            preds = model.predict(gray_three)
            #print(type(preds))
            label = cfg.class_labels[preds.argmax()]# + "("+ str(preds[preds.argmax()]) +")"  
 
            #assign prediction, and confidence interval to dataframe
            df1 = pd.DataFrame({"Predictions":[label], "Pred_Mapping":cfg.pos_neg_mapping[label], cfg.class_labels[0]:preds[0][0], cfg.class_labels[1]:preds[0][1], cfg.class_labels[2]:preds[0][2], 
                                cfg.class_labels[3]:preds[0][3], cfg.class_labels[4]:preds[0][4], cfg.class_labels[5]:preds[0][5], 
                                cfg.class_labels[6]:preds[0][6]})
            #print("df1",df1)
            
            if flag=="I": #image is using BGR format
                color = cfg.EMOTION_COLOR_MAP[cfg.class_labels[preds.argmax()]]
            elif flag=="V": #video is ing RGB format
                color = cfg.RGB_EMOTION_COLOR_MAP[cfg.class_labels[preds.argmax()]]
                
            #Overlay our detected emotion on our pic
            label_position = (rects[i][0] +1, abs(rects[i][1] - 5))
        
            # get the width and height of the text box
            fontsize = cfg.fontsize
            (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontsize, thickness=1)[0]
            # set the text start position
            # make the coords of the box with a small padding of two pixels
            box_coords = ((label_position[0]-2, label_position[1]+5), (label_position[0] + text_width + 2, label_position[1] - text_height-2))
            cv2.rectangle(image, box_coords[0], box_coords[1], color, cv2.FILLED) #this rect is for emotion labels as background
            cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,fontsize, cfg.WHITE, 1)
            cv2.rectangle(image,rects[i],color,1) #this rect is for face detection
            i+=1
            
            #append data rows per face
            df2 = df2.append(df1, ignore_index = True)
            #print("df2:\n",df2) 
        return image, df2
        #         try:
            # Convert image to grayscale        
#         except:
#             print("An exception has occurred occurred in module [face_detector]")
            

    #VGG16 support RGB, so need to pass back 3 channel for VGG16
    def analyse_emotionVGG9(rects, faces, image, model, flag):
        i = 0
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        
        for face in faces:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
             # make a prediction on the ROI, then lookup the class
            preds = model.predict(roi)
            #print(type(preds))
            label = cfg.class_labels[preds.argmax()]# + "("+ str(preds[preds.argmax()]) +")"  
 
            #assign prediction, and confidence interval to dataframe
            df1 = pd.DataFrame({"Predictions":[label], "Pred_Mapping":cfg.pos_neg_mapping[label], cfg.class_labels[0]:preds[0][0], cfg.class_labels[1]:preds[0][1], cfg.class_labels[2]:preds[0][2], 
                                cfg.class_labels[3]:preds[0][3], cfg.class_labels[4]:preds[0][4], cfg.class_labels[5]:preds[0][5], 
                                cfg.class_labels[6]:preds[0][6]})
            #print("df1",df1)

            if flag=="I": #image is using BGR format
                color = cfg.EMOTION_COLOR_MAP[cfg.class_labels[preds.argmax()]]
            elif flag=="V": #video is ing RGB format
                color = cfg.RGB_EMOTION_COLOR_MAP[cfg.class_labels[preds.argmax()]]
            
            #Overlay our detected emotion on our pic
            label_position = (rects[i][0] +1, abs(rects[i][1] - 5))
        
            # get the width and height of the text box
            fontsize = cfg.fontsize
            (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontsize, thickness=1)[0]
            # set the text start position
            # make the coords of the box with a small padding of two pixels
            box_coords = ((label_position[0]-2, label_position[1]+5), (label_position[0] + text_width + 2, label_position[1] - text_height-2))
            cv2.rectangle(image, box_coords[0], box_coords[1], color, cv2.FILLED) #this rect is for emotion labels as background
            cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,fontsize, cfg.WHITE, 1)
            cv2.rectangle(image,rects[i],color,1) #this rect is for face detection
            i+=1
            
            #append data rows per face
            df2 = df2.append(df1, ignore_index = True)
            #print("df2:\n",df2) 
        return image, df2


    #function to check video lenght and return second interval 
    def get_slice_interval(length):
        #within 3 min (180 s) will be 1 sec interval
        if length < 180:
            interval = 1
            
        #within 6 min (360 s) will be 2 sec interval
        elif length >= 180 and length < 360:
            interval = 2
            
        #within 9 min (540 s) will e 3 sec interval
        elif length >= 360 and length < 540:
            interval = 3
            
        #within 12 min (720 s) will be 4 sec interval
        elif length >= 540 and length < 720:
            interval = 4
            
        #within 15 min (900 s) will be 5 sec interval
        elif length >= 720 and length < 900:
            interval = 5
            
        #withih 18 min (1080) will be 6 sec interval
        elif length >= 900:
            interval = 6
            
        return interval
            
    

    
    