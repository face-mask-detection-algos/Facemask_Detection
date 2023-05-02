import cv2
import os
import argparse
import pandas as pd
import numpy as np 
from PIL import Image,ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="../images")
    parser.add_argument("--model_path", type=str, default='model/custom_4370_32_100_v2.h5')
    parser.add_argument("--weights_path", type=str, default='res10_300x300_ssd_iter_140000.caffemodel')
    parser.add_argument("--prototxt_path", type=str, default='deploy.prototxt.txt')
    parser.add_argument("--output_path", type=str, default='output.csv')
    args = parser.parse_args()
    return args

def model_inference(classifier, backbone, img_path):
    predictions = []
    image=cv2.imread(img_path)
    (h,w)=image.shape[:2]
    blob=cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))
    backbone.setInput(blob)
    detections = backbone.forward()
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.3:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
            if startX>w or startY>h or endX<0 or endY<0:
                print("Outbox")
                continue
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
            face=image[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(96,96))
            face=img_to_array(face)
            face=preprocess_input(face)
            face=np.expand_dims(face,axis=0)
            (withoutMask,mask)=classifier.predict(face)[0]

            category = "mask" if mask>withoutMask else "nomask"
            predictions.append([category, confidence, startX/w, startY/h, (endX-startX)/w, (endY-startY)/h])
    return predictions
    
def append_predictions_to_dataframe(df, image_path, predictions):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    for prediction in predictions:
        df.loc[len(df)] = [image_name, os.path.abspath(image_path), *prediction]
            

if __name__ == "__main__":
    args = get_args()

    if os.path.dirname(args.output_path).strip() != "":
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    backbone=cv2.dnn.readNet(args.weights_path, args.prototxt_path)
    classifier=load_model(args.model_path)

    results = pd.DataFrame(columns=['image_name', 'image_path', 'class', 'confidence', 'x', 'y', 'w', 'h'])

    image_paths = [os.path.join(args.folder,f) for f in os.listdir(args.folder) if os.path.isfile(os.path.join(args.folder, f))]
    for image_path in image_paths:
        predictions = model_inference(classifier, backbone, image_path)
        append_predictions_to_dataframe(results, image_path, predictions)
    
    results.to_csv(args.output_path, index=False)
