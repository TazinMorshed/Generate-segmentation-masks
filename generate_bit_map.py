# Preparing custom dataset
import os 
import json
import cv2
import numpy as np 


from ast import literal_eval
import numpy as np
from sklearn.cluster import KMeans
import plotly as px
from shapely.geometry import Polygon
import pandas as pd
import re
# import some common libraries
import numpy as np
import os, json, cv2, random

from shapely.geometry import Polygon

# import some common detectron2 utilities
import csv
import glob
# importing the module
import collections

#directory where images are location #CHANGE 
img_dir="val"

# load the VGG Annotator JSON file
json_file = os.path.join(img_dir, "via_region_data.json")
print(json_file)
with open(json_file) as f:
    imgs_anns = json.load(f)





dataset_dicts = []
# # loop through the entries in the JSON file
for idx, v in enumerate(imgs_anns.values()):


    record = {}
#     # add file_name, image_id, height and width information to the records
    filename = os.path.join(img_dir, v["filename"])
    height, width = cv2.imread(filename).shape[:2]


    record["file_name"] = filename
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width

    
    annos = v["regions"]
    
    num_of_annotations = len(annos)


           
            
    
 
    
    objs = []
    # one image can have multiple annotations, therefore this loop is needed
    for annotation in annos:
        # reformat the polygon information to fit the specifications
        anno = annotation["shape_attributes"]
        # print(anno)
        px = anno["all_points_x"]
        # print("X----- ", px)
        py = anno["all_points_y"]
        # print("Y----- ", py)
        poly = [[x + 0.5, y + 0.5] for x, y in zip(px, py)]
        # poly = [p for x in poly for p in x]
        
       
        region_attributes = annotation["region_attributes"]
        
        # print("Bounding box : ", np.min(px), " ",np.min(py), " ", np.max(px), " ",np.max(py) )
        # print("Bouonding box mode : ", BoxMode.XYXY_ABS )
        # np_poly = np.array(poly)
        # print("segmentation : ", np_poly.shape)


        objs.append(poly)

   

 
  
   
    image_name = record["file_name"]
    
    image = cv2.imread(image_name)
    for i, x in enumerate(objs):
        # print( record["file_name"])
        # print(x)

        if image is None:
          continue
        # reformat the polygon information to fit the specifications

        poly = x
        poly = np.array(poly, dtype=np.int32)
        poly = poly.reshape((-1, 1, 2))
        
        #for lines 
        # image = cv2.polylines(image, [poly], 
        #               True, (255, 0, 0), 2)
        
        #for fillups
        image = cv2.fillPoly(image, [poly], (255,150,255))
        cv2.imwrite(f"mask_image/mask_image_0{idx}.png", image)
        
        
    
        




