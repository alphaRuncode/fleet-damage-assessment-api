''' Use this script to validate data captured from payload
'''
import os
import cv2
import pdb
import yaml
import base64 
import numpy as np
from PIL import Image
from io import BytesIO

def read_images(img_data,cfg):
    ''' Use this function to convert image to readable format
    '''
    model = img_data.get('model')
    version = img_data.get('version')
    img_order = []
    imgs = []
    for key,value in img_data.items():
        if key not in ['model','version']:
            try:
                img = stringToImage(value)
                img_order.append(key)
            except:
                return None,f'Unable to read image from payload key: {key}',None,None

            # img = resize_longestside(img,cfg['pannel_resize']) ## Perform per-processing for batch inference
            imgs.append(img)

    ## Check if selected model path exists
    model_sts = check_model_path(model,version,cfg['weights_dir'])    
    if model_sts:
        return imgs,img_order,model,version
    else:
        return None,f'Unable to load model from model_type: {model}, version: {version} and model_name: model.onnx',None,None

def read_yaml(yaml_path):
    ''' Use this function to read a yaml file
    '''
    with open(yaml_path, "r") as stream:
        data = yaml.safe_load(stream)

    return data

def check_model_path(model,version,model_dir,model_name ='model.onnx'):

    model_pth = os.path.join(model_dir,model,str(version),model_name)
    return os.path.isfile(model_pth)

def resize_longestside(img,res=1080):
    ''' Use this fucntion to reisze longest side of image and also maintain aspect ratio of image
    '''
    img_size = img.shape
    max_indx = img_size.index(max(img_size))

    if max_indx == 0: ### max_height, resize width based on aspect ratio
        new_h = res
        new_w = int((res*img_size[1])/img_size[0])

    elif max_indx == 1: ### max_height, resize width based on aspect ratio
        new_w = res
        new_h = int((res*img_size[0])/img_size[1])

    return cv2.resize(img,(new_w,new_h))


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    imgdata = Image.open(BytesIO(imgdata))
    return cv2.cvtColor(np.array(imgdata), cv2.COLOR_BGR2RGB)
