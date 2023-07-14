'''
{
  "predictions-ploted": "Image with plotted predictions", [done]
  "predictions-info": { [done]
    "label": [polygon-coords],
    "label2": [polygon-coords]
  },
  "Load-time": "seconds to load model"
  "Inference-time": "seconds to perform inference" [done]
  "damages-detected": ["dent of fender","{damage-type} on {panel}"] [done]
}

## Different types of response

1. {'pred-info': {}, 'pred-ploted': [], 'inference-time': 0, 'damages-detected': [], 'inference-sever-status': '[StatusCode.UNAVAILABLE] failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:8091: Failed to connect to remote host: Connection refused'} // Unable to connect to inference server

2.{'pred-info': {'pred-label':polygon,'pred-label':polygon}, 'pred-ploted': array('Predictions on image'), 'inference-time': '0.069 sec.', 'damages-detected': [], 'inference-sever-status': True} // able to connect to inference server
'''

try:
    from scripts.utils.post_process_yv8 import _post_processing,extract_polygons
except:
    from utils.post_process_yv8 import _post_processing,extract_polygons
from tritonclient.utils import InferenceServerException
import tritonclient.grpc as grpcclient
import numpy as np
import base64
import time
import yaml
import cv2
import pdb

class panel_inference(object):
    """docstring for panel_inference"""
    def __init__(self,model_name,config_path):
        super(panel_inference, self).__init__()
        self.model_name = model_name
        self.yaml_info = self.read_yaml(config_path,model_name)
        self.client = grpcclient.InferenceServerClient(url=self.yaml_info['inference_url'])
        # self.img_path = img_path
        # self.str_img = self.image_base64()

    def read_yaml(self,yaml_path,model_name):
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

        return yaml_data[model_name]

    def check_server_connectivity(self):
        ''' Use this function to check if server is live
        '''
        try:
            self.client.is_server_live() == True
            return True,'is_live'

        except InferenceServerException as ex:
            error_message = str(ex)
            return False,error_message

    def yolo_inference(self,ori_image):
        ''' Use this function to perform inference on base64 image and return processed model op
        '''
        st_time = time.time()
        # server_sts = self.check_server_connectivity()
        server_sts = True,'is_live'
        if server_sts[0]:
            response = {'pred-poly':{},'inference-seconds':0,'damages-detected':[],'inference-sever-status':True,'cls-score':{}}
            # ori_image = self.read_base64_img(base64_img)
            img_h,img_w,_ = ori_image.shape
            image = self.prepocess_img(ori_image)
            try:
                op0,op1 = self.tis_inference(image)
            except:
                return f"Unable to perform inference on titron inference server, Plese check status of inference server"
            img_cpy = ori_image.copy()
            predictions = _post_processing(np.array(op0),np.array(op1),image,retina_masks=self.yaml_info['retina_mask'],nc=len(self.yaml_info['labels_info']))
            if predictions[0].get('masks') != None:
                for mask,label,score in zip(predictions[0]['masks'][0],predictions[0]['classes'].numpy().astype('int'),predictions[0]['scores'].numpy()):
                    c_label,color = self.yaml_info['labels_info'][label]
                    re_mask = cv2.resize(mask.numpy(),(img_w,img_h))
                    poly = extract_polygons(re_mask*255)
                    response['pred-poly'].setdefault(c_label, []).append(poly[0])
                    response['cls-score'].setdefault(c_label, []).append(float(score))
                    pts = np.array(poly)
                    pts = pts.reshape(-1, 1, 2)
                    cv2.fillPoly(img_cpy, pts=[pts], color=color)

                response['inference-seconds'] = "{:.3f}".format(time.time()-st_time)
                blended_image = cv2.addWeighted(ori_image, 1-self.yaml_info['opacity'], img_cpy, self.yaml_info['opacity'], 0)
                response['pred-ploted'] = str(self.cv_base64(blended_image))
                return response

        ## if unable to connect to TIS inference server
        return {'pred-poly':{},'pred-ploted':str(self.cv_base64(ori_image)),'inference-time':'','damages-detected':[],'inference-sever-status':server_sts[1],'cls-score':''}
    
    def tis_inference(self,image):
        ''' Use this function to perform inference on Titron inference server
        '''
        input_tensors = [grpcclient.InferInput(self.yaml_info['input_lyr_name'], image.shape, self.yaml_info['input_frmt'])]
        input_tensors[0].set_data_from_numpy(image)
        results = self.client.infer(model_name=self.model_name, inputs=input_tensors)

        model_op = []
        for op_layer in self.yaml_info['output_lyt_name']:
            model_op.append(results.as_numpy(op_layer))

        return model_op[0],model_op[1]

    def cv_base64(self,img):

        _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)

        return im_b64

    def read_base64_img(self,base64_img):
        ''' Use this function to convert base64 image to array
        '''
        image_bytes = base64.b64decode(base64_img)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        #### Visualize and check image
        # resize_img = cv2.resize(image,(640,640))
        # cv2.imshow('image',resize_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return image

    def prepocess_img(self,ori_image):
        ''' Per-Process image and feed to model
        '''
        resiz_image = cv2.resize(ori_image, (self.yaml_info['img_res'],self.yaml_info['img_res']))
        image = resiz_image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))  # Reshape to (C, H, W)
        image = np.ascontiguousarray(image)
        # Create a batch with a single image
        image = np.expand_dims(image, axis=0)
        return image

