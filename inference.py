import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence, load_img, img_to_array
from tensorflow.keras.models import load_model

def class_to_age(pre):
    """
    0: [0-10)
    1: [10-20)
    2: [20-30)
    ....
    """
    d = {
        0:"0-10", 1:"10-20", 2:"20-30", 3:"30-40", 4:"40-50", 5:"50-60", 6:"60-70", 7:"70-80", 8:"80-90"
    }
    return d[pre]
def preprocess(path):
    img = load_img(path,  target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, 0)
    return img

def inference(model, img):
    pre = model(img)
    gender = 0 if pre[0] < 0.5 else 1
    age = np.argmax(pre[1])
    d = class_to_age(age)
    return gender, d

model = load_model('mobilenetv3small31.h5')
img = preprocess('1.jpg')

print(inference(model, img))
