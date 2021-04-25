#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = 'ziyan.yin'

from PIL import Image
import cv2
from io import BytesIO
import base64
import requests
import numpy as np


# 读取线上图片
def load_image_online(url, method='PIL'):
    res = requests.get(url)
    image_content = BytesIO(res.content)
    return load_image_from_bytes(image_content, method)


# 读取image
def load_image(url, method='PIL'):
    try:
        if method == 'PIL':
            image = Image.open(url)
        elif method == 'CV2':
            image = cv2.imread(url)
        else:
            image = Image.open(url)
        return image
    except Exception as ex:
        print(str(ex))
        return ''


# 读取image bytes
def load_image_from_bytes(content, method='PIL'):
    try:
        if method == 'PIL':
            image = Image.open(content)
        elif method == 'CV2':
            stream = np.asarray(bytearray(content.read()), dtype=np.uint8)
            image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        else:
            image = Image.open(content)
        return image
    except Exception as ex:
        print(str(ex))
        return ''


# image转base64
def image2base64(image):
    output_buffer = BytesIO()
    image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


# base64转image
def base642image(b64content):
    image_data = base64.b64decode(b64content.encode())
    return image_data


# 定高缩放图片
def resize_image(image, base_height):
    ori_w, ori_h = image.size
    new_h = base_height if base_height < ori_h else ori_h
    new_w = int(ori_w * ((new_h * 1.0) / (ori_h * 1.0)))
    return image.resize((new_w, new_h), Image.ANTIALIAS)


# 定宽缩放图片
def resize_pics_h(image, base_width):
    ori_w, ori_h = image.size
    new_w = base_width if base_width < ori_w else ori_w
    new_h = int(ori_h * ((new_w * 1.0) / (ori_w * 1.0)))
    return image.resize((new_w, new_h), Image.ANTIALIAS)

