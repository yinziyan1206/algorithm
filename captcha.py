#!/usr/bin/env python
__author__ = 'ziyan.yin'

import cv2
import imageutils


def slide_block(background, block):
    background_img = imageutils.load_image_online(background, method='CV2')
    block_img = imageutils.load_image_online(block, method='CV2')

    block_blur = cv2.GaussianBlur(block_img, (3, 3), 0)
    background_blur = cv2.GaussianBlur(background_img, (3, 3), 0)

    target = cv2.Canny(block_blur, 50, 150)
    template = cv2.Canny(background_blur, 50, 150)

    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    tl = max_loc
    return tl

