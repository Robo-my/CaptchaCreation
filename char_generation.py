#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:18:12 2019

@author: anas
"""

import string
import numpy as np
import cv2
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random
backgroundlist = os.listdir('background')


def saveChar(char, charFont):
    os.mkdir('charDataset//'+charFont[:-4]+'//' + str(char))

    # find crop width and height of each char
    img = np.zeros([500, 500, 3], dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(charFont, 100)
    draw.text((10, 10), char, (255, 255, 255), font=font)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    H = 83
    y = y-int((H-h)/2)
    x = x-1
    y = y-1
    w = w+2
    h = h+2
    # generate 20 images each char using above postion with background
    for i in range(20):
        img = np.zeros([500, 500, 3], dtype=np.uint8)
        rand_int = random.randint(0, len(backgroundlist)-2)
        background = cv2.imread('background/'+backgroundlist[rand_int])
        background = cv2.resize(background, (500, 500))
        img = cv2.addWeighted(img, 0.5, background,
                              random.uniform(0.1, 0.5), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(charFont, 100)
        draw.text((10, 10), char, (255, 255, 255), font=font)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        crop = img[y:y + H, x:x + w]
        print('background/'+backgroundlist[rand_int])
        cv2.imwrite('charDataset//' +
                    charFont[:-4]+'//'+char+'//'+str(i)+'.jpg', crop)


charlist = string.ascii_uppercase

for i in charlist:
    saveChar(i, "big_noodle_titling.ttf")
    print(i)
numlist = '0123456789'
for i in numlist:
    saveChar(i, "big_noodle_titling.ttf")
    print(i)
