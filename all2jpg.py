#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:39:34 2020
@author: Mekakuactor
"""
 
from PIL import Image
import os
 
def IsValidImage(img_path):
    """
    判断文件是否为有效（完整）的图片
    :param img_path:图片路径
    :return:True：有效 False：无效
    """
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid
 
def transimg(path):
    """
    转换图片格式
    :param img_path:图片路径
    :return: True：成功 False：失败
    """
    for filename in os.listdir(path):
        img_path = path + '/' + filename
        if IsValidImage(img_path):
            try:
                str = img_path.rsplit(".")
                if str[-1] == 'jpg' or str[-1] == 'jpeg' or str[-1] == 'JPG' or str[-1] == 'JPEG':
                    pass
                else:
                    str = img_path.rsplit(".", 1)
                    output_img_path = str[0] + ".jpg"
                    print(output_img_path)
                    im = Image.open(img_path)
                    rgb_im = im.convert('RGB')
                    rgb_im.save(output_img_path)
            except:
                print("error1:", img_path)
        else:
            print("error2:", img_path)
 
if __name__ == '__main__':
    path = './data_aug'
    print(transimg(path))
