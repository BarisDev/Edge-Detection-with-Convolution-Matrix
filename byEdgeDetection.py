#Author: Baris Yarar
#Num: 1306160048

import cv2
import numpy as np

def conv_transform(image):
    image_copy = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
    return image_copy

def conv(image, kernel):
    #kernel = conv_transform(kernel)
    image_h = image.shape[0]
    image_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    h = kernel_h
    w = kernel_w//2

    image_conv = np.zeros(image.shape)

    for i in range(h, image_h-h):
        for j in range(w, image_w-w):
            sum = 0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel[m][n]*image[i-h+m][j-h+n]
            image_conv[i][j] = sum
    cv2.imshow('con_image',image_conv)
    return image_conv

##2x2 convolution
def byconv(image, kernel):
    kernel = conv_transform(kernel)
    image_h = image.shape[0]#768
    image_w = image.shape[1]#1024

    kernel_h = kernel.shape[0]#3
    kernel_w = kernel.shape[1]#3
    
    image_conv = np.zeros(image.shape)
    
    for image_row in range(kernel_h, image_h - kernel_h):
        for pixel in range(kernel_w, image_row - kernel_w):
            accumulator = 0
            for kernel_row in range(kernel_w):
                for row_element in range(kernel_row):
                    accumulator = accumulator + kernel[kernel_row][row_element] * image[image_row - kernel_h + kernel_row][pixel - kernel_h + row_element]
            if pixel == 768:
                print("pixel")
            if image_row == 768:
                print(image_row)
            image_conv[image_row][pixel] = accumulator
           
    cv2.imshow('con_image', image_conv)


#product = tMatrix.dot(temp)
image = cv2.imread('C:\python\me.jpeg',0)
kernel_Robert_x = np.array([[0,1],[-1,0]])
kernel_Robert_y = np.array([[1,0],[0,-1]])
kernel_Prewitt_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
kernel_Prewitt_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
#print(kernel_Robert_x.shape[0])
byconv(image,kernel_Prewitt_x)
