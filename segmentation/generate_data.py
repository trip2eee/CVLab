import numpy as np
import cv2
import os

# class
# 0: background
# 1: ellpse
# 2: rectangle
# 3: intersection

object_classes = 2
max_object = 2
num_data = 64
num_color = 3

num_size = 3

image_width = 256
image_height = 256

def generate(dst_path):

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    for i in range(num_data):
        
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        label = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        
        label_ellipse = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        label_rect = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        image_path = os.path.join(dst_path, "image_{:04d}.png".format(i))
        label_path = os.path.join(dst_path, "label_{:04d}.png".format(i))

        thickness = 10

        for idx_obj in range(max_object):
            obj_class = np.random.randint(0, object_classes) + 1
            if obj_class > 0:
                rows = np.random.randint(0, num_size)
                cols = np.random.randint(0, num_size)
                
                # color_r = np.random.randint(0, num_color) * 30 + 100
                # color_g = np.random.randint(0, num_color) * 30 + 100
                # color_b = np.random.randint(0, num_color) * 30 + 100

                color_r = 0
                color_g = 255
                color_b = 0

                width = 100 + (cols * 10)
                height = 100 + (rows * 10)

                x = np.random.randint(image_width - width)
                y = np.random.randint(image_height - height)

                label_color = obj_class*50
                if obj_class==1:
                    center_x = x + int(width/2)
                    center_y = y + int(height/2)

                    cv2.ellipse(image, (center_x, center_y), (int(width/2), int(height/2)), 0, 0, 360, (color_r, color_g, color_b), thickness=thickness)
                    cv2.ellipse(label_ellipse, (center_x, center_y), (int(width/2), int(height/2)), 0, 0, 360, (label_color, label_color, label_color), thickness=thickness)
                elif obj_class==2:
                    cv2.rectangle(image, (x, y), (x+width, y+height), (color_r, color_g, color_b), thickness=thickness)
                    cv2.rectangle(label_rect, (x, y), (x+width, y+height), (label_color, label_color, label_color), thickness=thickness)

        label = label_ellipse + label_rect
        cv2.imwrite(image_path, image)
        cv2.imwrite(label_path, label)

        
generate('train')
generate('validation')
generate('test')

