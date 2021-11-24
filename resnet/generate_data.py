import numpy as np
import cv2
import os

# class
# 0:circle
# 1:rectangle

num_class = 2
num_data = 32
num_color = 3

image_cols = 3
image_rows = 3
image_width = 224
image_height = 224

def generate(dst_path):

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    class_path = os.path.join(dst_path, 'class.txt')

    with open(class_path, 'w') as f:
    
        for i in range(num_data):
            data_class = np.random.randint(0, num_class)
            rows = np.random.randint(0, image_rows) + 1
            cols = np.random.randint(0, image_cols) + 1
            
            color_r = np.random.randint(0, num_color) * 30 + 100
            color_g = np.random.randint(0, num_color) * 30 + 100
            color_b = np.random.randint(0, num_color) * 30 + 100

            width = cols * image_width / image_cols
            height = rows * image_height / image_rows

            image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            image_path = os.path.join(dst_path, "{:04d}.png".format(i))

            if data_class==0:
                cv2.ellipse(image, (int(image_width/2), int(image_height/2)), (int(width/2), int(height/2)), 0, 0, 360, (color_r, color_g, color_b), thickness=2)
            elif data_class==1:
                cv2.rectangle(image, (int((image_width-width)/2), int((image_height-height)/2)), (int((image_width+width)/2), int((image_height+height)/2)), (color_r, color_g, color_b), thickness=2)
            cv2.imwrite(image_path, image)

            print("class: {}".format(data_class))
            f.write('{}\n'.format(data_class))

        
generate('train')
generate('validation')
generate('test')
