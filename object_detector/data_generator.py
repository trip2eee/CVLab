import cv2
import numpy as np
import os

from anchor import Anchor
from bbox import BBox

class DataGenerator():
    def __init__(self, width, height, channel):        
        self.image = np.zeros((height, width ,channel))
        self.labels = []

    def save(self, file_path):
        cv2.imwrite(filename=file_path+".png", img=self.image)

        with open(file_path + ".txt", "w") as f:
            for l in self.labels:
                f.write(str(l[0]) + " ")
                
                f.write(str(l[1][0]) + " ")
                f.write(str(l[1][1]) + " ")
                f.write(str(l[1][2]) + " ")
                f.write(str(l[1][3]) + "\n")
                
            
    
    def draw_rect(self, rect, color):
        pt1 = (rect[0], rect[1])
        pt2 = (rect[2], rect[3])

        cv2.rectangle(img=self.image, pt1=pt1, pt2=pt2, color=color, thickness=2)
        self.labels.append((0, rect))

    def draw_ellipse(self, rect, color):
        ct = (int((rect[0] + rect[2]) * 0.5), int((rect[1] + rect[3]) * 0.5))
        r = (int((rect[2] - rect[0]) * 0.5), int((rect[3] - rect[1]) * 0.5))

        #cv2.circle(img=self.image, center=ct, radius=r, color=color)

        cv2.ellipse(img=self.image, center=ct, axes=r, angle=0, startAngle=0, endAngle=360, color=color, thickness=2)
        self.labels.append((1, rect))

class DataLoader():
    def __init__(self, path):
        self.images = []
        self.bb_labels = []

        # predefined anchor sizes [width, height]
        self.anchor_sizes = [[3, 3],
                             [7, 7]]

        self.anchor_cls = []
        self.anchor_reg = []        
        
        file_list = os.scandir(path)
        for fn in file_list:
            if fn.name[-3:] == "png":
                image = cv2.imread(fn.path)
                self.images.append(image)

                bb_labels = []
                with open(fn.path[0:-3]+"txt", "r") as f:
                    while True:
                        bb = f.readline()

                        if bb == "":
                            break

                        bb = bb.split()
                        for i in range(len(bb)):
                            bb[i] = int(bb[i])

                        bb_labels.append(bb)
                self.bb_labels.append(bb_labels)

    def show(self):
        cv2.namedWindow("img")

        for i in range(len(self.images)):            
            for label in self.bb_labels[i]:
                pt1 = (label[1], label[2])
                pt2 = (label[3], label[4])

                cv2.rectangle(img=self.images[i], pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=1)

            cv2.imshow("img", self.images[i])
            cv2.waitKey(0)

    """
        @fn __compute_iou()
        @brief This method computes IoU between an anchor and a box
        @param x_a: anchor x
        @param y_a: anchor y
        @param w_a: anchor width
        @param h_a: anchor height
        @param scale: scale factor (label coordinate) = (feature coordinate) * scale.
    """
    def __compute_iou(self, anchor: Anchor, scale, bbox: BBox):
        
        abox = anchor.to_box()
        abox = abox * scale

        # intersection
        x0 = max(abox.x0, bbox.x0)
        x1 = min(abox.x1, bbox.x1)

        y0 = max(abox.y0, bbox.y0)
        y1 = min(abox.y1, bbox.y1)
                
        xi = x1 - x0
        yi = y1 - y0

        iou = 0
        if xi > 0 and yi > 0:
            ai = xi * yi
            au = ((abox.x1 - abox.x0) * (abox.y1 - abox.y0)) + ((bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)) - ai

            iou = ai / au

        return iou

    def __compute_reg(self, anchor, bb_gt):
        x_gt = (bb_gt.x0 + bb_gt.x1) * 0.5
        y_gt = (bb_gt.y0 + bb_gt.y1) * 0.5
        
        w_gt = bb_gt.x1 - bb_gt.x0
        h_gt = bb_gt.y1 - bb_gt.y0

        reg = [0] * 4
        reg[0] = (x_gt - anchor.x) / anchor.w
        reg[1] = (y_gt - anchor.y) / anchor.h
        reg[2] = np.log(w_gt / anchor.w)
        reg[3] = np.log(h_gt / anchor.h)

        return reg

    def label_anchors(self, scale, iou_true, iou_false):
        self.anchor_scale = scale
        num_anchor_sizes = len(self.anchor_sizes)

        # for each image
        for i in range(len(self.images)):

            img_height = self.images[i].shape[0]
            img_width = self.images[i].shape[1]

            label_height = int(img_height / scale)
            label_width = int(img_width / scale)

            
            aclass = np.zeros((label_height, label_width, num_anchor_sizes))
            areg = np.zeros((label_height, label_width, num_anchor_sizes, 4))

            label_tp = 1
            label_dc = 255

            #label_tp = 255
            #label_dc = 100

            # for each bounding box
            for bb in self.bb_labels[i]:                
                
                bbox_gt = BBox(bb[1], bb[2], bb[3], bb[4])

                max_iou = 0.0
                idx_anchor_max = 0

                # for each anchor
                for idx_anchor in range(num_anchor_sizes):
                #for idx_anchor in range(1, 2):
                    w_a = self.anchor_sizes[idx_anchor][0]     # anchor width
                    h_a = self.anchor_sizes[idx_anchor][1]     # anchor height

                    # for each anchor point
                    for y_a in range(label_height):
                        for x_a in range(label_width):
                            
                            anchor = Anchor(x_a, y_a, w_a, h_a)
                            
                            iou = self.__compute_iou(anchor=anchor, scale=scale, bbox=bbox_gt)

                            # positive
                            if iou >= iou_true:
                                aclass[y_a, x_a, idx_anchor] = label_tp
                                areg[y_a, x_a, idx_anchor] = self.__compute_reg(anchor=anchor*scale, bb_gt=bbox_gt)
                            
                            # don't care
                            elif iou > iou_false and aclass[y_a, x_a, idx_anchor] == 0:
                                aclass[y_a, x_a, idx_anchor] = label_dc
                                areg[y_a, x_a, idx_anchor,:] = self.__compute_reg(anchor=anchor*scale, bb_gt=bbox_gt)
                            
                            # find the anchor with the maximum IoU.
                            if iou >= max_iou:
                                max_iou = iou
                                idx_anchor_max = idx_anchor
                                x_max, y_max = x_a, y_a


                #print("anchor: {}, max_iou: {}".format(idx_anchor_max, max_iou))

                if max_iou > iou_false:
                    aclass[y_max, x_max, idx_anchor_max] = label_tp
                    

            self.anchor_cls.append(aclass)
            self.anchor_reg.append(areg)
            
            if False:
                cv2.namedWindow("img")
                cv2.imshow("img", self.images[i])

                cv2.namedWindow("anchor0")
                cv2.imshow("anchor0", self.anchor_cls[i][:,:,0]/255)

                cv2.namedWindow("anchor1")
                cv2.imshow("anchor1", self.anchor_cls[i][:,:,1]/255)

                cv2.waitKey(0)
                

        return self.anchor_cls



if __name__ == "__main__":

    for i in range(10):
        dataGen = DataGenerator(256, 256, 3)

        r1 = 50 #+ i*2
        s1 = 10

        r2 = 20 #+ i * 1
        s2 = 20

        dataGen.draw_ellipse((s1*i, s1*i, r1 + s1*i, r1 + s1*i), (0, 255, 0))
        dataGen.draw_ellipse((s2*i, s2*i, r2 + s2*i, r2 + s2*i), (0, 255, 0))

        #dataGen.draw_ellipse((50, s1*i + 10, r1 + 50, r1 + s1*i + 10), (0, 255, 0))
        #dataGen.draw_ellipse((150, s2*i + 10, r2 + 150, r2 + s2*i + 10), (0, 255, 0))

        dataGen.save("rcnn_data/data{0:02d}".format(i))

    loader = DataLoader("rcnn_data")
    anchor_labels = loader.label_anchors(8, 0.7, 0.30)

    loader.show()

