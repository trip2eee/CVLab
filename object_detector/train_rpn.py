from data_generator import DataLoader
from rp_net import RPNet

import numpy as np
import tensorflow as tf

import cv2

if __name__ == "__main__":
    tf.random.set_seed(123)
    print(tf.executing_eagerly())

    loader = DataLoader("rcnn_data")
    num_anchors = len(loader.anchor_sizes)    

    anchor_labels = loader.label_anchors(8, 0.7, 0.30)
    scale = loader.anchor_scale

    images = loader.images
    images = np.reshape(images, (-1, 256, 256, 3))
    images = images.astype(np.float32) / 255

    output_width = int(256 / scale)
    output_height = int(256 / scale)

    target_cls = loader.anchor_cls
    target_cls = np.reshape(target_cls, (-1, output_height, output_width, num_anchors))
    target_cls = target_cls.astype(np.float32)
    
    target_reg = loader.anchor_reg
    target_reg = np.reshape(target_reg, (-1, output_height, output_width, num_anchors*4))
    target_reg = target_reg.astype(np.float32)

    rp_net = RPNet(filters=8, num_anchors=num_anchors)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


    @tf.function
    def train_step(images, target_cls, target_reg):
        with tf.GradientTape() as tape:
            cls_out, reg_out = rp_net(images)
            loss = rp_net.rpn_loss(target_cls, cls_out, target_reg, reg_out)
            accuracy = rp_net.cls_accuracy(target_cls, cls_out)

        gradients = tape.gradient(loss, rp_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, rp_net.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy)

    @tf.function
    def test_step(images, target_cls, target_reg):
        cls_out, reg_out = rp_net(images)
        t_loss = rp_net.rpn_loss(target_cls, cls_out, target_reg, reg_out)

        test_loss(t_loss)
        #test_accuracy(target_cls, cls_out)

    EPOCHS = 1000
    if False:
        for epoch in range(EPOCHS):
            for i in range(images.shape[0]):
                x = images[i].reshape((-1, 256, 256, 3))

                y_cls = target_cls[i].reshape((-1, output_height, output_width, num_anchors))
                y_reg = target_reg[i].reshape((-1, output_height, output_width, num_anchors, 4))
                
                train_step(x, y_cls, y_reg)

            #train_step(images, target_cls, target_reg)

            for i in range(images.shape[0]):

                x = images[i].reshape((-1, 256, 256, 3))

                y_cls = target_cls[i].reshape((-1, output_height, output_width, num_anchors))
                y_reg = target_reg[i].reshape((-1, output_height, output_width, num_anchors, 4))
                

                test_step(x, y_cls, y_reg)

            template = 'Epoch: {}, loss: {:.06f}, accuracy: {:.06f}, test loss: {:.06f}'
            print (template.format(epoch+1,
                                    train_loss.result(),
                                    train_accuracy.result()*100,
                                    test_loss.result()))
    
        rp_net.save_weights('rpn_weights')
    else:
        rp_net.load_weights('rpn_weights')

    # test
    for i in range(images.shape[0]):

        img = images[i].reshape((-1, 256, 256, 3))
        cls_out, reg_out = rp_net(img)

        #img = x * 255
        #img = img.astype(np.uint8)

        y_cls = target_cls[i].reshape((-1, output_height, output_width, num_anchors))
        y_reg = target_reg[i].reshape((-1, output_height, output_width, num_anchors, 4))

        cls_out = cls_out.numpy()
        reg_out = reg_out.numpy()

        for i in range(cls_out.shape[1]):
            for j in range(cls_out.shape[2]):
                for k in range(cls_out.shape[3]):
                    if cls_out[0, i, j, k, 1] > 0.5:

                        img[0, i*scale, j*scale, 2] = 1
                        w_a = loader.anchor_sizes[k][0] * scale
                        h_a = loader.anchor_sizes[k][1] * scale

                        tx = reg_out[0, i, j, k, 0]
                        ty = reg_out[0, i, j, k, 1]
                        tw = reg_out[0, i, j, k, 2]
                        th = reg_out[0, i, j, k, 3]

                        x = tx*w_a + j*scale
                        y = ty*h_a + i*scale

                        w = w_a * np.exp(tw)
                        h = h_a * np.exp(th)

                        x0 = int(x - w*0.5)
                        y0 = int(y - h*0.5)

                        x1 = int(x + w*0.5)
                        y1 = int(y + h*0.5)

                        cv2.line(img[0], (x0, y0), (x1, y0), (0, 0, 1.0))
                        cv2.line(img[0], (x0, y0), (x0, y1), (0, 0, 1.0))
                        cv2.line(img[0], (x1, y0), (x1, y1), (0, 0, 1.0))
                        cv2.line(img[0], (x0, y1), (x1, y1), (0, 0, 1.0))


        cv2.namedWindow('test')
        cv2.imshow('test', img[0])
        cv2.waitKey(0)