import tensorflow as tf

class RPNet(tf.keras.Model):
    def __init__(self, filters, num_anchors):
        super(RPNet, self).__init__()

        self.num_anchors = num_anchors

        self.conv2d_11 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation=tf.keras.activations.relu)
        self.conv2d_12 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation=tf.keras.activations.relu)
        self.max_pool1 = tf.keras.layers.MaxPooling2D()

        self.conv2d_21 = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=3, padding='same', activation=tf.keras.activations.relu)
        self.conv2d_22 = tf.keras.layers.Conv2D(filters=filters*2, kernel_size=3, padding='same', activation=tf.keras.activations.relu)
        self.max_pool2 = tf.keras.layers.MaxPooling2D()

        self.conv2d_31 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=3, padding='same', activation=tf.keras.activations.relu)
        self.conv2d_32 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=3, padding='same', activation=tf.keras.activations.relu)
        self.conv2d_33 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=3, padding='same', activation=tf.keras.activations.relu)
        self.conv2d_34 = tf.keras.layers.Conv2D(filters=filters*4, kernel_size=3, padding='same', activation=tf.keras.activations.relu)
        self.max_pool3 = tf.keras.layers.MaxPooling2D()

        # intermediate layer.
        self.conv_interm = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.keras.activations.relu)

        # cls layer.
        self.conv_cls = tf.keras.layers.Conv2D(filters=num_anchors*2, kernel_size=1, padding='same', activation=tf.keras.activations.linear)
        self.softmax = tf.keras.layers.Softmax()

        # reg layer.
        self.conv_reg = tf.keras.layers.Conv2D(filters=num_anchors*4, kernel_size=1, padding='same', activation=tf.keras.activations.linear)

        self.loss_cls = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
        self.loss_reg = tf.keras.losses.Huber(reduction='none')     # soft L1 loss.
        self.accuracy_cls = tf.keras.metrics.SparseCategoricalAccuracy()

    def call(self, x):

        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.max_pool1(x)

        x = self.conv2d_21(x)
        x = self.conv2d_22(x)
        x = self.max_pool2(x)

        x = self.conv2d_31(x)
        x = self.conv2d_32(x)
        x = self.conv2d_33(x)
        x = self.conv2d_34(x)
        x = self.max_pool3(x)

        interm = self.conv_interm(x)        # intermediate layer.

        cls_out = self.conv_cls(interm)     # class output

        cls_out_shape = tf.shape(cls_out)

        # (batch, height, width, anchor, prob).
        cls_out = tf.reshape(cls_out, (cls_out_shape[0], cls_out_shape[1], cls_out_shape[2], self.num_anchors, 2))
        cls_out = self.softmax(cls_out)

        reg_out = self.conv_reg(interm)     # regression output
        reg_out = tf.reshape(reg_out, (cls_out_shape[0], cls_out_shape[1], cls_out_shape[2], self.num_anchors, 4))

        return cls_out, reg_out

    def cls_accuracy(self, cls_target, cls_pred):

        valid = tf.less(cls_target, 2.0)     # cls is valid only if <= 1        
        valid = tf.cast(valid, tf.float32)
        
        _cls_target = cls_target * valid                
        _cls_target = tf.cast(_cls_target, tf.int32)

        _cls_pred = tf.math.argmax(cls_pred, axis=-1, output_type=tf.int32)
        _cls_pred = _cls_pred * tf.cast(valid, tf.int32)
        _cls_pred = tf.cast(_cls_pred, tf.int32)

        cls_eq = tf.math.equal(_cls_target, _cls_pred)
        cls_eq = tf.cast(cls_eq, tf.float32)
        acc = tf.math.reduce_mean(cls_eq)

        return acc

    def cls_loss(self, cls_target, cls_pred):
        
        valid = tf.less(cls_target, 2.0)     # cls is valid only if <= 1
        valid = tf.cast(valid, tf.float32)

        valid_exp = tf.expand_dims(valid, 3)

        _cls_target = cls_target * valid        

        _loss = self.loss_cls(_cls_target, cls_pred)
        _loss = _loss * valid_exp
        _loss = tf.math.reduce_mean(_loss)

        return _loss

    def reg_loss(self, cls_target, reg_target, reg_pred):
        
        valid = tf.less(cls_target, 2.0)     # cls is valid only if <= 1
        valid = tf.cast(valid, tf.float32)

        valid_exp = tf.expand_dims(valid, 3)

        _loss = self.loss_reg(reg_target, reg_pred)
        _loss = _loss * valid_exp
        _loss = tf.math.reduce_mean(_loss)

        return _loss

    def rpn_loss(self, cls_target, cls_pred, reg_target, reg_pred):
        
        w_cls = 1.0
        w_reg = 10.0

        #_loss = cls_loss(cls_pred, cls_target) + w_reg * reg_loss(reg_pred, reg_target)
        loss_cls = self.cls_loss(cls_target, cls_pred)
        loss_reg = self.reg_loss(cls_target, reg_target, reg_pred)

        _loss = w_cls*loss_cls + w_reg*loss_reg

        return _loss




 