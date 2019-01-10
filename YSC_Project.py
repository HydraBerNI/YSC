import numpy as np
import tensorflow as tf
from PIL import Image
import h5py
import glob
import os

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                padding="same")(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        # second layer
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                padding="same")(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

def get_model(input_img, n_filters=16, dropout=0.5, batchnorm=True):
        # contracting path
        c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
        p1 = tf.keras.layers.MaxPooling2D((2, 2)) (c1)
        p1 = tf.keras.layers.Dropout(dropout*0.5)(p1)

        c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
        p2 = tf.keras.layers.MaxPooling2D((2, 2)) (c2)
        p2 = tf.keras.layers.Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
        p3 = tf.keras.layers.MaxPooling2D((2, 2)) (c3)
        p3 = tf.keras.layers.Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)
        p4 = tf.keras.layers.Dropout(dropout)(p4)
        
        c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
        
        # expansive path
        u6 = tf.keras.layers.Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        u6 = tf.keras.layers.Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

        u7 = tf.keras.layers.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        u7 = tf.keras.layers.Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

        u8 = tf.keras.layers.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        u8 = tf.keras.layers.Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

        u9 = tf.keras.layers.Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        u9 = tf.keras.layers.Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
        
        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
        model = tf.keras.models.Model(inputs=[input_img], outputs=[outputs])
        return model

class YSC:
    def __init__(self, model_file, input_path, output_path, valid_path=None, size_x=64, size_y=64):
        self.global_filename = 0
        self.validation_filename = []
        self.global_foldername = 0
        self.divide_color = 255.0
        self.model_file = model_file

        self.x = np.empty(shape=(0,size_x,size_y,1))
        self.y = np.empty(shape=(0,size_x,size_y,1))
        self.valid = np.empty(shape=(0,size_x,size_y,1))

        self.size_x = size_x
        self.size_y = size_y

        if not os.path.exists("./save/"):
            os.makedirs("./save/")
        if not os.path.exists("./generation/"):
            os.makedirs("./generation/")
        
        try:
            self.model = tf.keras.models.load_model('./save/'+str(self.model_file))
        except:
            h5py.File('./save/'+str(model_file))
            inputs = tf.keras.Input(shape=(size_x,size_y,1))
            self.model = get_model(inputs)

        for filename in glob.glob(str(input_path)+"/*.png"):
            image = Image.open(filename).resize(size=(self.size_x,self.size_y))
            image_array = np.average(np.asarray(image), axis=2).reshape(self.size_x,self.size_y,1)/self.divide_color
            self.x = np.append(self.x, [image_array], axis=0)

        for filename in glob.glob(str(output_path)+"/*.png"):
            image = Image.open(filename).resize(size=(self.size_x,self.size_y))
            image_array = np.average(np.asarray(image), axis=2).reshape(self.size_x,self.size_y,1)/self.divide_color
            self.y = np.append(self.y, [image_array], axis=0)

        if valid_path == None:
            print("Doesn't exist validation")
            self.hasvalid = False
        else:
            self.hasvalid = True
            for filename in glob.glob(str(valid_path)+"/*.png"):
                self.validation_filename.append(filename[len(str(valid_path)):])
                image = Image.open(filename).resize(size=(self.size_x,self.size_x))
                image_array = np.average(np.asarray(image), axis=2).reshape(self.size_x,self.size_y,1)/self.divide_color
                self.valid = np.append(self.valid, [image_array], axis=0)
    
    def generate_image(self, image):
        image = np.asarray(image)

        if not os.path.exists("./generation/"+str(self.global_foldername)+"/"):
            os.makedirs("./generation/"+str(self.global_foldername)+"/")
            
        if len(np.shape(image)) == 4:
            predicts = self.model.predict(image).reshape(-1,self.size_x,self.size_y)*self.divide_color
            for i in predicts:
                result = Image.fromarray(i.astype(np.uint8)).resize(size=(500,500))
                result.save("./generation/"+str(self.global_foldername)+"/"+self.validation_filename[self.global_filename])
                self.global_filename += 1
        elif len(np.shape(image)) == 3:
            predicts = self.model.predict(image.reshape(1,self.size_x,self.size_y,1)).reshape(-1,self.size_x,self.size_y)*self.divide_color
            result = Image.fromarray(predicts[0].astype(np.uint8)).resize(size=(500,500))
            result.save("./generation/"+str(self.global_foldername)+"/"+self.validation_filename[self.global_filename])
            self.global_filename += 1
        else:
            print("Error Dimension")
            return
        self.global_filename = 0
        self.global_foldername += 1
    
    def run(self, times=100, batch_size=10, epochs=200):
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        if self.hasvalid:
            self.generate_image(self.valid)
        for i in range(times):
            
            self.model.fit(x=self.x, y=self.y, batch_size=batch_size, epochs=epochs)
            if self.hasvalid:           
                self.generate_image(self.valid)

            self.model.save('./save/'+str(self.model_file))