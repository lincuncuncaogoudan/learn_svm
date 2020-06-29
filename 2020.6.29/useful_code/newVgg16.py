#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/11 10:17
# @Author  : hhy
# @FileName: newVgg16.py
# @Software: PyCharm


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from useful_code import utils_paths
import numpy as np
import random
import cv2
import os
from keras import backend as K
from numpy import linalg as LA
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D


class MyModel:


    def __init__(self,train=False):
        self.num_classes = 2
        self.weight_decay = 0.0005
        self.x_shape=(224,224,3)
        self.INIT_LR = 0.001
        self.Epoch=100
        self.model = self.build_model()
        #self.data_path="data/train/data"
        self.data_path="/home/hhy1/searchGraph/star/"
        # self.data_path= "/newFruit/"
        self.lb = LabelBinarizer()
        self.weight=""
        if(train):
            print("data_path--->"+self.data_path)
            self.model.summary()
            self.trainX,self.testX=self.load_data(self.data_path)
        else:

            pass



        # if train:
        #     self.train(self.model)
        # else:
        #     self.model.load_weights(self.weight)


    def load_data(self,data_path):
        data = []
        labels = []

        #datapath="'./data/train/data'"
        imagePaths = sorted(list(utils_paths.list_images(data_path)))
        # random.seed(42)
        random.shuffle(imagePaths)

        print("---->")

        for i,imagePath in enumerate(imagePaths):

            image = cv2.imread(imagePath)
            image = cv2.resize(image, (224, 224))
            # img = image.load_img(imagePath, target_size=(56,56))
            # img = image.img_to_array(img)
            # img = np.expand_dims(img, axis=0)
            # img = preprocess_input(img)
            data.append(image)

            label = imagePath.split(os.path.sep)[-2]
            # print(label)
            if label=="other":
                label=0
            else:
                label=1
            labels.append(label)

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)


        # (trainX, testX, trainY, testY) = train_test_split(data,
        #                                                   labels, test_size=0.25, random_state=42)
        #
        # #lb = LabelBinarizer()
        #
        # trainY = self.lb.fit_transform(trainY)
        # testY = self.lb.transform(testY)



        return data,labels

    def build_model(self,name="vgg16"):


        if name=="vgg16":
            model = Sequential(name="vgg16-sequential")

            model.add(Conv2D(64,(3,3),padding='same',activation="relu",input_shape=self.x_shape,
                             name="block1_conv1"))
            model.add(Conv2D(64,(3,3),padding='same',activation="relu",name="block1_conv2"))
            model.add(MaxPooling2D((2,2),strides=2,name="block_pool1"))

            model.add(Conv2D(128,(3,3),padding='same',activation='relu',name="block2_conv1"))
            model.add(Conv2D(128,(3,3),padding='same',activation='relu',name='block2_conv2'))
            model.add(MaxPooling2D((2,2),strides=(2,2),name="block2_pool"))

            model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name="block3_conv1"))
            model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))
            model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))
            model.add(MaxPooling2D((2, 2), strides=2, name="block3_pool"))

            model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name="block4_conv1"))
            model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2'))
            model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3'))
            model.add(MaxPooling2D((2, 2), strides=2, name="block4_pool"))

            model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name="block5_conv1"))
            model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2'))
            model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3'))
            model.add(MaxPooling2D((2, 2), strides=2, name="block5_pool"))

            ## load imagenet weight
            model.load_weights("/home/hhy1/searchGraph/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

            model.add(Flatten(name='flatten'))
            model.add(Dense(4096, activation='relu', name='fc1'))
            model.add(Dense(4096, activation='relu', name='fc2'))

            # layer111=model.get_layer("fc2")
            # print("-----",layer111.output_shape)

            model.add(Dense(self.num_classes, activation='softmax', name='dense_3'))



            return model

    def train(self, model):

        freeze_layer= 13
        for i in range(freeze_layer):model.layers[i].trainable=False

        freeze_Epoch=25

        opt = SGD(lr=self.INIT_LR)

        if True:
            # model.compile(loss="categorical_crossentropy", optimizer=opt,
            #    metrics=["accuracy"])
            model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                          metrics=["accuracy"])

            print(self.testX.shape)
            model.fit(self.trainX, self.testX, validation_split=0.2,
                          epochs=freeze_Epoch, batch_size=10)

            # model.fit(self.trainX, self.textX, validation_data=(self.trainY,self.testY),
            #           epochs=freeze_Epoch, batch_size=32)

        for i in range(freeze_layer):model.layers[i].trainable=True

        Epoch=self.Epoch-freeze_Epoch


        model.compile(loss="sparse_categorical_crossentropy",optimizer=opt,
                      metrics=["accuracy"])
        model.fit(self.trainX, self.testX,validation_split=0.2,
                  epochs=Epoch, batch_size=10)





        # H = model.fit(self.trainX, self.trainY, validation_data=(self.testY, self.testY),
        #     epochs=self.EPOCHS, batch_size=32)
        # H = model.fit(self.trainX, self.trainY,
        #               epochs=self.EPOCHS, batch_size=32)

        model.save_weights('vgg_weight_voc.h5')

        model.save("my_model_voc.h5")
        # model.save('./output/simple_nn.model')
        # f = open('./output/simple_nn_lb.pickle', "wb")  #
        # f.write(pickle.dumps(self.lb))
        # f.close()
        return model

    def predict(self):
        print("---------")
        predictions = self.model.predict(self.testX, batch_size=32)
        print(classification_report(self.testY.argmax(axis=1),
            predictions.argmax(axis=1), target_names=self.lb.classes_))

    def predict_signal(self,img):
        # signal iamge result
        self.model.load_weights("vgg_weight_voc.h5")
        print("-------signal image result-----")
        prediction_signal=self.model.predict(img,batch_size=1)
        prediction_signal=np.argmax(prediction_signal)
        print(prediction_signal)
        return prediction_signal


    def get_layer_output(self,model, x, index=-1):
        """
        get the computing result output of any layer you want, default the last layer.
        :param model: primary model
        :param x: input of primary model( x of model.predict([x])[0])
        :param index: index of target layer, i.e., layer[23]
        :return: result
        """
        layer = K.function([model.input], [model.layers[index].output])
        return layer([x])[0]

    def predict_img(self,img):

        self.model.load_weights("vgg_weight_voc.h5")
        layer_name = "fc2"
        mid_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        mid_output=mid_layer_model.predict(img)
        print(mid_output.shape)
        return  mid_output


    def extract_feat(self, img_path: object) -> object:
        # img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        # img = image.img_to_array(img)
        # img = np.expand_dims(img, axis=0)
        # img = preprocess_input(img)
        # feat = self.model.predict(img)
        # norm_feat = feat[0]/LA.norm(feat[0])
        # return norm_feat

        image=cv2.imread(img_path)
        print("---------------------->",img_path)
        image=cv2.resize(image,(224,224))

        #scale
        image=image.astype("float")/255.0
        img = np.expand_dims(image, axis=0)
        feat = self.predict_img(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

if __name__ == "__main__":
    vggModel = MyModel(train=True)
    vggModel.train(vggModel.model)
    # vggModel.predict()





# N = np.arange(0, EPOCHS)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N[1500:], H.history["accuracy"][1500:], label="train_acc")
# plt.plot(N[1500:], H.history["val_accuracy"][1500:], label="val_acc")
# plt.title("Training and Validation Accuracy (Simple NN)")
# plt.xlabel("Epoch #")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig('./output/simple_nn_plot_acc.png')
#
# plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.title("Training and Validation Loss (Simple NN)")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig('./output/simple_nn_plot_loss.png')





