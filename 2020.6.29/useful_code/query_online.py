#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/9 19:58
# @Author  : hhy
# @FileName: query_online.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
import shutil

import  cv2
import os

import numpy as np
import h5py

import argparse

## 查询的路径为database里的图，index索引的特征 result database
from useful_code.newVgg16 import MyModel

ap = argparse.ArgumentParser()
ap.add_argument("-query", required=False,default="query",
                help="Path to query which contains image to be queried")
ap.add_argument("-index", required=False,default="featureCNN_VOC.h5",
                help="Path to index")
ap.add_argument("-result", required=False,default="StarDatabase",
                help="Path for output retrieved images")
args = vars(ap.parse_args())

# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"], 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

queryDataset=[]
classes=[]
# read and show query image
for i,im in enumerate(os.listdir(args["query"])):
    queryImage=np.uint8(cv2.imread(os.path.join(args["query"], im)))
    x_shape=queryImage.shape
    img2=cv2.cvtColor(queryImage,cv2.COLOR_BGR2RGB)
    queryDataset.append(queryImage)
    # plt.title("Query Image"+str(i))
    # plt.imshow(img2)

##-------------------ALL picture Mean
# start=np.zeros(x_shape,dtype=np.uint16)
# for i,im in enumerate(queryDataset):
#    start=cv2.add(start,im)
# start=np.uint8(start/(i+1))
# cv2.imwrite("query.png",start)

# queryImg = mpimg.imread(queryDir)
# plt.title("Query Image")
# plt.imshow(queryImg)
# plt.show()


# init VGGNet16 model
model = MyModel()
# get All class
for i in queryDataset:
    image = cv2.resize(i, (224, 224))
    # scale图像数据
    image = image.astype("float") / 255.0
    img = np.expand_dims(image, axis=0)
    classes.append(model.predict_signal(img))

## summary each temp count
dict={}
for key in classes:
    dict[key]=dict.get(key,0)+1
temp=0
max_classes=classes[np.argmax(dict.values())]


###-------------maxClass Picture Mean------------------------
# start=np.zeros(x_shape,dtype=np.uint16)
# for i ,j in enumerate(classes):
#     if j==max_classes:
#         start =cv2.add(start,np.uint16(queryDataset[i]))
#
# start=np.uint8(start/(i+1))
# cv2.imwrite("query.png",start)

###extract query's maxClass image's feature, compute simlarity score and sort
featureDateset=[]
index_number=imgNames.tolist()
index_name=[str(c,encoding='utf-8') for c in index_number]
'''------------------------------------'''

for i ,j in enumerate(classes):
    if j==max_classes:
        pictureName=os.listdir("/home/hhy1/searchGraph/query")
        x=pictureName[i]
        index=index_name.index(x)
        feats[index]=0
        feature=model.extract_feat(os.path.join("/home/hhy1/searchGraph/StarDatabase", x))
        featureDateset.append(feature)

### feature h5
output="featureCNN_VOC.h5"
# copy_output=shutil.copyfile("featureCNN_VOC.h5","copyfeature.h5")
h5f = h5py.File(output, 'w')
h5f.create_dataset('dataset_1', data=feats)
h5f.create_dataset('dataset_2', data=imgNames)
h5f.close()

start=np.zeros(([4096,]),np.float)
for count,j in enumerate(featureDateset):
    start+=j
start/=(i+1)
# np.savetxt("feature.txt",start)
scores=np.dot(start,feats.T)
rank_ID=np.argsort(scores)[::-1]
rank_score=scores[rank_ID]







# extract query image's feature, compute simlarity score and sort
# queryVec = model.extract_feat("query.png")
# # queryVec = model.extract_feat(queryImage)
# scores = np.dot(queryVec, feats.T)
# rank_ID = np.argsort(scores)[::-1]
# rank_score = scores[rank_ID]
# print rank_ID
# print rank_score



# number of top retrieved images to show
maxres = 60
# maxres=random.randint(4,6)
#---------from class select-----
# pathDir=os.listdir("newFruit/"+str(max_classes))
# imlist=random.sample(pathDir,maxres)
imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " % maxres, imlist)

### random sample 95 picture from score
# randompos=[]
# for i in range(95):
#     randompos.append(random.randint(50,3000))
# for i in (randompos):
#     sample_list=[imgNames[index] for i ,index in enumerate(rank_ID[randompos])]
#
# sample_list+=imlist
#
# ###sample 100 picture
# sample=[]
# for i in sample_list:
#     sample.append(str(i,encoding="utf-8"))



# show top #maxres retrieved result one by one
for i, im in enumerate(imlist):
    #image = mpimg.imread(args["result"] + "/" + str(im, encoding='utf-8'))
    img = cv2.imread(args["result"] + "/" + str(im, encoding='utf-8'))
    ### deal not b'
    #img = cv2.imread(args["result"] + "/" + str(im))
    #cv2.imwrite(os.path.join("sava",str(im, encoding='utf-8')), img)

    save_path= "/home/hhy1/searchGraph/get_search_result"

    if i==0:
        if os.path.isdir(save_path):
            if len(os.listdir((save_path)))!=0:
                shutil.rmtree(save_path)
                os.makedirs(save_path)
            pass
        else:
            os.makedirs(save_path)
    # print(args["result"] + "/" + str(im, encoding='utf-8'),"----->",os.path.exists(args["result"] + "/" + str(im, encoding='utf-8')))
    cv2.imwrite(save_path + "/" + str(im,encoding='utf-8'),img)
    img=cv2.imread(save_path + "/" + str(im,encoding='utf-8'))
    ###---------deal not b'
    #cv2.imwrite(save_path + "/" + str(im), img)
    # plt.title("search output %d" % (i + 1))
    # plt.imshow(img)
    # plt.show()
    ###---------deal not b'
    # img = cv2.imread(save_path + "/" + str(im))

    ## cvshow
    # cv2.namedWindow("Search Out")
    # cv2.resizeWindow("Search Out",640,480)
    # cv2.moveWindow("Search Out",800,100)
    # cv2.imshow("Search Out",img)
    # cv2.waitKey(2000)

# random sample
# pathdir=os.listdir("dataset_rsvp/other")
# sample=random.sample(pathdir,95)
# if os.path.exists("pictureSeq"):
#     shutil.rmtree("pictureSeq")
# else:
#     os.makedirs("pictureSeq")


# data=[]
#
# for name in sample:
#     data.append(name)
#     # shutil.move("DataSet/1/"+name,"pictureSeq/"+name)
#
# result=os.listdir("get_search_result")
#
#
#     # shutil.move("DataSet/1/"+name,"pictureSeq/"+name)
#
# random.shuffle(data)
# count=np.zeros(40,dtype=np.int)
# count[1]=1
# count[2]=1
#
#
# insert_pos=[random.randint(6,100) for _ in range(len(result))]
#
# for i in range(len(result)):
#     flag=0
#     while True:
#         bullet=int(insert_pos[i]/3)
#         if bullet==0 or bullet==1:
#             pass
#         elif count[bullet+2]==0 and count[bullet-2]==0 and count[bullet+1]==0 and count[bullet-1]==0:
#             count[bullet]+=1
#             break
#         else:
#             insert_pos[i]+=6
#     data.insert(insert_pos[i],result[i])
'''
look at this xqq!!!!
data is a shuffle_data size is 100 ,it's a order list what you need ,you can directly read it.
'''







