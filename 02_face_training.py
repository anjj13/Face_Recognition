''''
    训练数据集里的图片
	训练模型会放到trainer文件夹，如果没有这文件夹，请先创建
    参考代码: https://github.com/thecodacus/Face-Recognition
    Author:An
    E-mail:2562490983@qq.com
'''

import cv2
import numpy as np
from PIL import Image
import os

# 数据集根文件夹
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create() # 创建一个识别模型对象
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");#级联分类（正面人脸检测）




# 获取数据和标签
def getImagesAndLabels(path):
    folderPaths = [os.path.join(path, f) for f in os.listdir(path)]
    # class_es = len(folderPaths)
    label_conv = {} # 存放每个人对应的标签
    i = 0
    # 从dataset文件夹内读出每个人的数据 并按照文件夹排序 给每个人定义一个标签 从0开始 1,2,3
    for name in os.listdir(path):
        label_conv[name] = i
        i += 1

    '''
    label_conv = {
        'Anys':0,
        'chennj':1,
        'krolle':2,
        'lvyd';3
    }
    '''
    print(label_conv)
    faceSamples = [] # 存放采集到的人脸
    lable = [] # 存放采集到的人脸对应的标签
    # 进入大的文件夹
    for folderPath in folderPaths:
        imagePaths = [os.path.join(folderPath, f) for f in os.listdir(folderPath)] # 获取文件夹内的文件名
        # print(len(imagePaths))
        for imagePath in imagePaths:
            # 遍历每个人的文件夹
            print("image path: ",imagePath)
            PIL_img = Image.open(imagePath).convert('L')  # 转成灰度图
            # print(type(PIL_img))
            # plt.imshow(PIL_img)
            # plt.show()

            # cv2.imshow("read",PIL_img)
            img_numpy = np.array(PIL_img, 'uint8') # 格式转换
            name = os.path.split(imagePath)[-1].split(".")[1] # 根据文件名提取出name
            print("name: ",name)

            id = int(label_conv[name]) # 根据name 设置标签
            # print(id)
            faceSamples.append(img_numpy)
            lable.append(id)

            # 如果之前采集数据时没有处理人脸，执行下面的代码
            # faces = detector.detectMultiScale(img_numpy)
            # for (x, y, w, h) in faces:
            #     # Count+=1
            #     faceSamples.append(img_numpy[y:y + h, x:x + w])
            #     cv2.imshow("test", img_numpy[y:y + h, x:x + w])
            #     cv2.waitKey(10000)
            #     break
            #     label.append(id)

    return faceSamples,lable

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,lable = getImagesAndLabels(path)
# for i in range(len(lable)):
#     cv2.imshow(str(lable[i]),faces[i])
#     cv2.waitKey(100)
recognizer.train(faces, np.array(lable))

# 将模型参数存进 trainer/trainer.yml
Trainer_path = 'trainer'
if not os.path.exists(Trainer_path):
    os.mkdir(Trainer_path)
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(lable))))
