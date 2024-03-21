''''
    利用笔记本电脑的摄像头收集数据
	数据想放在/dataset/name
    程序开始运行后 需要输入人脸姓名（用作标签） 同时训练的时候会按照文件夹排序读取 依次编码为0，1......
    Author:An
    E-mail: 2562490983@qq.com

'''

import cv2
import os
# 设置摄像头参数
cam= cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
# detector的读取
# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# 每个人输入自己的名字
face_name = input('\n enter user name end press <return> ==>  ')
# 路径设置
path = os.path.join('dataset',face_name)
if not os.path.exists(path):#ture执行  false跳过 os.path.exists(path)为判断path是否存在
    os.mkdir(path)#以数字权限创建目录（相对或绝对路径，
    print(path+"create successfully!")

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# 用count来标记采集的数据
count = 0

while(True):
    ret,img = cam.read() # 从摄像头获取数据

    # img = cv2.flip(img, 1) # 将相机拍到的数据和我们的坐标对齐 垂直翻转
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #  灰度处理 LBP特征是用灰度图的

    faces = face_detector.detectMultiScale(gray, 1.3, 5) # 检测当前帧的人脸 返回坐标
    # print(faces)
    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # print("Save the captured image into the datasets folder")#以"/User." + face_name + '.' + str(count) + ".jpg"命名格式写入图像
        cv2.imwrite("dataset/"+face_name+"/User." + face_name + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img) # 实时显示当前帧图像采集情况
        print("count:",count)

    k = cv2.waitKey(100) & 0xff
    if k == 27: # 'ESC' 退出采集
        break
    elif count >= 100: # 采满100张退出当前人的采集
         break

# 清理缓冲池
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()#释放视频读取对象
cv2.destroyAllWindows()#销毁全部窗口清理内存


