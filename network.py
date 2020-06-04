# coding: utf-8
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras import optimizers

#手势文件夹
GESTURE_FOLDER = "geskind"
#手势模型和手势标签文件名
MODEL_NAME = "ss_model.h5"
LABEL_NAME = "ss_labels.dat"


# 初始化
data = []
labels = []
class_num = 0
train_num = 20

#创建存放数据集的手势文件夹
try:
    os.makedirs(GESTURE_FOLDER)
except OSError as e:
    print(GESTURE_FOLDER+'文件夹已创建')



print('请将手势图片种类放在目录下的geskind文件夹中')
class_num = int(input('请输入准确的手势种类数：'))
train_num = int(input('请输入训练训练次数：'))
    
#遍历手势图片
for image_file in paths.list_images(GESTURE_FOLDER):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100,100))
    #增加维度
    image = np.expand_dims(image, axis=2)
    #获取手势图片对应名称
    label = image_file.split(os.path.sep)[-2]
    #将标签和数据添加进data和label
    data.append(image)
    labels.append(label)
print('数据标签加载完成')

#归一化
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

#测试集，数据集分离
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.2, random_state=0)

# 标签二值化
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)


#标签数据文件保存
with open(LABEL_NAME, "wb") as f:
    pickle.dump(lb, f)
print('生成dat文件，开始构建神经网络')

#神经网络构建
model = Sequential()
# 第一层卷积池化
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(100,100, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 第二层卷积池化
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 第三层卷积池化
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 全连接层
model.add(Flatten())
model.add(Dense(64, activation="relu"))

#防止过拟合
model.add(Dropout(0.5))

#输出层(我这里用的是多分类的对数损失函数softmax，如果你只有两个手势，用二分类sigmoid
model.add(Dense(class_num, activation="softmax"))

#建立优化器(我这里loss用的是categorical_crossentropy，如果你知有两个手势，loss用binary_crossentropy
model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=0.0001), metrics=["accuracy"])

print('构建成功，开始训练')
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), batch_size=32, epochs=train_num, verbose=1)

#保存模型
model.save(MODEL_NAME)

print('训练保存完成')
