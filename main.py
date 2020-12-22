import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

train_path = 'mnist_train_s/'
train_file = 'mnist_train_s.txt'
test_path = 'mnist_test_s/'
test_file = 'number_test.txt'

# 设置随机数种子
np.random.seed(42)

# 加载数据集
X_train = []
Y_train = []
X_test = []

with open(train_file, 'r') as f:
    for line in f:
        img_name, label = line.split()

        img = Image.open(train_path + img_name)
        x = np.array(img)
        X_train.append(x)

        y = np.zeros((10,))
        y[int(label)] = 1
        Y_train.append(y)

test_names = os.listdir(test_path)
for name in test_names:
    img = Image.open(test_path + name)
    x = np.array(img)
    X_test.append(x)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

# 数据预处理
X_train = X_train.astype('float32')
X_train /= 255
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

X_test = X_test.astype('float32')
X_test /= 255
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# 构建模型
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(1, 28, 28)))   # 卷积层
model.add(MaxPooling2D(pool_size=(2, 2)))                                   # 池化层
model.add(Dropout(0.2))                                                     # Dropout层
model.add(Flatten())                                                        # Flatten层
model.add(Dense(128, activation='relu'))                                    # 全连接层
model.add(Dense(10, activation='softmax'))                                  # 全连接层，输出

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, Y_train, batch_size=32, epochs=8, verbose=1)

# 保存模型
model.save('model.h5')

# 输出结果
result = model.predict_classes(X_test)
with open(test_file, 'w') as f:
    for i in range(len(test_names)):
        line = test_names[i] + ' ' + str(result[i]) + '\n'
        f.write(line)
