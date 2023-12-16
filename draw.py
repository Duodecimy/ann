import numpy as np
import matplotlib.pyplot as plt

# 将训练集中的一行，也就是一个手写体数字的数据，可视化
def draw(filepath):
  data_file = open(filepath, "r")
  data_list = data_file.readlines()
  data_file.close()

  # data_line = data_list[1].split(',')
  # img_array = np.asfarray(data_line[1:]).reshape(28, 28)
  # plt.imshow(img_array, cmap='Greys', interpolation='None')
  # plt.show()
  
  # 将训练集中的一百张图片都绘制出来
  plt.figure(num='手写体数字', figsize=(10, 10))  # 创建一个名为astronaut的窗口,并设置大小 
  for i in range(100):
    data_line = data_list[i].split(',')
    img_array = np.asfarray(data_line[1:]).reshape(28, 28)
    plt.subplot(10, 10, i+1)
    plt.imshow(img_array, cmap='Greys', interpolation='None')
  plt.show()

  pass

if __name__ == "__main__":
  filepath = "./mnist_dataset/mnist_train_100.csv"
  draw(filepath)
  