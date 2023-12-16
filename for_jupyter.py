import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# 引入含有所需激活函数的库
import scipy.special as special

# 定义神经网络类
class neuralNetwork:

  # 初始化网络
  def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

    # 设置输入层、隐藏层和输出层的节点数量
    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes

    # 设置学习率
    self.lr = learningrate

    # 设置链接权重矩阵
    self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
    self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    # 设置激活函数
    self.activation_function = lambda x: special.expit(x)

    pass

  # 训练神经网络
  def train(self, inputs_list, targets_list):
    # 将输入列表转化为二维数组
    inputs = np.array(inputs_list, ndmin=2).T
    targets = np.array(targets_list, ndmin=2).T

    # 训练神经网络分为以下两步：
    # 1. 针对训练样本给出输出
    hidden_inputs = np.dot(self.wih, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs = np.dot(self.who, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)

    # 2. 使用差值更新权重网格
    # 输出层的误差是期望值-输出值
    output_errors = targets - final_outputs
    # 隐藏层的误差是输出层误差依链接权重的点积
    hidden_errors = np.dot(self.who.T, output_errors)

    # 更新隐藏层和输出层之间的链接权重
    self.who = self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
    # 更新输入层和隐藏层之间的链接权重
    self.wih = self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    pass

  def query(self, inputs_list):
    # 将输入列表转化为二维数组
    inputs = np.array(inputs_list, ndmin=2).T

    hidden_inputs = np.dot(self.wih, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs = np.dot(self.who, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)
    return final_outputs



  # 从文件中提取数据
filepath = "./mnist_dataset/mnist_train_100.csv"
training_data_file = open(filepath, "r")
training_data_lines = training_data_file.readlines()
training_data_file.close()

# 3. 配置神经网络
input_nodes = 28 * 28   # 手写数字灰度图的像素数
hidden_nodes = 100
output_nodes = 10       # 对应10个手写数字
learning_rate = 0.3
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 4. 开始训练神经网络
for record in training_data_lines:
  # 获得一行数据，也就是一个手写数字
  training_data_line = record.split(',')

  # 1. 准备用于训练和查询的输入数据：将输入数据规范到0.01到1.00之间（为了使输入和输出的“形状”匹配，参考S函数）
  scaled_inputs = (np.asfarray(training_data_line[1:]) / 255.0 * 0.99) + 0.01

  # 2. 准备用于训练的输出数据：输出应该匹配激活函数可以输出值的范围
  targets = np.zeros(output_nodes) + 0.01
  targets[int(training_data_line[0])] = 0.99   # 标签代表的数字输出值应该为0.99
  nn.train(scaled_inputs, targets)

# 测试一下神经网络的训练程度
data_file = open("./mnist_dataset/mnist_test_10.csv", "r")
data_list = data_file.readlines()
data_file.close()

data_line = data_list[0].split(',')
result = nn.query(np.asfarray(data_line[1:]) / 255.0 * 0.99) + 0.01
print(result)
print(data_line[0])
pass
  

