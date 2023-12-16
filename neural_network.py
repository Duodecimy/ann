import numpy as np
# 引入含有所需激活函数的库
import scipy.special as special

# 定义神经网络类
class NeuralNetwork:

  # 1. 初始化网络
  def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

    # 设置输入层、隐藏层和输出层的节点数量
    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes

    # 设置学习率
    self.lr = learningrate

    # 设置链接权重矩阵
    self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.inodes, self.hnodes))
    self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.hnodes, self.onodes))

    # 设置激活函数
    self.activation_function = lambda x: special.expit(x)
    pass

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
    self.wih = self.lr * np.dat((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    pass

  def query(self, inputs_list):
    # 将输入列表转化为二维数组
    inputs = np.array(inputs_list, ndmin=2).T

    hidden_inputs = np.dot(self.wih, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    final_inputs = np.dot(self.who, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)
    return final_outputs

if __name__ == "__main__":
  print("Hello, world!")
  input_nodes = 3
  hidden_nodes = 3
  output_nodes = 3
  learning_rate = 0.3
  nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
  print(nn.query([1.0, 0.5, -1.5]))
  pass
