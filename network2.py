import numpy as np


# 二次代价函数
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """
        返回网络输出向量与预期的差异,这里返回二次代价
        （代价不会再训练时陪计算，但会在监控网络性能是被计算）
        :param a: 网络输出向量 len(a)==10
        :param y: 输出预期 0-9 中的整数
        :return: 预期与输入的差异，非负
        """
        _y = y
        y = [0 for x in range(10)]
        y[_y] = 1
        return 0.5 * np.sum([(ai - yi) ** 2 for ai, yi in zip(a, y)])

    @staticmethod
    def delta(z, a, y):
        """
        根据反向传播基本方程(1) 给出输出层误差
        """
        _y = y
        y = [0 for x in range(10)]
        y[_y] = 1
        return (a - y) * sigmoid_prime(z)


# 交叉熵代价函数
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        返回网络输出向量与预期的差异,这里返回交叉熵代价
        （代价不会再训练时陪计算，但会在监控网络性能是被计算）
        :param a: 网络输出向量 len(a)==10
        :param y: 输出预期 0-9 中的整数
        :return: 预期与输入的差异，非负
        """
        # 预期值的向量化
        _y = y
        y = [0 for x in range(10)]
        y[_y] = 1

        _sum = 0.0
        for ai, yi in zip(a, y):
            # 异常值处理，当预期与输出不符时，输出0、1，则预期必为1、0，此时代价无穷大，故返回最大数
            if ai != yi and (ai == 0 or ai == 1):
                return 1.7976931348623157e+308
            # 输出与预期相符合时的异常值 0，1 处理
            if yi == 0:
                _sum += - np.log(1 - ai)
            elif yi == 1:
                _sum += - np.log(ai)
            else:
                # 预期值必为 0 或 1 ，因此这里的代码永远不会执行，写在这里为表达计算逻辑
                _sum += -yi * np.log(ai) - (1 - yi) * np.log(1 - ai)
        # 由于a可以取0，1 ，对数运算会产生 -inf 这里调用 np.nan_to_num 将其替换成可计算的数值
        return _sum  # np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        根据反向传播基本方程(1) 给出输出层误差
        交叉熵代价的输出层误差与 sigma_prime(z)无关，故z 参数不会使用
        但未保持统一，仍然在形参列表中保留该参数
        """
        _y = y
        y = [0 for x in range(10)]
        y[_y] = 1
        return a - y


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 权重初始化改进为 均值为0 ，标准差为1/(sqrt(n))的随机变量
        """
        原因在于：标准差为1 的初始化使得 z = wa + b 变得很大或很小，这意味着神经元的饱和，
        因此这里按照输入规模缩小标准差，z 将是一个接近 0 的值，即不饱和 
        """
        self.biases = [np.random.randn(y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost = cost  # 表示代价函数，用对象替代函数计算代价，及输出层误差

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            stop_delay=-1):
        """
        训练逻辑与之前基本一样，但增加了L2规范化，故增加了参数 lmbda
        增加了网络性能监控的逻辑
        :param training_data:
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :param lmbda: 非负，规范化参数
        :param evaluation_data:
        :param monitor_evaluation_cost:
        :param monitor_evaluation_accuracy:
        :param monitor_training_cost:
        :param monitor_training_accuracy:
        :param stop_delay: 当给出一个大于零的整数是，将会自动停止，延迟周期为该数值
        :return:
        """
        if evaluation_data != None:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            # ##########################下面是性能监控###################################
            print("周期 {} 训练完成".format(j + 1))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("训练数据集的代价: {}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                print("数据集正确数: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print("估算数据集的代价: {}".format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("估算数据集的正确数: {} / {}".format(
                    accuracy, n_data))
            print("------------------------------------------")
            # 提前终止
            if stop_delay > 0 and monitor_evaluation_accuracy and monitor_evaluation_cost:
                if len(evaluation_cost) >= stop_delay:
                    _d = 1
                    _i = 1
                    while _d > 0 and _i <= stop_delay:
                        _d = evaluation_cost[-_i] - evaluation_cost[-_i - 1]
                        _i += 1
                    if _d > 0:
                        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        使用规范化的小批量学习
        :param mini_batch:
        :param eta:
        :param lmbda: 规范化参数
        :param n:训练数据总数， 规范化改变了权重更新规则，需要该参数
        :return: None
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 使用规范化改变了权重更新规则
        decay = 1 - eta * lmbda / n  # 权重衰减系数
        self.weights = [decay * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 这里模拟一个前向传播的过程，将每一层的输出（未压缩）保存在zs中，压缩后的激活值（作为下一层的输入）保存在activations
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 反向传播核心算法
        delta = self.cost.delta(zs[-1], activations[-1], y)  # 输出层误差计算
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta.reshape(10, 1), activations[-2].reshape(1, len(activations[-2])))
        # 输出层梯度计算完毕，这里从倒数第二层反向计算所有层梯度（不包含输入层）
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)  # -l 层误差计算使用方程 (2) 由下一层表示当前层
            # 原理与计算输出层相同
            nabla_b[-l] = delta
            # 方程 (4)
            nabla_w[-l] = np.dot(delta.reshape(len(delta), 1), activations[-l - 1].reshape(1, len(activations[-l - 1])))
        return nabla_b, nabla_w

    def accuracy(self, data):
        """
        数据集正确率
        :param data: 数据集
        :return:数据集正确数
        """
        results = [(np.argmax(self.feedforward(x)), y)
                   for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda):
        """
        返回对数据集data的代价
        :param data:数据集
        :param lmbda:规范化参数
        :return: 非负数，代价
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            cost += self.cost.fn(a, y)
        cost /= len(data)
        # np.linalg.norm(w) 计算矩阵 w 的Frobenius范数：各个元素平方和的平方根
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)  # 规范化项
        return cost

    def save(self, file_name: str):
        """
        保存当前网络参数数据：以（权重向量，偏置）的二维数组方式保存
        :param file_name: 保存的文件名，不含后缀
        :return: None
        """
        data = []
        for b, w in zip(self.biases, self.weights):
            layer = []
            for bj, wj in zip(b, w):
                layer.append((wj, bj))
            data.append(layer)
        data = np.array(data)
        np.save(file_name, data)

    def load(self, file_name: str):
        """
        从文件中读取数据并初始化当前网络，改变网络结构并丢弃所有参数
        :param file_name: save()保存的文件不包含后缀
        :return:None
        """
        data = np.load(file_name + '.npy', allow_pickle=True)
        self.num_layers = len(data) + 1
        self.sizes = [len(l) for l in data]
        self.biases = []
        self.weights = []
        for layer in data:
            vb = []
            vw = []
            for w, b in layer:
                vb.append(b)
                vw.append(w)
            self.biases.append(vb)
            self.weights.append(vw)



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


if __name__ == '__main__':
    # 加载数据
    training_data = np.load('training_data.npy', allow_pickle=True)
    test_data = np.load('test_data.npy', allow_pickle=True)
    validation_data = np.load('validation_data.npy', allow_pickle=True)

    net = Network([784, 10])
    epo = 20
    eta = 0.02
    lmbda = 2

    # 加载训练数据训练
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, epo, 10, eta,
                                                                                     lmbda=lmbda,
                                                                                     evaluation_data=validation_data,
                                                                                     monitor_evaluation_accuracy=True,
                                                                                     monitor_evaluation_cost=True,
                                                                                     monitor_training_accuracy=True,
                                                                                     monitor_training_cost=True,
                                                                                     # auto_stop=True,
                                                                                     # stop_delay=4
                                                                                     )
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

    x = [x + 1 for x in range(epo)]

    fig = plt.figure(figsize=(8, 6), dpi=120)
    fig.add_subplot(221)
    plt.title('代价')
    plt.plot(x, training_cost)
    plt.plot(x, evaluation_cost)
    plt.legend(['训练数据集', '估测数据集'])

    fig.add_subplot(222)
    plt.title('正确率')
    plt.plot(x, [x / len(training_data) for x in training_accuracy])
    plt.plot(x, [x / len(validation_data) for x in evaluation_accuracy])
    plt.legend(['训练数据集', '估测数据集'])

    fig.add_subplot(223)
    plt.title('代价差：训练-估测')
    plt.plot(x, [t - v for t, v in zip(training_cost, evaluation_cost)])
    plt.legend(['训练 - 估测'])

    fig.add_subplot(224)
    plt.title('准确率差：训练 - 估测')
    plt.plot(x, [t / len(training_data) - v / len(validation_data) for t, v in
                 zip(training_accuracy, evaluation_accuracy)])
    plt.legend(['训练-估测'])
    # axs[0].plot(training_cost, label='training cost')
    # axs[0].plot(evaluation_cost, label='evaluation cost')
    #
    # axs[1].plot([x / 50000 for x in training_accuracy], label='training accuracy')
    # axs[1].plot([x / 10000 for x in evaluation_accuracy], label='evaluation accuracy')
    #
    # plt.title('学习速率：{}，规范化参数：{}'.format(eta, lmbda))
    plt.show()
