import numpy as np

"""
Note:所有激活函数同时接受数值，或多维数组
     所有数组都是 C-style
"""


# 装饰器，装饰单参数函数，使其支持 array_like 输入
def array_like(fun):
    def wrapper(z):
        z = np.array(z)
        if z.size <= 1:
            return fun(z)
        shape = z.shape
        z = list(z.flat)
        out = []
        for i in z:
            out.append(fun(i))
        out = np.array(out).reshape(shape)
        return out

    return wrapper


# z 参数是数值或array_like
@array_like
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@array_like
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


@array_like
def tanh(z):
    return 2 * sigmoid(2 * z) - 1


@array_like
def tanh_prime(z):
    return 2 * sigmoid_prime(z)


# 可能需要更低的学习速率，由于导数较大（1）
@array_like
def ReLU(z):
    if z > 0:
        return z
    else:
        return 0


@array_like
def ReLU_prime(z):
    if z > 0:
        return 1.0
    else:
        return 0.0


# 池化层
class max_pooling(object):
    """
    池化层没有参数，故无需调整，
    没有激活函数，故令激活函数为自身，导数为1
    """

    @staticmethod
    # 池化层前向传播
    def pooling(input_data, size=2, point=None):
        """
        输入是一个二维矩阵,最大值混合
        :param input_data:
        :param size:
        :param point:
        :return:
        """
        if size <= 1:
            return input_data
        input_data = np.array(input_data)
        x, y = input_data.shape
        out = np.zeros((int(x / size), int(y / size)))
        if point is None:
            point = {}
        for i in range(int(x / size)):
            for j in range(int(y / size)):
                max_value = input_data[i * size, j * size]
                pos = (i * size, j * size)
                for _x in range(size):
                    for _y in range(size):
                        if input_data[i * size + _x, j * size + _y] > max_value:
                            max_value = input_data[i * size + _x, j * size + _y]
                            pos = (i * size + _x, j * size + _y)
                out[i, j] = max_value
                point[(i, j)] = pos
        return out

    @staticmethod
    # 池化层的反向传播
    def upsample(input_data, size=2, point=None):
        if size <= 1:
            return input_data
        input_data = np.array(input_data)
        w, h = input_data.shape
        cnnl = np.zeros((w * size, h * size))
        if point is not None:
            for i in range(w):
                for j in range(h):
                    cnnl[point[(i, j)]] = input_data[i, j]
            return cnnl
        for i in range(w):
            for j in range(h):
                cnnl[i * size, j * size] = input_data[i, j]
        return cnnl


class L2_pooling(object):
    @staticmethod
    def pooling(input_data, size=2):
        """
        平方和的平方根混合
        :param input_data:
        :param size:
        :return:
        """
        if size <= 1:
            return input_data
        input_data = np.array(input_data)
        x, y = input_data.shape
        out = np.zeros((int(x / size), int(y / size)))
        for i in range(int(x / size)):
            for j in range(int(y / size)):
                out[i, j] = np.sqrt(np.sum([input_data[i + w, j + h] ** 2 for w in range(size) for h in range(size)]))
        return out


def convolution_2D(a, b, same=False):
    a = np.array(a)
    b = np.array(b)
    if a.shape[0] < b.shape[0] and a.shape[1] < b.shape[1]:
        a, b = b, a
    if same:
        x_b, y_b = b.shape
        a = padding(a, x_b)
    x_a, y_a = a.shape
    x_b, y_b = b.shape
    if x_a > x_b and y_a > y_b:
        out = np.zeros((x_a - x_b + 1, y_a - y_b + 1))
        for i in range(x_a - x_b + 1):
            for j in range(y_a - y_b + 1):
                out[i, j] += np.sum([a[i + w, j + h] * b[w, h] for w in range(x_b) for h in range(y_b)])
        return out


def convolution(a, b, same=False):
    a = np.array(a)
    b = np.array(b)
    if len(a.shape) == 2 and len(b.shape) == 2:
        return convolution_2D(a, b, same=same)
    elif len(a.shape) == 3 and len(b.shape) == 3 and a.shape[0] == b.shape[0]:
        z, x_a, y_a = a.shape
        z, x_b, y_b = b.shape
        if same:
            x, y = x_a, y_a
        else:
            x, y = x_a - x_b + 1, y_a - y_b + 1
        out = np.zeros((x, y))
        for ai, bi in zip(a, b):
            out += convolution_2D(ai, bi, same=same)
        return np.array(out)
    else:
        if len(a.shape) != len(b.shape):
            raise TypeError('卷积类型不匹配 {} != {}'.format(a.shape, b.shape))
        elif len(a.shape) > 3:
            raise TypeError('参数维度过多：维度必须为 2 或 3，当前为{}'.format(len(a.shape)))
        elif len(a.shape) < 2:
            raise TypeError('参数维度过少：维度必须为 2 或 3，当前为{}'.format(len(a.shape)))
        else:
            raise TypeError('卷积核厚度必须与输入保持一致 输入：{} 卷积核{}'.format(a.shape[0], b.shape[0]))


##############
def padding(input_data, core_size):
    input_data = np.array(input_data)
    if len(input_data.shape) != 2:
        raise TypeError('input_data 必须为二维数组！')
    p = core_size - 1
    x, y = input_data.shape
    o = np.zeros((x + p, y + p))
    p = int(p / 2)
    for i in range(x):
        for j in range(y):
            o[p + i, p + j] = input_data[i, j]
    return o


# 输出层代价计算 && 输出层误差计算
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
        # 将y转化为向量
        _y = y
        y = [0] * 10
        y[_y] = 1
        return 0.5 * np.sum([(ai - yi) ** 2 for ai, yi in zip(a, y)])

    @staticmethod
    def delta(z, a, y):
        """
        根据反向传播基本方程(1) 给出输出层误差
        """
        _y = y
        y = [0] * 10
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
        y = [0] * 10
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
        a = np.array(a)
        _y = y
        y = [0] * 10
        y[_y] = 1
        y = np.array(y)
        return a - y


# 定义层类型，每一层可以正向传播&&反向传播，以实现不同层连接

# 全连接层 接受n维数组作为输入，输出向量
# 这里还定义了任何层应当有的方法
"""
Note:
    层对象是对网络中没一层功能的抽象，曾对象接受一个任意维的输入
    并返回任意维的输出，曾对象包含反向传播公式的实现，因而可以方便的计算梯度和改变参数
    层对象包含激活函数及其导数，因而不同层可以有不同激活函数
"""


class Layer(object):
    def __init__(self, input_size, size, activation_function, activation_derivative_function, frozen=False):
        self.input_size = input_size
        self.size = size
        self.sigmoid = activation_function
        self.sigmoid_prime = activation_derivative_function
        self.weights = np.array([np.random.randn(input_size) / np.sqrt(size) for x in range(size)])
        self.biases = np.random.randn(size)
        # 缓存计算结果
        self.input_data = None
        self.z = None
        self.a = None
        # 参数
        self.frozen = frozen  # 是否冻结该层，冻结将会使得参数更改无效化

    def feedforward(self, a, cache=False):
        """
        进行前向传播，输入和计算结果会被缓存
        :param a:输入
        :param cache:是否保存中间数据
        :return:每个神经元的激活值
        """
        a = np.array(a)
        a = list(a.flat)
        if cache:
            self.input_data = a
        if len(a) != self.input_size:
            raise Exception('a.size != self.input_size:{}'.format(self.input_size))
        out = [np.dot(a, x) + b for x, b in zip(self.weights, self.biases)]
        if cache:
            self.z = np.array(out)
        out = [self.sigmoid(x) for x in out]
        if cache:
            self.a = np.array(out)
        return out

    def backprop(self, err):
        """
        根据传入的误差，计算梯度，
        :param err: 该层神经元输出误差
        :return: 该层的样本梯度
        """
        d_b = err[:]
        d_w = [[x * y for y in self.input_data] for x in err]
        return np.array(d_w), np.array(d_b)

    def front_layer_err(self, err, front_layer_z):
        """
        计算前一层的误差
        :param err:本层 误差
        :param front_layer_z:上一层的sigmoid_prime(z)
        :return:上一层误差
        """
        w_e = np.dot(np.array(self.weights).transpose(), np.array(err).reshape(len(err), -1))
        w_e = list(w_e.flat)
        return [a * b for a, b in zip(w_e, front_layer_z)]

    def update(self, delta_w, delta_b, decay=1):
        """
        根据给定的权重、偏置 改变量改变当前层的权重、偏置
        :param delta_w: 权重改变量,应当为 backprop() 返回类型
        :param delta_b: 偏置改变量,应当为 backprop() 返回类型
        :param decay: 权重衰减系数
        :return: None
        """
        if self.frozen:
            return
        self.weights = np.array([[decay * w - nw for w, nw in zip(w, d_w)] for w, d_w in zip(self.weights, delta_w)])
        self.biases = np.array([b - d_b for b, d_b in zip(self.biases, delta_b)])

    def new(self, data: []):
        # TODO
        # 从data中的数据创建Layer,data 应当是有get_data()创建的对象
        # return Layer()
        pass

    def get_data(self):
        # TODO
        # 将层中数据包装为data并返回
        pass


class CNNLayer(object):
    def __init__(self, core_size, core_num, activation_function, activation_derivative_function, core_thickness=1,
                 same=False, pooling=max_pooling, pooling_size=2, frozen=False):
        self.frozen = frozen
        self.sigmoid = activation_function
        self.sigmoid_prime = activation_derivative_function
        self.cores_w = []
        self.cores_b = [np.random.rand() for x in range(core_num)]
        self.core_size = core_size  # 一般为奇数
        self.core_thickness = core_thickness
        self.core_num = core_num
        self.same = same
        self.pooling = pooling
        self.pooling_size = pooling_size
        self.pooling_data = None
        if core_thickness > 1:
            for core in range(core_num):
                self.cores_w.append(
                    [[[np.random.randn() / core_size for y in range(core_size)] for x in range(core_size)] for t in
                     range(core_thickness)])
        else:
            for core in range(core_num):
                self.cores_w.append(
                    [[np.random.randn() / core_size for y in range(core_size)] for x in range(core_size)])
        self.input_data = None
        self.z = None
        self.a = None
        self.convolution_z = None
        self.convolution_a = None

    @staticmethod
    def rot180(input_data):
        input_data = np.array(input_data)
        if len(input_data.shape) == 2:
            out = np.zeros(input_data.shape)
            w, h = input_data.shape
            for i in range(w):
                for j in range(h):
                    out[-i - 1, -j - 1] = input_data[i, j]
            return np.array(out)

    def feedforward(self, a, cache=False):
        a = np.array(a)
        if len(a.shape) == 2:
            x, y = a.shape
        else:
            z, x, y = a.shape
            # 纠正 输入数据的维度
            if z == 1:
                a = a[0]
        if cache:
            self.input_data = a
            self.a = []
            self.z = []
            self.convolution_z = []
            self.convolution_a = []
            self.pooling_data = []
        outs = []
        for core_w, core_b in zip(self.cores_w, self.cores_b):
            # 卷积
            out = convolution(a, core_w, self.same) + np.float(core_b)
            if cache:
                self.convolution_z.append(out)
            # 激活
            out = self.sigmoid(out)
            pooling_data = {}
            if cache:
                self.convolution_a.append(out)
            # 池化
            out = self.pooling.pooling(out, self.pooling_size, point=pooling_data)  # 池化
            if cache:
                self.pooling_data.append(pooling_data)
                self.z.append(out)
                self.a.append(out)
            # 输出
            outs.append(out)
        return outs

    def backprop(self, err):
        err = np.array(err)
        err = err.reshape((self.core_num, int(np.sqrt(err.size / self.core_num)), -1))
        # 池化层反向传播误差
        cnnls = []  # 获得卷积层的误差
        for i in range(len(err)):
            d_a = self.pooling.upsample(err[i], self.pooling_size, self.pooling_data[i])
            # cnnls.append(np.array(d_a) * self.sigmoid_prime(self.convolution_z[i]))
            cnnls.append(d_a)
        # 计算卷积层梯度
        cnn_b = [np.sum(cnnl) for cnnl in cnnls]
        cnn_w = []
        for i in range(self.core_num):
            if self.core_thickness > 1:
                """
                    三维卷积：误差分别与输入卷积得到三维卷积核的梯度
                """
                w = []
                for j in range(self.core_thickness):
                    w.append(convolution(self.input_data[j], cnnls[i]))
            else:
                w = convolution(self.input_data, cnnls[i])
            cnn_w.append(w)
        return np.array(cnn_w), np.array(cnn_b)

    def update(self, delta_w, delta_b, decay=1):
        if self.frozen:
            return
        self.cores_b = [b - db for b, db in zip(self.cores_b, delta_b)]
        # self.cores_w = [[[w * decay - dw for w, dw in zip(lw, ldw)] for lw, ldw in zip(wi, dwi)] for wi, dwi in
        #                 zip(self.cores_w, delta_w)]
        self.cores_w = np.array(self.cores_w) - delta_w

    def front_layer_err(self, err, front_layer_z):
        err = np.array(err)
        front_layer_z = np.array(front_layer_z)
        err = err.reshape((self.core_num, int(np.sqrt(err.size / self.core_num)), -1))
        # 池化层反向传播误差
        cnnls = []
        outs = np.zeros(front_layer_z.shape)
        for e in err:
            cnnls.append(self.pooling.upsample(e, self.pooling_size))

        for core_w, e in zip(self.cores_w, cnnls):
            e = padding(e, self.core_size)
            if self.core_thickness > 1:
                for i in range(self.core_thickness):
                    outs[i] += convolution_2D(e, core_w[i], same=True)
            else:
                outs += convolution_2D(e, core_w, same=True)

        outs = outs * self.sigmoid_prime(front_layer_z)
        return outs


class Network(object):

    def __init__(self, layers, cost=CrossEntropyCost):
        self.layers = layers
        self.cost = cost

    def feedforward(self, a, cache=False):
        for layer in self.layers:
            a = layer.feedforward(a, cache=cache)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            show_log=True,
            stop_delay=-1):
        """
        训练逻辑与之前基本一样，但增加了L2规范化，故增加了参数 lmbda
        增加了网络性能监控的逻辑
        :param training_data:训练数据集
        :param epochs:训练周期
        :param mini_batch_size:随机小样本大小
        :param eta:学习速度
        :param lmbda: 非负，规范化参数
        :param evaluation_data:评估数据集
        :param monitor_evaluation_cost:是否监测评估数据集代价性能，仅当提供评估数据集时有效
        :param monitor_evaluation_accuracy:是否监测评估数据集正确度性能，仅当提供评估数据集时有效
        :param monitor_training_cost:是否监测训练数据集代价性能
        :param monitor_training_accuracy:是否监测训练数据集正确度性能
        :param show_log:是否显示实时日志
        :param stop_delay: 当给出一个大于零的整数是，将会自动停止，延迟周期为该数值
        :return:全部周期的监测数据，未监测的项返回空列表
        """
        if evaluation_data != None:
            n_data = len(evaluation_data)
        else:
            monitor_evaluation_cost = False
            monitor_evaluation_accuracy = False
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
            # 性能监控
            if show_log:
                print("周期 {} 训练完成".format(j + 1))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                if show_log:
                    print("训练数据集的代价: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                if show_log:
                    print("训练数据集正确数: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                if show_log:
                    print("估算数据集的代价: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                if show_log:
                    print("估算数据集的正确数: {} / {}".format(
                        accuracy, n_data))
            if show_log:
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
        # 保存各个层的 w,b 的梯度平均值
        nabla_b = []
        nabla_w = []
        for x, y in mini_batch:
            # 进行一次前向传播，纪录中间数据
            self.feedforward(x, cache=True)
            # 保存单个样本各个层梯度
            d_b, d_w = [], []
            # 计算输出层误差
            delta = self.cost.delta(self.layers[-1].z, self.layers[-1].a, y)
            # 反向传播输出层误差，计算各层 w,b 的梯度
            for i in [x for x in range(len(self.layers))][::-1]:
                # 得到误差的层
                layer = self.layers[i]
                # 利用该层误差计算梯度
                w, b = layer.backprop(delta)
                # 取均值
                w = w * eta / len(mini_batch)
                b = b * eta / len(mini_batch)

                if i > 0:  # 若存在前一层（非输入层），则将误差向前传播，计算前一层误差
                    front_layer = self.layers[i - 1]
                    delta = layer.front_layer_err(delta, front_layer.sigmoid_prime(front_layer.z))
                # 保存该层梯度
                d_b.append(b)
                d_w.append(w)
            if len(nabla_b) < len(self.layers):  # 如果是第一个样本
                # 初始化平均梯度
                nabla_b = d_b
                nabla_w = d_w
            else:  # 如果是后续样本
                for i in range(len(self.layers)):
                    nabla_b[i] = nabla_b[i] + d_b[i]
                    nabla_w[i] = nabla_w[i] + d_w[i]
        # 计算权重衰减
        decay = 1 - eta * lmbda / n
        # 依次更新各层参数
        for i in range(len(self.layers)):
            # 保存的梯度是由最后一层向前依次保存
            self.layers[i].update(nabla_w[-(i + 1)], nabla_b[-(i + 1)], decay)

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
        return cost

    def accuracy(self, data):
        """
        数据集正确率
        :param data: 数据集
        :return:数据集正确数
        """
        results = [(np.argmax(self.feedforward(x)), y)
                   for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


if __name__ == "__main__":
    # ################ 测试 Network #################################

    net = Network([CNNLayer(5, 3, tanh, tanh_prime), CNNLayer(5, 3, tanh, tanh_prime, core_thickness=3),
                   Layer(48, 10, sigmoid, sigmoid_prime)])
    from my_data import my_data

    my_data = [(np.array(data[0]).reshape(28, -1), data[1]) for data in my_data]
    net.SGD(my_data, 50, 10, 0.05, monitor_training_accuracy=True, monitor_training_cost=True)

    # ################ 测试 CNNLayer front_layer_err(err) ####################
    # data_in = np.random.randn(3, 3, 3)
    # print(data_in)
    # layer = CNNLayer(2, 2, ReLU, ReLU_prime, core_thickness=3, pooling_size=1)
    # out = layer.feedforward(data_in, cache=True)
    # print(out)
    # pe = layer.front_layer_err(out, data_in)
    # print(pe)

    ##############################################################
    # cnn = CNNLayer(5, 1, tanh, tanh_prime, pooling_size=1)
    # cnn.cores_b = [0, 0, 0]
    # data_in = np.random.randn(10, 10)
    # out = np.array(cnn.feedforward(data_in))
    # out -= np.array(cnn.feedforward(data_in))
    # out[0][0][0] += 1
    # print(np.array(cnn.cores_w[0]))
    # # print(np.array(out[0]))
    # print(np.array(convolution_2D(data_in, out[0])))

    # err = np.random.randn(144)
    # dw, db = cnn.backprop(err)
    # print(list(np.array(dw).flat))
    # print(np.average(np.array(err).flat))
    # print(np.average(np.array(dw).flat))
    # print(list(np.array(db).flat))
    # cnn.update(dw, db)

    # data_in = np.random.randn(3, 3, 3)
    # cnn = CNNLayer(2, 1, tanh, tanh_prime, core_thickness=3)
    # _w = np.random.randn(1, 3, 2, 2)
    # _b = np.random.randn(1)
    # cnn.update(_w, _b)
    # print('end')

    #  ###############  单元测试：测试单个层的 feedforward && backprop 单层网络性能 ####################
    # import matplotlib.pyplot as plt
    #
    # # 生成输入数据作为唯一样本
    # data_in = np.random.randn(3, 7, 7)
    # # 学习速率
    # eta = 0.05
    # # 收集没个学习周期（在线学习）的输出误差
    # plt_data = []
    # cost = []
    # # 期望输出向量，期望向量每个值相同
    # expect_value = 0.5
    # # 配置测试的层
    # test_layer = CNNLayer(2, 1, tanh, tanh_prime, pooling_size=2, core_thickness=3)
    #
    # out = test_layer.feedforward(data_in)
    # for i in range(np.array(out).size):
    #     plt_data.append([])
    # for ep in range(500):
    #     out = test_layer.feedforward(data_in, cache=True)
    #     out = np.array(out)
    #     _sum = 0
    #     for i, j in zip(list(out.flat), range(out.size)):
    #         plt_data[j].append(i)
    #         _sum += (i - expect_value) ** 2
    #     cost.append(_sum)
    #     _w, b = test_layer.backprop(out - expect_value)
    #     test_layer.update(_w * eta, b * eta)
    # print(test_layer.feedforward(data_in))
    # for d in plt_data:
    #     plt.plot(d)
    # plt.plot([0, 500], [0, 0], color='black')
    # plt.plot(cost)
    # plt.show()
