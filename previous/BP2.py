import numpy as np
# Sigmoid激活函数


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Sigmoid激活函数的导数


def sigmoidDerivationx(y):
    return y * (1 - y)


if __name__ == '__main__':
    # 初始化一些参数
    alpha = 0.5
    numIter = 1
    b1 = 0.001
    b2 = 0.01  # 迭代次数
    w1 = [[0.15, 0.20, 0.10], [0.25, 0.30, 0.20]]   # Weight of input layer
    w2 = [[0.40], [0.45], [0.40]]   # 两层

    x = [[0.01, 0.02], [0.03, 0.00], [0.01, 0.00]]
    y = [[0.01], [0.02], [0.03]]

    z1 = np.dot(x, w1)+b1
    a1 = sigmoid(z1)
    #print(np.array(a1))
    z2 = np.dot(a1, w2)+b2
    a2 = sigmoid(z2)
   # a2=np.mat(a2)
    #print(np.array(a2))
    for n in range(numIter):
        delta2 = np.multiply(a2-y, np.multiply(a2, 1-a2))
        #print(np.array(delta2))
        delta1 = np.multiply(np.dot(np.array(w2).T, delta2), np.multiply(a1, 1-a1))
        print(np.array(delta1))
        w2 = np.mat(w2)-np.mat(alpha*(np.multiply(a1, delta2)))
        #print(np.array(w2))
        # #print(np.array(w1))
        w1 = np.mat(w1) - np.mat(alpha * (np.multiply(delta1, x)))
        
        
        z1 = np.dot(x, w1)+b1
        a1 = sigmoid(z1)
    # print(np.array(a1))
        z2 = np.dot(a1, w2)+b2
        a2 = sigmoid(z2)
    #    print(np.array(a2))
