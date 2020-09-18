import csv
import numpy as np
from matplotlib import pyplot as plt


def fai(x, d):
    res = []
    for item in x:
        ans = [1]
        for i in range(d):
            ans.append(item ** (i + 1))
        res.append(ans)
    return np.array(res)


def read_csv(filename):
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(float(row[0]))
    return np.array(data)


def mse(w_d, ds, x, y):
    mse_d = []
    if ds == 7:
        ds = [7] * len(w_d)
    for w, d in zip(w_d, ds):
        mse = np.linalg.norm(y - np.dot(fai(x, d), w)) ** 2 / len(y)
        mse_d.append(mse)
    return mse_d


if __name__ == "__main__":
    x_train = read_csv('hw2_data/x_train.csv')
    y_train = read_csv('hw2_data/y_train.csv')

    # 1c
    # plt.scatter(x_train,y_train, label="scatter figure")

    # 1d
    w_d = []
    ds = [1, 2, 3, 7, 10]
    for d in ds:
        f = fai(x_train, d)
        w = np.dot(np.dot(np.linalg.inv(np.dot(f.T, f)), f.T), y_train)
        print('d=', d, ': w=', w)
        w_d.append(w)

    # 1e
    plt.figure(1)
    mse_train = mse(w_d, ds, x_train, y_train)
    plt.plot(ds, mse_train, label='mse_train')

    # 1f
    x_test = read_csv('hw2_data/x_test.csv')
    y_test = read_csv('hw2_data/y_test.csv')
    mse_test = mse(w_d, ds, x_test, y_test)
    plt.plot(ds, mse_test, label='mse_test')
    plt.xlabel('degree')
    plt.ylabel('mse')
    plt.legend()

    # 1g
    w_rr = []
    lamd = [10 ** (-5), 10 ** (-3), 10 ** (-1), 1, 10]
    f = fai(x_train, 7)
    for l in lamd:
        w = np.dot(np.dot(np.linalg.inv(np.diag([l] * len(f[0])) + np.dot(f.T, f)), f.T), y_train)
        print('lamda=', l, ': w=', w)
        w_rr.append(w)

    # 1h
    plt.figure(2)

    mse_train = mse(w_rr, 7, x_train, y_train)
    plt.plot([-5, -3, -1, 0, 1], mse_train, label='error_train')

    # 1f

    mse_test = mse(w_rr, 7, x_test, y_test)
    plt.plot([-5, -3, -1, 0, 1], mse_test, label='error_test')
    plt.xlabel('log(lamda)')
    plt.ylabel('mse')
    plt.legend()

    print('yes ok')
