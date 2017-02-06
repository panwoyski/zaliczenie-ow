import numpy as np
import matplotlib.pyplot as plt
import math


def f1(x, y):
    return x - y


def f2(x, y):
    return x + y


def f3(x, y):
    return -(x ** 2) + y


def f4(x, y):
    return -x - (y ** 2)


def zz(xx, yy, l, ff1, ff2):
    return l * ff1(xx, yy) + (1 - l) * ff2(xx, yy)


def plot2(ax, ff1, ff2):
    x = np.arange(-1, 1, 0.0001)
    y1 = np.sqrt(1 - np.power(x, 2))
    y2 = -y1
    tx1 = ff1(x, y1)
    ty1 = ff2(x, y1)
    tx2 = ff1(x, y2)
    ty2 = ff2(x, y2)
    ax.plot(tx1, ty1)
    ax.plot(tx2, ty2)
    ax.axis('equal')
    ax.grid()


def main():
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1)
    l = 0.5
    xx, yy = np.meshgrid(x, y, sparse=True)
    z1 = zz(xx, yy, l, f1, f2)
    z2 = zz(xx, yy, l, f2, f3)
    z3 = zz(xx, yy, l, f1, f3)
    f, xarr = plt.subplots(2, 2)
    xarr[0, 0].contourf(x, y, z1)
    xarr[0, 1].contourf(x, y, z2)
    xarr[1, 0].contourf(x, y, z3)
    plt.show()


def plot_f1f2(ff1, ff2):
    x = np.arange(-1, 1, 0.0001)
    print(x[0], x[-1])
    y1 = np.sqrt(1 - np.power(x, 2))
    print(y1[-1])
    y2 = -y1
    f, xarr = plt.subplots(1, 2)

    xarr[0].plot(x, y1)
    xarr[0].plot(x, y2)
    xarr[0].axis('equal')
    xarr[0].grid()

    tx1 = ff1(x, y1)
    ty1 = ff2(x, y1)
    tx2 = ff1(x, y2)
    ty2 = ff2(x, y2)
    xarr[1].plot(tx1, ty1)
    xarr[1].plot(tx2, ty2)
    xarr[1].axis('equal')
    xarr[1].grid()

    tx1m = tx1.min()
    tx2m = tx2.min()

    ty1m = ty1.min()
    ty2m = ty2.min()

    f1min = min(tx1m, tx2m)
    f2min = min(ty1m, ty2m)

    yind1 = ty1.argmin()
    yind2 = ty2.argmin()

    c1y = tx1[yind1], ty1[yind1]
    c2y = tx2[yind2], ty2[yind2]
    cymin = c1y if c1y[1] < c2y[1] else c2y
    xarr[1].plot(cymin[0], cymin[1], 'ok', label='cy')

    xind1 = tx1.argmin()
    xind2 = tx2.argmin()

    c1x = tx1[xind1], ty1[xind1]
    c2x = tx2[xind2], ty2[xind2]

    cxmin = c1x if c1x[0] < c2x[0] else c2x
    xarr[1].plot(cxmin[0], cxmin[1], 'ok', label='cx')

    zenit = cxmin[0], cymin[1]
    nadir = cymin[0], cxmin[1]

    l12x = np.arange(zenit[0], nadir[0], 0.0001)
    l12y = np.ones(len(l12x)) * nadir[1]

    l23y = np.arange(zenit[1], nadir[1], 0.0001)
    l23x = np.ones(len(l23y)) * nadir[0]

    l34x = np.arange(zenit[0], nadir[0], 0.0001)
    l34y = np.ones(len(l34x)) * zenit[1]

    l41y = np.arange(zenit[1], nadir[1], 0.0001)
    l41x = np.ones(len(l41y)) * zenit[0]

    xarr[1].plot(l12x, l12y, '--', color='grey')
    xarr[1].plot(l23x, l23y, '--', color='grey')
    xarr[1].plot(l34x, l34y, '--', color='grey')
    xarr[1].plot(l41x, l41y, '--', color='grey')

    xarr[1].plot(zenit[0], zenit[1], '*', label='punkt idealny')
    xarr[1].plot(nadir[0], nadir[1], '^', label='nadir')

    # xx = np.hstack((tx1, tx2))
    # yy = np.hstack((ty1, ty2))
    #
    # predicate = lambda x: zenit[0] <= x[0] <= nadir[0] and zenit[1] <= x[1] <= nadir[1]
    #
    # val_tup = list(filter(predicate, zip(xx, yy)))
    # xxn = [x[0] for x in val_tup]
    # yyn = [x[1] for x in val_tup]
    # print(xxn)
    # print(yyn)
    # print(len(xxn))
    # print(len(yyn))
    #
    # xarr[1].plot(xxn, yyn, 'k', label='front pareto')

    # xval = 0.5
    # bound = np.sqrt(1 - np.power(xval, 2))
    #
    # test1x = np.arange(-bound, bound, 0.0001)
    # test1y = np.ones(len(test1x)) * (-0.5)
    # test1f1 = ff1(test1x, test1y)
    # test1f2 = ff2(test1x, test1y)
    #
    # xarr[0].plot(test1x, test1y, 'r', label='y = 0.5')
    # xarr[1].plot(test1f1, test1f2, 'r')
    #
    # test2x = test1x
    # test2y = -1 * test1y
    # test2f1 = ff1(test2x, test2y)
    # test2f2 = ff2(test2x, test2y)
    #
    # xarr[0].plot(test2x, test2y, 'g', label='y = -0.5')
    # xarr[1].plot(test2f1, test2f2, 'g')
    #
    # test3x = test1y
    # test3y = test1x
    # test3f1 = ff1(test3x, test3y)
    # test3f2 = ff2(test3x, test3y)
    #
    # xarr[0].plot(test3x, test3y, 'b', label='x = 0.5')
    # xarr[1].plot(test3f1, test3f2, 'b')
    #
    # test4x = -1 * test3x
    # test4y = test3y
    # test4f1 = ff1(test4x, test4y)
    # test4f2 = ff2(test4x, test4y)
    #
    # xarr[0].plot(test4x, test4y, 'c', label='x = -0.5')
    # xarr[1].plot(test4f1, test4f2, 'c')
    #
    # test5x = np.arange(-1, 1, 0.0001)
    # test5y = np.zeros(len(test5x))
    # test5f1 = ff1(test5x, test5y)
    # test5f2 = ff2(test5x, test5y)
    #
    # xarr[0].plot(test5x, test5y, 'm', label='y = 0')
    # xarr[1].plot(test5f1, test5f2, 'm', label='y = 0')
    #
    # test6x = test5y
    # test6y = test5x
    # test6f1 = ff1(test6x, test6y)
    # test6f2 = ff2(test6x, test6y)
    #
    # xarr[0].plot(test6x, test6y, 'y', label='x = 0')
    # xarr[1].plot(test6f1, test6f2, 'y')

    xarr[0].legend()
    xarr[1].legend()
    plt.show()


def test():
    x = np.linspace(0, 4)
    y = np.linspace(0, 1)

    def f(x, y):
        return y * np.sin(x)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros((50, 50))

    for i in range(50):
        for j in range(50):
            Z[i, j] = f(X[i, j], Y[i, j])

    plt.pcolor(X, Y, Z)
    plt.show()


def mytest():
    N = 100
    x_start, x_end = -1.0, 1.0
    y_start, y_end = -1.0, 1.0

    x = np.linspace(x_start, x_end, N)
    y = np.linspace(y_start, y_end, N)

    x0, y0, radius = 0.0, 0.0, 1

    x, y = np.meshgrid(x, y)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    inside = r <= radius

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].set(xlabel='X', ylabel='Y', aspect=1.0)

    ax[0, 0].scatter(x[inside], y[inside])
    ax[0, 0].grid()

    ax[0, 1].set(xlabel='f1', ylabel='f2', aspect=1.0)
    ax[0, 1].scatter(f1(x[inside], y[inside]), f2(x[inside], y[inside]))
    ax[0, 1].grid()

    ax[1, 0].set(xlabel='f2', ylabel='f3', aspect=1.0)
    ax[1, 0].scatter(f2(x[inside], y[inside]), f3(x[inside], y[inside]))
    ax[1, 0].grid()

    ax[1, 1].set(xlabel='f1', ylabel='f3', aspect=1.0)
    ax[1, 1].scatter(f1(x[inside], y[inside]), f3(x[inside], y[inside]))
    ax[1, 1].grid()

    plt.show()

# mytest()
plot_f1f2(f1, f2)
plot_f1f2(f2, f3)
plot_f1f2(f1, f3)
