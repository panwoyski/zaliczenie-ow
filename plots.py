import numpy as np
import matplotlib.pyplot as plt


def F1(x, y):
    return x - y


def F2(x, y):
    return x + y


def F3(x, y):
    return -(x ** 2) + y


def F4(x, y):
    return -x - (y ** 2)


def mytest():
    N = 1000
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
    ax[0, 1].scatter(F1(x[inside], y[inside]), F2(x[inside], y[inside]))
    ax[0, 1].grid()

    ax[1, 0].set(xlabel='f2', ylabel='f3', aspect=1.0)
    ax[1, 0].scatter(F2(x[inside], y[inside]), F3(x[inside], y[inside]))
    ax[1, 0].grid()

    ax[1, 1].set(xlabel='f1', ylabel='f3', aspect=1.0)
    ax[1, 1].scatter(F1(x[inside], y[inside]), F3(x[inside], y[inside]))
    ax[1, 1].grid()

    plt.show()


def sort_by_cartesian(arr, point):
    temp_arr = arr - point
    sort = np.sum(np.power(temp_arr, 2), axis=1)
    return arr[sort.argsort()]


def plot_points(ax, points, *args, **kwargs):
    ax.plot(points[:, 0], points[:, 1], *args, **kwargs)


def transform_by_func(points, ff1, ff2):
    tx1 = ff1(points[:, 0], points[:, 1])
    ty1 = ff2(points[:, 0], points[:, 1])
    return np.vstack((tx1, ty1)).T


def find_significant(points):
    f1min = points[points[:, 0].argmin()]
    f2min = points[points[:, 1].argmin()]
    nadir = f2min[0], f1min[1]
    zenit = f1min[0], f2min[1]
    return zenit, nadir, f1min, f2min


def add_square_boundary_to_plot(ax, zenit, nadir, *args, **kwargs):
    l12x = np.arange(zenit[0], nadir[0], 0.0001)
    l12y = np.ones(len(l12x)) * nadir[1]

    l23y = np.arange(zenit[1], nadir[1], 0.0001)
    l23x = np.ones(len(l23y)) * nadir[0]

    l34x = np.arange(zenit[0], nadir[0], 0.0001)
    l34y = np.ones(len(l34x)) * zenit[1]

    l41y = np.arange(zenit[1], nadir[1], 0.0001)
    l41x = np.ones(len(l41y)) * zenit[0]

    ax.plot(l12x, l12y, *args, **kwargs)
    ax.plot(l23x, l23y, *args, **kwargs)
    ax.plot(l34x, l34y, *args, **kwargs)
    ax.plot(l41x, l41y, *args, **kwargs)


def get_pareto_front(points, zenit, nadir):
    front = points[points[:, 0] <= nadir[0]]
    front = front[zenit[0] <= front[:, 0]]
    front = front[zenit[1] <= front[:, 1]]
    front = front[front[:, 1] <= nadir[1]]

    return sort_by_cartesian(front, (zenit[0], nadir[1]))


def get_front_in_xy(xy_points, ff1, ff2, zenit, nadir):
    temp_list = [(x, y) for x, y in xy_points if zenit[0] <= ff1(x, y) <= nadir[0] and zenit[1] <= ff2(x, y) <= nadir[1]]
    temp2_list = np.asarray(temp_list)
    return sort_by_cartesian(temp2_list, (0, 1))


def generate_parameters(xy_points, ff1, ff2):
    f1f2_points = transform_by_func(xy_points, ff1, ff2)
    zenit, nadir, cxmin, cymin = find_significant(f1f2_points)

    f1f2_front = get_pareto_front(f1f2_points, zenit, nadir)
    xy_front = get_front_in_xy(xy_points, ff1, ff2, zenit, nadir)

    return {
        'xy_points': xy_points,
        'f1f2_points': f1f2_points,
        'xy_front': xy_front,
        'f1f2_front': f1f2_front,
        'zenit': zenit,
        'nadir': nadir,
        'cxmin': cxmin,
        'cymin': cymin,
        'ff1': ff1,
        'ff2': ff2
    }


def plot_f1f2(params_dict, plot_geometry=False):
    xy_points = params_dict['xy_points']
    f1f2_points = params_dict['f1f2_points']
    xy_front = params_dict['xy_front']
    f1f2_front = params_dict['f1f2_front']
    zenit = params_dict['zenit']
    nadir = params_dict['nadir']
    cxmin = params_dict['cxmin']
    cymin = params_dict['cymin']
    ff1 = params_dict['ff1']
    ff2 = params_dict['ff2']

    f, xarr = plt.subplots(1, 2)

    plot_points(xarr[0], xy_points)
    xarr[0].axis('equal')
    xarr[0].grid()

    plot_points(xarr[1], f1f2_points)
    xarr[1].axis('equal')
    xarr[1].grid()

    xarr[1].plot(cxmin[0], cxmin[1], 'ok')
    xarr[1].plot(cymin[0], cymin[1], 'ok')
    xarr[1].plot(zenit[0], zenit[1], '^c', label='zenit')
    xarr[1].plot(nadir[0], nadir[1], 'xm', label='nadir')
    xarr[1].legend()

    add_square_boundary_to_plot(xarr[1], zenit, nadir, '--', color='grey')

    plot_points(xarr[0], xy_front, 'y', linewidth=3)
    plot_points(xarr[1], f1f2_front, 'y', linewidth=3)

    if plot_geometry:
        geometry_plot(xarr, ff1, ff2)

    plt.show()


def geometry_plot(xarr, ff1, ff2):
    xval = 0.5
    bound = np.sqrt(1 - np.power(xval, 2))

    test1x = np.arange(-bound, bound, 0.0001)
    test1y = np.ones(len(test1x)) * (-0.5)
    test1f1 = ff1(test1x, test1y)
    test1f2 = ff2(test1x, test1y)

    xarr[0].plot(test1x, test1y, 'r', label='y = 0.5')
    xarr[1].plot(test1f1, test1f2, 'r')

    test2x = test1x
    test2y = -1 * test1y
    test2f1 = ff1(test2x, test2y)
    test2f2 = ff2(test2x, test2y)

    xarr[0].plot(test2x, test2y, 'g', label='y = -0.5')
    xarr[1].plot(test2f1, test2f2, 'g')

    test3x = test1y
    test3y = test1x
    test3f1 = ff1(test3x, test3y)
    test3f2 = ff2(test3x, test3y)

    xarr[0].plot(test3x, test3y, 'b', label='x = 0.5')
    xarr[1].plot(test3f1, test3f2, 'b')

    test4x = -1 * test3x
    test4y = test3y
    test4f1 = ff1(test4x, test4y)
    test4f2 = ff2(test4x, test4y)

    xarr[0].plot(test4x, test4y, 'c', label='x = -0.5')
    xarr[1].plot(test4f1, test4f2, 'c')

    test5x = np.arange(-1, 1, 0.0001)
    test5y = np.zeros(len(test5x))
    test5f1 = ff1(test5x, test5y)
    test5f2 = ff2(test5x, test5y)

    xarr[0].plot(test5x, test5y, 'm', label='y = 0')
    xarr[1].plot(test5f1, test5f2, 'm', label='y = 0')

    test6x = test5y
    test6y = test5x
    test6f1 = ff1(test6x, test6y)
    test6f2 = ff2(test6x, test6y)

    xarr[0].plot(test6x, test6y, 'y', label='x = 0')
    xarr[1].plot(test6f1, test6f2, 'y')

    xarr[0].legend()


def generate_xy_params(density):
    x = np.arange(-1, 1, density)
    y2 = np.sqrt(1 - np.power(x, 2))
    y1 = -y2

    base_points1 = np.vstack((x, y1)).T
    base_points2 = np.flipud(np.vstack((x, y2)).T)
    return np.vstack((base_points1, base_points2))


def main():
    xy_points = generate_xy_params(0.0001)
    # mytest()
    f1f2_dict = generate_parameters(xy_points, F1, F2)
    plot_f1f2(f1f2_dict)
    f2f3_dict = generate_parameters(xy_points, F2, F3)
    plot_f1f2(f2f3_dict, plot_geometry=True)
    f1f3_dict = generate_parameters(xy_points, F1, F3)
    plot_f1f2(f1f3_dict)


if __name__ == '__main__':
    main()
