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


def plot_full_shapes(pairs_list):
    N = 1000
    x_start, x_end = -1.0, 1.0
    y_start, y_end = -1.0, 1.0

    x = np.linspace(x_start, x_end, N)
    y = np.linspace(y_start, y_end, N)

    x0, y0, radius = 0.0, 0.0, 1

    x, y = np.meshgrid(x, y)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    inside = r <= radius

    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.scatter(x[inside], y[inside])
    plt.title('Zbior U {(x,y): x^2 + y^2 <= 1}')
    plt.grid()

    for ff1, ff2 in pairs_list:
        plt.figure()
        plt.title('Obraz zbioru U w (%s, %s)' % (ff1.__name__, ff2.__name__))
        plt.xlabel(ff1.__name__)
        plt.ylabel(ff2.__name__)
        plt.axis('equal')
        plt.scatter(ff1(x[inside], y[inside]), ff2(x[inside], y[inside]))
        plt.grid()

    # plt.figure()
    # plt.title('Obraz zbioru U w (F2, F3)')
    # plt.xlabel('f2')
    # plt.ylabel('f3')
    # plt.axis('equal')
    # plt.scatter(ff2(x[inside], y[inside]), ff3(x[inside], y[inside]))
    # plt.grid()
    #
    # plt.figure()
    # plt.title('Obraz zbioru U w (F1, F3)')
    # plt.xlabel('f1')
    # plt.ylabel('f3')
    # plt.axis('equal')
    # plt.scatter(F1(x[inside], y[inside]), F3(x[inside], y[inside]))
    # plt.grid()

    plt.show()


def sort_by_cartesian(arr, point):
    if not len(arr):
        return arr
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
    xf1minval = points[:, 0].min()
    suspect1_indices = np.where(points[:, 0] == xf1minval)
    yf1minval = points[:, 1][suspect1_indices].min()
    f1min = xf1minval, yf1minval

    yf2minval = points[:, 1].min()
    suspect2_indices = np.where(points[:, 1] == yf2minval)
    xf2minval = points[:, 0][suspect2_indices].min()
    f2min = xf2minval, yf2minval

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
    return sort_by_cartesian(temp2_list, (1, 0))


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


def plot_f1f2(params_dict):
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

    plt.figure(1)
    plot_points(plt, xy_points)
    plt.axis('equal')
    plt.grid()
    plt.title('Front paretu na brzegu zbioru U dla przestrzeni (%s, %s)' % (ff1.__name__, ff2.__name__))
    plt.xlabel('X')
    plt.ylabel('Y')
    plot_points(plt, xy_front, 'k', linewidth=3, label='front pareto')
    plt.legend()

    plt.figure(2)
    plt.title('Front pareto na obrazie zbioru U w przestrzeni (%s, %s)' % (ff1.__name__, ff2.__name__))
    plot_points(plt, f1f2_points)
    plt.xlabel(ff1.__name__)
    plt.ylabel(ff2.__name__)
    plt.axis('equal')
    plt.grid()

    plt.plot(cxmin[0], cxmin[1], 'ok')
    plt.plot(cymin[0], cymin[1], 'ok')
    plt.plot(zenit[0], zenit[1], '^c', label='zenit')
    plt.plot(nadir[0], nadir[1], 'xm', label='nadir')

    add_square_boundary_to_plot(plt, zenit, nadir, '--', color='grey')

    plot_points(plt, f1f2_front, 'k', linewidth=3, label='front pareto')
    plt.legend()

    # if plot_geometry:
    #     geometry_plot(xarr, ff1, ff2)

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

    base_points1 = np.vstack((x, y2)).T
    base_points2 = np.flipud(np.vstack((x, y1)).T)
    return np.vstack((base_points1, base_points2))


def intersect_2d_array(a, b):
    temp_arr = np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])
    return sort_by_cartesian(temp_arr, (0, -1))


def plot_summary(f1f2d, f2f3d, f1f3d):
    f1f2 = f1f2d['xy_front']
    f2f3 = f2f3d['xy_front']
    f1f3 = f1f3d['xy_front']
    print(f1f2.shape, f2f3.shape, f1f3.shape)

    f1f2_n_f2f3 = intersect_2d_array(f1f2, f2f3)
    f2f3_n_f1f3 = intersect_2d_array(f2f3, f1f3)
    f1f2_n_f1f3 = intersect_2d_array(f1f2, f1f3)

    plot_points(plt, f1f2d['xy_points'])
    plot_points(plt, f1f2, '-', linewidth=4, label='f1f4')
    plot_points(plt, f2f3, '-', linewidth=4, label='f2f4')
    plot_points(plt, f1f3, '-', linewidth=4, label='f3f4')
    if len(f1f2_n_f1f3):
        plot_points(plt, f1f2_n_f1f3, '-', linewidth=3, label='f1f4 n f2f4')
    if len(f1f2_n_f2f3):
        plot_points(plt, f1f2_n_f2f3, '-', linewidth=3, label='f1f4 n f3f4')
    if len(f2f3_n_f1f3):
        plot_points(plt, f2f3_n_f1f3, 'ok', linewidth=3, label='f2f4 n f3f4')
        print(len(f2f3_n_f1f3))

    plt.title('Zlozenie frontow pareto dla par (F1, F4), (F2, F4), (F3, F4)')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    xy_points = generate_xy_params(0.0001)
    # plot_full_shapes([(F1, F4), (F2, F4), (F3, F4)])
    f1f4_dict = generate_parameters(xy_points, F1, F4)
    plot_f1f2(f1f4_dict)
    f2f4_dict = generate_parameters(xy_points, F2, F4)
    plot_f1f2(f2f4_dict)
    f3f4_dict = generate_parameters(xy_points, F3, F4)
    plot_f1f2(f3f4_dict)
    plot_summary(f1f4_dict, f2f4_dict, f3f4_dict)

    np.savetxt('xy_points.csv', xy_points, delimiter=',')

    np.savetxt('xy_f1f4_pareto.csv', f1f4_dict['xy_front'], delimiter=',')
    np.savetxt('ff_f1f4_pareto.csv', f1f4_dict['f1f2_front'], delimiter=',')

    np.savetxt('xy_f2f4_pareto.csv', f2f4_dict['xy_front'], delimiter=',')
    np.savetxt('ff_f2f4_pareto.csv', f2f4_dict['f1f2_front'], delimiter=',')

    np.savetxt('xy_f3f4_pareto.csv', f3f4_dict['xy_front'], delimiter=',')
    np.savetxt('ff_f3f4_pareto.csv', f3f4_dict['f1f2_front'], delimiter=',')

if __name__ == '__main__':
    main()
