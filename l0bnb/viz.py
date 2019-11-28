import matplotlib.pyplot as plt


def graph_plot(x, y, x_axis, y_axis, title, loglog=True):
    plt.figure()
    func = plt.plot if not loglog else plt.loglog
    func(x, y)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)


def show_plots():
    plt.show()
