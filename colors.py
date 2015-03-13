import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import common

dx, dy = 0.05, 0.05
xl, xr = -1, 1
yl, yr = -1, 1


def draw(func, steps):
    y, x = np.mgrid[slice(yl, yr + dy, dy),
                    slice(xl, xr + dx, dx)]
    z = func(x, y)
    z = z[:-1, :-1]
    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
    cmap = plt.get_cmap('PiYG')
    plt.contourf(x[:-1, :-1] + dx / 2.,
                 y[:-1, :-1] + dy / 2., z, levels=levels,
                 cmap=cmap)
    plt.colorbar()

    plt.plot(steps[0], steps[1], 'k-')

    plt.show()

if __name__ == '__main__':
    draw(common.f, common.grad_descend(common.f, 0.2, 0.8))
