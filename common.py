import math

step_count = 1000
step_size = 0.001
d_step_count = 5000
d_step_size = 0.0001
eps = 0.0001

f_calc, g_calc = 0, 0


def f(x, y):
    global f_calc
    f_calc += 1
    return x ** 4 + y ** 4 - 5 * (x * y - 5 * x * x * y * y)


def vec_len(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def inv_vec((x, y)):
    return -x, -y


def sum_vec((x1, y1), (x2, y2)):
    return x1 + x2, y1 + y2


def scale_vec((x1, y1), k):
    return x1 * k, y1 * k


def get_ort_vec((x, y)):
    return y, -x


def normalize(v):
    l = vec_len(v)
    if l != 0:
        return scale_vec(v, 1 / l)
    else:
        return v


def grad(func, x, y):
    global g_calc
    g_calc += 1
    d = 0.01
    fdx = (func(x + d, y) - func(x - d, y)) / 2 / d
    fdy = (func(x, y + d) - func(x, y - d)) / 2 / d
    return normalize((fdx, fdy))


def reset_counters():
    global g_calc
    global f_calc
    g_calc = 0
    f_calc = 0


def grad_descend(func, x0, y0):
    reset_counters()
    x = x0
    y = y0
    steps = ([x], [y])
    for i in range(step_count):
        direction = inv_vec(grad(func, x, y))
        x, y = sum_vec((x, y), scale_vec(direction, step_size))
        steps[0].append(x)
        steps[1].append(y)
    print f_calc, g_calc
    return steps


def fastest_descend(func, x0, y0):
    reset_counters()
    x = x0
    y = y0
    steps = ([x], [y])
    while True:
        direction = inv_vec(grad(func, x, y))
        min_val = func(x, y)
        nx, ny = x, y
        for i in xrange(1, d_step_count):
            cx, cy = sum_vec((x, y), scale_vec(direction, d_step_size * i))
            cur_val = func(cx, cy)
            if cur_val < min_val:
                min_val = cur_val
                nx, ny = cx, cy
        steps[0].append(nx)
        steps[1].append(ny)
        if len(steps[0]) >= step_count or max(math.fabs(nx - x), math.fabs(ny - y)) < eps:
            break
        x, y = nx, ny

    print f_calc, g_calc
    return steps
