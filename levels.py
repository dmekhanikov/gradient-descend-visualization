from Tkinter import *
import math
import common

CANVAS_WIDTH = 600
CANVAS_HEIGHT = 600


def to_window_coord((x, y)):
    xl = float(xl_entry.get())
    xr = float(xr_entry.get())
    yl = float(yl_entry.get())
    yr = float(yr_entry.get())
    x = min(1, max(0, (x - xl) / (xr - xl)))
    y = min(1, max(0, (1 - (y - yl) / (yr - yl))))
    return x * CANVAS_WIDTH, y * CANVAS_HEIGHT


def draw_line(func, x0, y0, inv):
    x, y = x0, y0
    eps = 0.005
    step_size = 0.001
    step_count = 50
    max_step_counts = 10000
    i = 0
    while i < step_count or (i < max_step_counts and max(math.fabs(x - x0), math.fabs(y - y0)) > eps):
        i += 1
        direction = common.get_ort_vec(common.grad(func, x, y))
        if inv:
            direction = common.inv_vec(direction)
        (nx, ny) = common.sum_vec((x, y), common.scale_vec(direction, step_size))
        (wnx, wny) = to_window_coord((nx, ny))
        (wx, wy) = to_window_coord((x, y))
        if wnx == 0 or wnx == CANVAS_WIDTH:
            return False
        if wny == 0 or wny == CANVAS_HEIGHT:
            return False
        canvas.create_line(wx, wy, wnx, wny)
        x, y = nx, ny
    return True


def draw(func, steps):
    canvas.delete("all")

    for i in range(1, len(steps[0])):
        prev = to_window_coord((steps[0][i - 1], steps[1][i - 1]))
        cur = to_window_coord((steps[0][i], steps[1][i]))
        canvas.create_line(prev[0], prev[1], cur[0], cur[1], fill="red")

    max_lines = 20
    lines_mod = max(1, len(steps[0]) / max_lines)

    for i in range(min(len(steps[0]), max_lines)):
        x0 = steps[0][i * lines_mod]
        y0 = steps[1][i * lines_mod]
        if not draw_line(func, x0, y0, False):
            draw_line(func, x0, y0, True)


def const_step(event):
    draw(common.f, common.grad_descend(common.f, float(x0_entry.get()), float(y0_entry.get())))


def fastest_descend(event):
    draw(common.f, common.fastest_descend(common.f, float(x0_entry.get()), float(y0_entry.get())))


root = Tk()
root.resizable(0, 0)
root.title("Gradient descent")

canvas = Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
panel_frame = Frame(root)
canvas.pack(side="left", fill="both", expand=1)
panel_frame.pack(side="right", fill="both", expand=1, pady=150)

xl_entry = Entry(panel_frame, width=3)
xr_entry = Entry(panel_frame, width=3)
xl_entry.insert(0, "-1")
xr_entry.insert(0, "1")
x_bounds_label = Label(panel_frame, text="<  x  <")

xl_entry.grid(row=1, column=1)
x_bounds_label.grid(row=1, column=2, columnspan=2)
xr_entry.grid(row=1, column=4)


yl_entry = Entry(panel_frame, width=3)
yr_entry = Entry(panel_frame, width=3)
yl_entry.insert(0, "-1")
yr_entry.insert(0, "1")
y_bounds_label = Label(panel_frame, text="<  y  <")

yl_entry.grid(row=2, column=1)
y_bounds_label.grid(row=2, column=2, columnspan=2)
yr_entry.grid(row=2, column=4)


x0_label = Label(panel_frame, text="x0:")
x0_entry = Entry(panel_frame, width=3)
y0_label = Label(panel_frame, text="y0:")
y0_entry = Entry(panel_frame, width=3)

x0_label.grid(row=3, column=1)
x0_entry.grid(row=3, column=2)
y0_label.grid(row=3, column=3)
y0_entry.grid(row=3, column=4)


const_step_btn = Button(panel_frame, text="Constant", width=6)
fastest_btn = Button(panel_frame, text="Fastest", width=6)
const_step_btn.bind('<Button-1>', const_step)
fastest_btn.bind("<Button-1>", fastest_descend)

const_step_btn.grid(row=4, column=1, columnspan=2)
fastest_btn.grid(row=4, column=3, columnspan=2)

if __name__ == "__main__":
    root.mainloop()
