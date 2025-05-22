import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Размер матрицы
N = 100
mask = np.zeros((N, N), dtype=int)    # апертура
intensity = np.zeros((N, N))          # дифракционная картина

drawing = False  # флаг для рисования по движению

# Создаем фигуру
fig = plt.figure(figsize=(12, 6))

# Координаты и размеры областей
btn_x = 0.02; btn_w = 0.18; btn_h = 0.08; btn_gap = 0.02
ap_x = btn_x + btn_w + 0.02; ap_w = 0.35
diff_x = ap_x + ap_w + 0.02; diff_w = 0.35

# Левая панель кнопок для готовых объектов
btns = []
labels = ['Верт. сетка', 'Гориз. сетка', 'Квадр. решетка', 'Треугольник', 'Окружность']
for i, lab in enumerate(labels):
    ax = fig.add_axes([btn_x,
                       1 - (i+1)*(btn_h+btn_gap),
                       btn_w, btn_h])
    btn = Button(ax, lab)
    btns.append(btn)

# Кнопка "Запустить" и "Сброс"
ax_run = fig.add_axes([btn_x, 0.15, btn_w, btn_h])
btn_run = Button(ax_run, 'Запустить')
ax_reset = fig.add_axes([btn_x, 0.05, btn_w, btn_h])
btn_reset = Button(ax_reset, 'Сброс')

# Оси экранов
ax_obj = fig.add_axes([ap_x, 0.1, ap_w, 0.85])
ax_diff = fig.add_axes([diff_x, 0.1, diff_w, 0.85])

# Отображение апертуры
im_obj = ax_obj.imshow(mask, cmap='gray_r', vmin=0, vmax=1)
ax_obj.set_title('Апертурный экран (рисование)')
ax_obj.set_xticks(np.arange(-0.5, N, 1))
ax_obj.set_yticks(np.arange(-0.5, N, 1))
ax_obj.grid(color='lightgray', linewidth=0.5)
for spine in ax_obj.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)
ax_obj.set_xticklabels([])
ax_obj.set_yticklabels([])

# Отображение дифракции
im_diff = ax_diff.imshow(intensity, cmap='gray', vmin=0, vmax=1)
ax_diff.set_title('Дифракционная картина')
ax_diff.axis('off')

# Функция для обновления объекта (маски)
def update_mask(new_mask):
    global mask
    mask = new_mask.copy()
    im_obj.set_data(mask)
    im_diff.set_data(np.zeros_like(mask))
    fig.canvas.draw_idle()

# Функции для готовых объектов

def vert_lines(event):
    m = np.zeros_like(mask)
    for j in range(0, N, 8): m[:, j:j+4] = 1
    update_mask(m)


def horiz_lines(event):
    m = np.zeros_like(mask)
    for i in range(0, N, 8): m[i:i+4, :] = 1
    update_mask(m)


def square_grid(event):
    m = np.zeros_like(mask)
    for i in range(0, N, 8):
        for j in range(0, N, 8): m[i:i+4, j:j+4] = 1
    update_mask(m)


def triangle(event):
    m = np.zeros_like(mask)
    side = 15; h = int(side*np.sqrt(3)/2); thickness = 1
    x0 = (N-side)/2; y0 = (N+h)/2
    V = [(x0, y0), (x0+side, y0), (x0+side/2, y0-h)]
    def draw_thick_line(mat, p1, p2):
        x1,y1=p1; x2,y2=p2
        length=int(np.hypot(x2-x1, y2-y1)*2)+1
        xs=np.linspace(x1,x2,length); ys=np.linspace(y1,y2,length)
        dx,dy=(y1-y2),(x2-x1); norm=np.hypot(dx,dy)
        dx/=norm; dy/=norm
        for t in np.linspace(-thickness/2, thickness/2, thickness*2):
            for x,y in zip(xs,ys):
                xi=int(round(x+dx*t)); yi=int(round(y+dy*t))
                if 0<=xi<N and 0<=yi<N: mat[yi, xi]=1
    for p1,p2 in [(V[0],V[1]),(V[1],V[2]),(V[2],V[0])]: draw_thick_line(m,p1,p2)
    update_mask(m)


def circle(event):
    m = np.zeros_like(mask)
    r=10; thickness=1; cy, cx = N//2, N//2
    yy, xx = np.ogrid[:N, :N]
    dist = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    mask_ring = (dist >= r-thickness/2) & (dist <= r+thickness/2)
    m[mask_ring] = 1
    update_mask(m)

# Привязка кнопок пресетов
btns_funcs = [vert_lines, horiz_lines, square_grid, triangle, circle]
for btn, func in zip(btns, btns_funcs):
    btn.on_clicked(func)

# Функции клика и перетаскивания

def toggle_pixel(event):
    if event.inaxes == ax_obj:
        i, j = int(event.ydata), int(event.xdata)
        if 0 <= i < N and 0 <= j < N:
            mask[i, j] = 1 - mask[i, j]
            im_obj.set_data(mask)
            fig.canvas.draw_idle()


def set_pixel(event):
    if event.inaxes == ax_obj:
        i, j = int(event.ydata), int(event.xdata)
        if 0 <= i < N and 0 <= j < N and mask[i, j] == 0:
            mask[i, j] = 1
            im_obj.set_data(mask)
            fig.canvas.draw_idle()


def on_press(event):
    global drawing
    if event.inaxes == ax_obj:
        drawing = True
        toggle_pixel(event)


def on_motion(event):
    if drawing:
        set_pixel(event)


def on_release(event):
    global drawing
    drawing = False

# Привязка событий мыши
def connect_events():
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

connect_events()

# Функция расчета и отображения дифракции

def run_diffraction(event):
    A = 1.0 - mask
    F = np.fft.fftshift(np.fft.fft2(A))
    I = np.abs(F) ** 2
    I_log = np.log1p(I)
    I_norm = I_log / I_log.max()
    im_diff.set_data(I_norm)
    fig.canvas.draw_idle()

btn_run.on_clicked(run_diffraction)

# Функция сброса

def reset_all(event):
    global mask, intensity
    mask[:] = 0
    intensity[:] = 0
    im_obj.set_data(mask)
    im_diff.set_data(intensity)
    fig.canvas.draw_idle()

btn_reset.on_clicked(reset_all)

plt.show()


