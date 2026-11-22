import numpy as np
import pygame as pg
import pygame_gui as pgui


class cont_map_mode:
    def __init__(self, manager, font, image_rect):
        self.name = "cont_map_mode"
        self.font = font
        self.image_rect = image_rect
        self.manager = manager
    
        self.mean_filter = pgui.elements.UICheckBox(
            text="Размытие по Гауссу/усреднением",
            relative_rect=pg.Rect((10, 10), (30, 30)),
            manager=manager,
        )

        self.highpass = pgui.elements.UICheckBox(
            text="Highpass filter",
            relative_rect=pg.Rect((10, 50), (30, 30)),
            manager=manager,
        )

        self.highpass_coef = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((160, 50), (190, 30)),
            start_value=1.0,
            value_range=(0.0, 2.0),
            manager=manager,
        )

        self.window_size_slider = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((110, 90), (190, 30)),
            start_value=1,
            value_range=(1, 8),
            manager=manager,
        )

        self.search_corners = pgui.elements.UICheckBox(
            text="Поиск углов",
            relative_rect=pg.Rect((420, 10), (30, 30)),
            manager=manager,
        )

        self.search_corners_thrsh = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((420, 90), (190, 30)),
            start_value=0.0,
            value_range=(0.0, 1.0),
            manager=manager,
        )
    
        self.save_button = pgui.elements.UIButton(
            relative_rect=pg.Rect((690, 90), (110, 30)),
            text="Save image",
            manager=manager,
        )


    def update_image(self, rgb):
        if self.mean_filter.get_state():
            kernel_size = int(self.window_size_slider.get_current_value()) * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
            kernel /= kernel.sum()
        else:
            kernel_size = int(self.window_size_slider.get_current_value()) * 2 + 1
            sigma = 0.2 * (kernel_size + 1)
            coords = np.arange(kernel_size) - (kernel_size - 1) / 2

            kernel_1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
            kernel = np.outer(kernel_1d, kernel_1d)
            kernel /= kernel.sum()

        print(kernel)
        image = rgb.astype(np.float32)
        gray = np.mean(image[..., :3], axis=-1)

        kh, kw = kernel.shape
        pad = kh // 2
        padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="edge")

        # создание набора масок
        windows = np.lib.stride_tricks.sliding_window_view( # нет Linux
            padded,
            (kh, kw)
        )

        # Поиск углов
        if self.search_corners.get_state():
            res = (self.get_corners(rgb) * 255).astype(np.uint8)
            res_image = np.stack((res, res, res), axis=-1)
            return pg.surfarray.make_surface(res_image).convert_alpha()

        # Highpass фильтр
        if self.highpass.get_state(): 
            # result[y, x] = sum(window[y, x, i, j] * kernel[i, j])
            blurred = np.einsum("yxij,ij->yx", windows, kernel)
            res = gray - blurred * self.highpass_coef.get_current_value()

            res_min = res.min()
            res_max = res.max()
            if res_max == res_min: res_max += 1
            res = (res - res_min) / (res_max - res_min) * 255.0

            res_image = np.stack((res, res, res), axis=-1)
            return pg.surfarray.make_surface(res_image).convert_alpha()
        # Карта контрастности
        else:
            mask = kernel.astype(bool)
            values = windows[:, :, mask]
            means = values.mean(axis=-1)
            contrast_map = np.sqrt(np.mean((values - means[:, :, None]) ** 2, axis=-1))

            c_min = contrast_map.min()
            c_max = contrast_map.max()
            if c_max == c_min: c_max += 1
            cont_map = (contrast_map - c_min) / (c_max - c_min) * 255.0
            cont_map = cont_map.astype(np.uint8)

            res_image = np.stack((cont_map, cont_map, cont_map), axis=-1)
            return pg.surfarray.make_surface(res_image).convert_alpha()


    @staticmethod
    def convolve2d(image, kernel): # свертка (применения окна к изображению)
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
            windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
            return np.einsum("yxij,ij->yx", windows, kernel)


    def get_corners(self, rgb):
        kernel_size = int(self.window_size_slider.get_current_value()) * 2 + 1
        sigma = 0.2 * (kernel_size + 1)
        coords = np.arange(kernel_size) - (kernel_size - 1) / 2

        kernel_1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss_kernel = np.outer(kernel_1d, kernel_1d)
        gauss_kernel /= gauss_kernel.sum()

        gray = np.mean(rgb.astype(np.float32)[..., :3], axis=-1) / 255.0 # диапазон 0..1
        blurred = self.convolve2d(gray, gauss_kernel)

        sobel_x = np.array(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        sobel_y = np.array(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            dtype=np.float32,
        )

        # частные производные по x и y
        l_x = self.convolve2d(blurred, sobel_x)
        l_y = self.convolve2d(blurred, sobel_y)

        l_x2 = l_x * l_x
        l_y2 = l_y * l_y
        l_xy = l_x * l_y

        # сглаживания для учета соседних пикселей
        s_xx = self.convolve2d(l_x2, gauss_kernel)
        s_yy = self.convolve2d(l_y2, gauss_kernel)
        s_xy = self.convolve2d(l_xy, gauss_kernel)

        det = s_xx * s_yy - s_xy * s_xy
        Tr = s_xx + s_yy
        h_response = det - 0.04 * Tr

        threshold = self.search_corners_thrsh.get_current_value()

        corners = (h_response > threshold).astype(np.uint8)
        return corners


    def draw(self, screen, display_image):
        screen.fill((30, 30, 30))
        screen.blit(display_image, self.image_rect)
        screen.blit(self.font.render(f"Размер окна {self.window_size_slider.get_current_value() * 2 + 1}",
                                      True, (255, 255, 255)), (10, 100))
        screen.blit(self.font.render(f"Порог {round(self.search_corners_thrsh.get_current_value(), 3)}",
                                      True, (255, 255, 255)), (420, 50))
        self.manager.draw_ui(screen)