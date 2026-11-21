import numpy as np
import pygame as pg
import pygame_gui as pgui


class cont_map_mode:
    def __init__(self, manager, font, image_rect):
        self.name = "cont_map_mode"
        self.font = font
        self.image_rect = image_rect
        self.manager = manager
        self.four_conn_checkbox = pgui.elements.UICheckBox(
            text="4-связность",
            relative_rect=pg.Rect((10, 10), (30, 30)),
            manager=manager,
        )
    
        self.eight_conn_checkbox = pgui.elements.UICheckBox(
            text="8-связность",
            relative_rect=pg.Rect((10, 50), (30, 30)),
            manager=manager,
        )
    
        self.window_size_slider = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((120, 90), (190, 30)),
            start_value=1,
            value_range=(1, 8),
            manager=manager,
        )
    
        self.save_button = pgui.elements.UIButton(
            relative_rect=pg.Rect((510, 70), (110, 30)),
            text="Save image",
            manager=manager,
        )


    def update_image(self, rgb):
        if self.four_conn_checkbox.get_state():
            kernel=np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
        elif self.eight_conn_checkbox.get_state():
            kernel = np.ones((3, 3), dtype=np.uint8)
        else:
            kernel_size = self.window_size_slider.get_current_value() * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

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
        # print(windows.shape) # выведет (image.w, image.h, kw, kh)

        # приведение ядра к бинарным значениям
        mask = kernel.astype(bool)

        values = windows[:, :, mask]
        means = values.mean(axis=-1)
        contrast_map = np.sqrt(np.mean((values - means[:, :, None]) ** 2, axis=-1))


        # нормализация в диапазон 0..255
        c_min = contrast_map.min()
        c_max = contrast_map.max()
        cont_map = (contrast_map - c_min) / (c_max - c_min) * 255.0
        cont_map = cont_map.astype(np.uint8)

        res_image = np.stack((cont_map, cont_map, cont_map), axis=-1)
        return pg.surfarray.make_surface(res_image).convert_alpha()



    def draw(self, screen, display_image):
        screen.fill((30, 30, 30))
        screen.blit(display_image, self.image_rect)
        screen.blit(self.font.render(f"Размер окна {self.window_size_slider.get_current_value() * 2 + 1}", True, (255, 255, 255)), (10, 100))
        self.manager.draw_ui(screen)