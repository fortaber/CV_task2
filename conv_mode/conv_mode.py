import math
import re

import numpy as np
import pygame as pg
import pygame_gui as pgui


class conv_mode:
    def __init__(self, screen_size, main_image, image_rect, font):
        self.name = "Свёртка"
        self.screen_size = screen_size
        self.main_image = main_image
        self.image_rect = image_rect
        self.font = font
        self.manager = None

        self.current_kernel = None
        self.normalize = False
        self.add_128 = False


    def get_ui(self, mgr: pgui.UIManager, img: pg.surface.Surface):
        self.manager = mgr

        self.matrix_entry = pgui.elements.UITextEntryLine(
            initial_text="1",
            relative_rect=pg.Rect((0, 0), (740, 30)),
            manager=mgr,
        )

        self.normalize_checkbox = pgui.elements.UICheckBox(
            relative_rect=pg.Rect((0, 40), (30, 30)),
            text="Нормализация",
            manager=mgr,
        )
        self.add128_checkbox = pgui.elements.UICheckBox(
            relative_rect=pg.Rect((230, 40), (30, 30)),
            text="+128",
            manager=mgr,
        )

        self.laplassian_btn = pgui.elements.UIButton(
            relative_rect=pg.Rect((330, 40), (200, 30)),
            text="Дискретный лапласиан",
            manager=mgr,
        )

        self.sobel_hor_btn = pgui.elements.UIButton(
            relative_rect=pg.Rect((330, 80), (210, 30)),
            text="Оператор Собеля гор.",
            manager=mgr,
        )

        
        self.sobel_ver_btn = pgui.elements.UIButton(
            relative_rect=pg.Rect((330, 120), (210, 30)),
            text="Оператор Собеля верт.",
            manager=mgr,
        )

        self.save_btn = pgui.elements.UIButton(
            relative_rect=pg.Rect((600, 70), (110, 30)),
            text="Save image",
            manager=mgr,
        )


    def parse_kernel(self, text):
        tokens = [t for t in re.split(r"[ ;]", text.strip()) if t] # if t - отбрасывает пустые строки
        if not tokens:
            raise ValueError("empty")
        values = [float(t) for t in tokens]
        n = int(round(math.sqrt(len(values))))
        if n * n != len(values) or n < 1:
            raise ValueError("invalid size")
        return np.array(values, dtype=np.float64).reshape(n, n)


    def apply_convolution(self, rgb, kernel):
        kh, kw = kernel.shape
        if kh == 0 or kw == 0:
            return rgb.copy()

        pad_top = kh // 2
        pad_bottom = kh - 1 - pad_top
        pad_left = kw // 2
        pad_right = kw - 1 - pad_left

        img_f = np.pad(
            rgb.astype(np.float64),
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )

        windows = np.lib.stride_tricks.sliding_window_view(img_f, (kh, kw), axis=(0, 1))
        result = np.einsum("hwcij,ij->hwc", windows, kernel, optimize=True)

        if self.normalize_checkbox.get_state():
            kernel_sum = float(kernel.sum())
            if abs(kernel_sum) > 1e-12:
                result = result / kernel_sum

        if self.add128_checkbox.get_state():
            result = result + 128

        return np.clip(result, 0, 255).astype(np.uint8)


    def update_image(self, rgb):
        try:
            text = ""
            if hasattr(self, "matrix_entry") and self.matrix_entry is not None:
                text = self.matrix_entry.get_text()
            kernel = self.parse_kernel(text)
            self.current_kernel = kernel
            processed = self.apply_convolution(rgb, kernel)
            return pg.surfarray.make_surface(processed)
        except Exception:
            self.current_kernel = None
            return pg.surfarray.make_surface(np.zeros_like(rgb, dtype=np.uint8))


    def process_event(self, event):
        if event.type == pgui.UI_TEXT_ENTRY_FINISHED and getattr(event, "ui_element", None) == getattr(self, "matrix_entry", None):
            try:
                self.current_kernel = self.parse_kernel(event.text)
            except Exception:
                self.current_kernel = None
        if event.type == pgui.UI_BUTTON_PRESSED:
            if event.ui_element == self.laplassian_btn:
                self.matrix_entry.set_text("1 1 1; 1 -8 1; 1 1 1")
            if event.ui_element == self.sobel_hor_btn:
                self.matrix_entry.set_text("-1 -2 -1; 0 0 0; 1 2 1")
            if event.ui_element == self.sobel_ver_btn:
                self.matrix_entry.set_text("-1 0 1; -2 0 2; -1 0 1")


    def draw(self, screen, display_image):
        screen.fill((30, 30, 30))
        self.manager.draw_ui(screen)
        screen.blit(display_image, self.image_rect)
