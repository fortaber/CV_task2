import math

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import pygame_gui as pgui
from matplotlib.backends.backend_agg import FigureCanvasAgg


class effects_mode:
    def __init__(self, manager, screen_size, image, image_rect, font, rgb):
        self.name = "effects_mode"
        self.manager = manager
        self.screen_size = screen_size
        self.image = image
        self.image_rect = image_rect
        self.font = font
        self.grayscale_image = np.mean(rgb[..., :3], axis=-1)

        channels = ["Red", "Green", "Blue"]
        self.select_r_source = pgui.elements.UIDropDownMenu(
            options_list=channels,
            starting_option=channels[0],
            relative_rect=pg.Rect((30, 30), (80, 30)),
            manager=manager,
        )
        self.select_g_source = pgui.elements.UIDropDownMenu(
            options_list=channels,
            starting_option=channels[1],
            relative_rect=pg.Rect((120, 30), (80, 30)),
            manager=manager,
        )
        self.select_b_source = pgui.elements.UIDropDownMenu(
            options_list=channels,
            starting_option=channels[2],
            relative_rect=pg.Rect((210, 30), (80, 30)),
            manager=manager,
        )

        self.r_level_slider = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((30, 80), (190, 30)),
            start_value=1.0,
            value_range=(0, 5),
            manager=manager,
        )
        self.r_level_inv = pgui.elements.UICheckBox(
            text="",
            relative_rect=pg.Rect((230, 80), (30, 30)),
            manager=manager,
        )
        self.g_level_slider = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((270, 80), (190, 30)),
            start_value=1.0,
            value_range=(0, 5),
            manager=manager,
        )
        self.g_level_inv = pgui.elements.UICheckBox(
            text="",
            relative_rect=pg.Rect((470, 80), (30, 30)),
            manager=manager,
        )
        self.b_level_slider = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((510, 80), (190, 30)),
            start_value=1.0,
            value_range=(0, 5),
            manager=manager,
        )
        self.b_level_inv = pgui.elements.UICheckBox(
            text="",
            relative_rect=pg.Rect((710, 80), (30, 30)),
            manager=manager,
        )
        self.contrast_slider = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((300, 30), (190, 30)),
            start_value=5.0,
            value_range=(0, 50),
            manager=manager,
        )
        self.mirror_v = pgui.elements.UICheckBox(
            text="Mirror vertical",
            relative_rect=pg.Rect((500, 30), (30, 30)),
            manager=manager,
        )
        self.mirror_h = pgui.elements.UICheckBox(
            text="Mirror horizontal",
            relative_rect=pg.Rect((640, 30), (30, 30)),
            manager=manager,
        )
        self.line_slider_for_graph = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((30, 140), (190, 30)),
            start_value=0.0,
            value_range=(0, 1),
            manager=manager,
        )
        self.save_button = pgui.elements.UIButton(
            relative_rect=pg.Rect((270, 140), (110, 30)),
            text="Save image",
            manager=manager,
        )

        self.fig_line, self.ax_line = plt.subplots(figsize=(3, 2), dpi=100)
        self.ax_line.plot(self.grayscale_image[:, 0])
        self.ax_line.grid(True)
        self.plot_line = FigureCanvasAgg(self.fig_line)
        self.plot_line.draw()
        self.plot_line_surf = self._make_plot_surface()

    def _make_plot_surface(self):
        return pg.image.frombytes(
            self.plot_line.buffer_rgba().tobytes(),
            self.plot_line.get_width_height(),
            "RGBA",
        )

    def update_image(self, rgb):
        image = rgb.astype(np.float32)

        channels = ["Red", "Green", "Blue"]
        r_source = channels.index(self.select_r_source.selected_option[0])
        g_source = channels.index(self.select_g_source.selected_option[0])
        b_source = channels.index(self.select_b_source.selected_option[0])
        image = image[:, :, [r_source, g_source, b_source]]

        channel_levels = np.array([
            self.r_level_slider.get_current_value(),
            self.g_level_slider.get_current_value(),
            self.b_level_slider.get_current_value(),
        ], dtype=np.float32)
        image *= channel_levels

        if self.r_level_inv.get_state():
            image[:, :, 0] = 255 - image[:, :, 0]
        if self.g_level_inv.get_state():
            image[:, :, 1] = 255 - image[:, :, 1]
        if self.b_level_inv.get_state():
            image[:, :, 2] = 255 - image[:, :, 2]

        contrast = self.contrast_slider.get_current_value()
        image = 256 / (1 + np.exp(contrast * (1 / 2 - (image / 256.0))))
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)

        if self.mirror_h.get_state():
            image = image[::-1, :, :]
        if self.mirror_v.get_state():
            image = image[:, ::-1, :]

        line = math.floor(
            self.line_slider_for_graph.get_current_value()
            * (self.image.get_height() - 1)
        )
        self.ax_line.clear()
        self.ax_line.plot(self.grayscale_image[:, line])
        self.ax_line.grid(True)
        self.ax_line.set_ylim(0, 255)
        self.plot_line.draw()
        self.plot_line_surf = self._make_plot_surface()

        return pg.surfarray.make_surface(np.ascontiguousarray(image))

    def draw(self, screen, display_image):
        screen.fill((30, 30, 30))
        screen.blit(
            self.font.render("Источники цветов (каналы)", True, (255, 255, 255)),
            (30, 10),
        )
        screen.blit(
            self.font.render("Количество цветов", True, (255, 255, 255)),
            (30, 65),
        )
        screen.blit(
            self.font.render("Контрастность", True, (255, 255, 255)),
            (300, 10),
        )
        screen.blit(display_image, self.image_rect)

        cur_line = round(
            self.line_slider_for_graph.get_current_value()
            * self.image.get_height()
        )
        screen.blit(
            self.font.render(f"Строка {cur_line}", True, (255, 255, 255)),
            (30, 120),
        )

        mouse_pos = pg.mouse.get_pos()
        if self.line_slider_for_graph.rect.collidepoint(mouse_pos):
            pg.draw.rect(
                screen,
                (200, 200, 200, 128),
                (
                    self.image_rect.left,
                    self.image_rect.top + cur_line - 1,
                    self.image.get_width(),
                    3,
                ),
            )

        screen.blit(self.plot_line_surf, (self.screen_size[0] - 330, 120))
        self.manager.draw_ui(screen)
