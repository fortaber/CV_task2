import numpy as np
import pygame as pg
import pygame_gui as pgui
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import kernel_operations


class channels_mode:
    def __init__(self, screen_size, main_image, image_rect, font, zoom_scale):

        # ====== Гистограммы =========
        pixels = pg.surfarray.pixels3d(main_image)

        # RGB каналы
        red = np.zeros_like(pixels, dtype=np.uint8)
        red[:, :, 0] = pixels[:, :, 0]

        green = np.zeros_like(pixels, dtype=np.uint8)
        green[:, :, 1] = pixels[:, :, 1]

        blue = np.zeros_like(pixels, dtype=np.uint8)
        blue[:, :, 2] = pixels[:, :, 2]

        del pixels

        hist_red_surf = self.create_hist(red[:, :, 0], 'red')
        hist_green_surf = self.create_hist(green[:, :, 1], 'green')
        hist_blue_surf = self.create_hist(blue[:, :, 2], 'blue')

        self.name = "channels_mode"
        self.screen_size = screen_size
        self.main_image = main_image
        self.image_rect = image_rect
        self.hist_surfaces = (hist_red_surf, hist_green_surf, hist_blue_surf)
        self.font = font
        self.zoom_scale = zoom_scale
        self.display_image = main_image.copy()
        self.slider_value = 0
        self.win_size = 1
        self.manager = None


    def create_hist(self, img_channel, color='blue'):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
        ax.hist(img_channel.flatten(), color=color, bins=256, density=True)
        ax.grid(True)
        hist = FigureCanvasAgg(fig)
        hist.draw()
        hist_surf = pg.image.frombytes(
            hist.buffer_rgba().tobytes(),
            hist.get_width_height(),
            "RGBA",
        )
        return hist_surf
    

    def get_ui(self, mgr: pgui.UIManager, img: pg.surface.Surface):
        self.manager = mgr
        color_modes = ["Original", "Red", "Blue", "Green", "Gray"]

        container = pgui.core.UIContainer(
            relative_rect=pg.Rect((20, 20), (600, 300)),
            manager=mgr,
        )

        self.modes_button = pgui.elements.UIDropDownMenu(
            options_list=color_modes,
            starting_option=color_modes[0],
            relative_rect=pg.Rect((20, 30), (100, 30)),
            container=container,
            manager=mgr,
        )

        self.compare_slider = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((140, 30), (250, 30)),
            start_value=self.slider_value,
            value_range=(0, img.get_width()),
            container=container,
            manager=mgr,
        )

        self.d_slider = pgui.elements.UIHorizontalSlider(
            relative_rect=pg.Rect((140, 0), (100, 30)),
            start_value=self.win_size,
            value_range=(1, 10),
            container=container,
            manager=mgr,
        )

        self.four_checkbox = pgui.elements.UICheckBox(
            text="4-связность",
            relative_rect=pg.Rect((390, 30), (30, 30)),
            container=container,
            manager=mgr,
        )

        self.ate_checkbox = pgui.elements.UICheckBox(
            text="8-связность",
            relative_rect=pg.Rect((510, 30), (30, 30)),
            container=container,
            manager=mgr,
        )

        self.HandmadeFilter_checkbox = pgui.elements.UICheckBox(
            text="weird filter",
            relative_rect=pg.Rect((390, 2), (30, 30)),
            container=container,
            manager=mgr,
        )

        self.save_button = pgui.elements.UIButton(
            relative_rect=pg.Rect((510, 70), (110, 30)),
            text="Save image",
            container=container,
            manager=mgr,
        )

    def process(self, rgb_image, mode):
        if mode == "Gray":
            gray_image = np.dot(
                rgb_image.astype(np.float32)[..., :3],
                [0.2989, 0.5870, 0.1140],
            )
            gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
            return np.repeat(gray_image[:, :, np.newaxis], 3, axis=2)

        rgb_image = rgb_image.astype(np.uint8)
        if mode == "Red":
            rgb_image[:, :, 1:] = 0
        elif mode == "Green":
            rgb_image[:, :, [0, 2]] = 0
        elif mode == "Blue":
            rgb_image[:, :, :2] = 0
        return rgb_image

    def update_image_with_channels(self, image):
        rgb_filtered = image.copy()
        if self.four_checkbox.get_state():
            rgb_filtered = kernel_operations.the_four_connectivity(image)
        elif self.ate_checkbox.get_state():
            rgb_filtered = kernel_operations.the_eight_connectivity(image)
        elif self.HandmadeFilter_checkbox.get_state():
            rgb_filtered = kernel_operations.drunk_filter(image, self.win_size)

        x = int(self.slider_value)
        rgb_filtered[:x] = self.process(
            rgb_filtered[:x],
            self.modes_button.selected_option[0],
        )
        self.display_image = pg.surfarray.make_surface(rgb_filtered).convert_alpha()
        return self.display_image

    def draw_channels(self, screen: pg.surface.Surface):
        screen.fill((30, 30, 30))
        pg.draw.rect(screen, (50, 50, 70), (0, 0, self.screen_size[0], 120))
        screen.blit(self.display_image, self.image_rect)

        screen.blit(self.hist_surfaces[0], (self.screen_size[0] - 250, 150))
        screen.blit(self.hist_surfaces[1], (self.screen_size[0] - 250, 350))
        screen.blit(self.hist_surfaces[2], (self.screen_size[0] - 250, 550))

        mouse_x, mouse_y = pg.mouse.get_pos()
        if self.image_rect.collidepoint(mouse_x, mouse_y):
            image_x = mouse_x - self.image_rect.x
            image_y = mouse_y - self.image_rect.y
            color = self.display_image.get_at((image_x, image_y))

            area = pg.Rect(0, 0, self.zoom_scale, self.zoom_scale)
            area.center = (image_x, image_y)
            area = area.clip(self.main_image.get_rect())

            r_channel = np.array(0)
            g_channel = np.array(0)
            b_channel = np.array(0)
            s_red = 0
            s_green = 0
            s_blue = 0
            if area.width > 0 and area.height > 0:
                piece = self.display_image.subsurface(area).copy()
                rgb_piece = pg.surfarray.array3d(piece)
                r_channel = rgb_piece[:, :, 0]
                g_channel = rgb_piece[:, :, 1]
                b_channel = rgb_piece[:, :, 2]

                avg_r = r_channel.sum() / r_channel.size
                avg_g = g_channel.sum() / g_channel.size
                avg_b = b_channel.sum() / b_channel.size

                s_red = ((r_channel - avg_r) ** 2).sum() / r_channel.size
                s_green = ((g_channel - avg_g) ** 2).sum() / g_channel.size
                s_blue = ((b_channel - avg_b) ** 2).sum() / b_channel.size

                piece = pg.transform.scale(
                    piece, (area.width * self.zoom_scale, area.height * self.zoom_scale)
                )
                screen.blit(piece, (self.screen_size[0] - 170, 5))
                pg.draw.rect(
                    screen,
                    (255, 255, 255),
                    (*(self.screen_size[0] - 170, 5), *piece.get_size()),
                    2,
                )

            text = self.font.render(
                f"R: {color.r} G: {color.g} B: {color.b}",
                True,
                (255, 255, 255),
            )
            screen.blit(text, (10, 10))
            slider_text = self.font.render(
                f"Координаты p = {image_x, image_y} | Интенсивность : "
                f"{round((color.r + color.g + color.b) / 3, 2)}\n"
                f"μ_Wp(RED) = {round(r_channel.sum() / r_channel.size, 2)}, "
                f"μ_Wp(GREEN) = {round(g_channel.sum() / g_channel.size, 2)}, "
                f"μ_Wp(BLUE) = {round(b_channel.sum() / b_channel.size, 2)}\n"
                f"S_Wp(RED) = {round(s_red, 2)}, "
                f"S_Wp(GREEN) = {round(s_green, 2)}, "
                f"S_Wp(BLUE) = {round(s_blue, 2)},",
                True,
                (255, 255, 255),
            )
            screen.blit(slider_text, (10, 70))

        self.manager.draw_ui(screen)


    def process_event(self, event):
        if event.type == pgui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.compare_slider:
                self.slider_value = event.value
            elif event.ui_element == self.d_slider:
                self.win_size = int(event.value)
