import pygame
import pygame_gui
from PIL import Image

from pathlib import Path
# =================================  НА КНОПКУ "0" ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ  ==========================
from channels_mode.channels_mode import channels_mode
from effects_mode.effects_mode import effects_mode
from cont_map_mode.cont_map_mode import cont_map_mode

IMAGE = "img.png"
ZOOM_SCALE = 12
BASE_DIR = Path(__file__).resolve().parent

with Image.open(IMAGE) as img:
    SIZE_SCREEN = (max(800, img.width + 10), max(800, img.height + 120))

pygame.init()
pygame.display.set_caption("python gaming 4k")
screen = pygame.display.set_mode(SIZE_SCREEN)

channel_mode_manager = pygame_gui.UIManager(SIZE_SCREEN, BASE_DIR / "theme.json")
effects_mode_manager = pygame_gui.UIManager(SIZE_SCREEN, BASE_DIR / "theme.json")
cont_map_mode_manager = pygame_gui.UIManager(SIZE_SCREEN, BASE_DIR / "theme.json")

# convert_alpha замена прозрачных частей на черный
main_image = pygame.image.load(BASE_DIR / IMAGE).convert_alpha()
display_image = main_image.copy()
image_rect = main_image.get_rect(topleft=(5, 180))
rgb = pygame.surfarray.array3d(main_image)
print(f"rgb shape: {rgb.shape}")
mean_rgb = rgb.mean(axis=(0, 1))
mean_all = rgb.mean()

clock = pygame.time.Clock()
font = pygame.font.Font()

channel_obj = channels_mode(
    SIZE_SCREEN,
    main_image,
    image_rect,
    font,
    ZOOM_SCALE,
)
channel_obj.get_ui(channel_mode_manager, main_image)

effects_obj = effects_mode(
    effects_mode_manager,
    SIZE_SCREEN,
    main_image,
    image_rect,
    font,
    rgb,
)

cont_map_obj = cont_map_mode(cont_map_mode_manager, font, image_rect)

show_modes = [channel_obj.name, effects_obj.name, cont_map_obj.name]
cur_show_mode = channel_obj.name


def update_image():
    global display_image
    if cur_show_mode == channel_obj.name:
        display_image = channel_obj.update_image_with_channels(rgb)
    elif cur_show_mode == effects_obj.name:
        display_image = effects_obj.update_image(rgb)
    elif cur_show_mode == cont_map_obj.name:
        display_image = cont_map_obj.update_image(rgb)


# Основной цикл отрисовки
running = True
while running:
    time_delta = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        channel_mode_manager.process_events(event)
        effects_mode_manager.process_events(event)
        cont_map_mode_manager.process_events(event)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0:
                cur_show_mode = show_modes[(show_modes.index(cur_show_mode) + 1) % 3]
                update_image()
        elif event.type in [
            pygame_gui.UI_CHECK_BOX_CHECKED,
            pygame_gui.UI_CHECK_BOX_UNCHECKED,
            pygame_gui.UI_DROP_DOWN_MENU_CHANGED,
            pygame_gui.UI_HORIZONTAL_SLIDER_MOVED,
        ]:
            update_image()
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element in [
                channel_obj.save_button,
                effects_obj.save_button,
                cont_map_obj.save_button,
            ]:
                pygame.image.save(display_image, "output_image.png")
        channel_obj.process_event(event)

    if cur_show_mode == channel_obj.name:
        channel_obj.draw_channels(screen)
        channel_mode_manager.update(time_delta)
    elif cur_show_mode == effects_obj.name:
        effects_obj.draw(screen, display_image)
        effects_mode_manager.update(time_delta)
    elif cur_show_mode == cont_map_obj.name:
        cont_map_obj.draw(screen, display_image)
        cont_map_mode_manager.update(time_delta)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()