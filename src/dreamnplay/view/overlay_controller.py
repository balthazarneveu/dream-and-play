from pygame_emojis import load_emoji
# Load emojis
size = (40, 40)
hand_emoji = load_emoji('🖐️', size)
keyboard_emoji = load_emoji('⌨️', size)
body_emoji = load_emoji('🕺', size)


# Generic method to display the control mode using an emoji
def display_control_mode(controller, screen, width, height):
    # Display control method emoji
    bottom_live_position = height - 40
    disp_emoji_location = (width - 50, 20)
    if controller.hand_control:
        screen.blit(hand_emoji, disp_emoji_location)
        disp_emoji_location = (
            int(width*controller.current_position), bottom_live_position)
        screen.blit(hand_emoji, disp_emoji_location)
    elif controller.body_control:
        screen.blit(body_emoji, disp_emoji_location)
        disp_emoji_location = (
            int(width*controller.current_position), bottom_live_position)
        screen.blit(body_emoji, disp_emoji_location)
    else:
        screen.blit(keyboard_emoji, disp_emoji_location)
