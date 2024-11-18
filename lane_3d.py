from ursina import *
from random import choice, randint

app = Ursina()

# Player setup
player = Entity(model='cube', color=color.orange, scale=(0.5, 0.5, 0.5), position=(0, -2, -15))
player_velocity = 0  # For jumping mechanics

# Ground setup
lane_positions = [-1, 0, 1]  # Three lanes: left, middle, right
lane_blocks = []

# Function to create lanes with gaps
def create_lanes():
    z_pos = -15  # Start lanes closer to the player
    for _ in range(40):  # Create 40 blocks initially (longer lanes)
        for lane in lane_positions:
            if randint(0, 7) > 0 or z_pos < -5:  # Ensure blocks exist at the start
                lane_block = Entity(
                    model='cube',
                    color=color.azure,
                    scale=(0.95, 0.1, 3),  # Extend the length of each lane block
                    position=(lane, -2.5, z_pos)
                )
                lane_blocks.append(lane_block)
        z_pos += 3  # Increase distance between blocks to give time for jumps

# Call to create lanes initially
create_lanes()

# Obstacle setup
obstacles = []


def create_obstacle():
    lane = choice(lane_positions)
    obstacle = Entity(
        model='cube',
        color=color.magenta,
        scale=(0.5, 0.5, 0.5),
        position=(lane, -2, 20)
    )
    obstacles.append(obstacle)


# Movement logic
def update():
    global player_velocity

    # Player movement between lanes
    if held_keys['left arrow']:
        player.x = max(player.x - 0.1, -1)

    if held_keys['right arrow']:
        player.x = min(player.x + 0.1, 1)

    # Jumping
    if held_keys['space'] and player.y <= -2:  # Allow jump only if player is on the ground
        player_velocity = 0.25  # Stronger upward velocity for longer jumps

    # Apply gravity
    player.y += player_velocity
    player_velocity -= 0.01  # Gravity effect

    # Prevent falling through ground
    if player.y < -2:
        # Check if the player is on or near a lane block
        is_on_lane = any(
            abs(block.x - player.x) < 0.8  # Allow a margin for moving between lanes
            and abs(block.z - player.z) < 1.5
            for block in lane_blocks
        )
        if is_on_lane:
            player.y = -2
            player_velocity = 0
        else:
            # print("Game Over!")
            # application.quit()  # Quit the game when falling off

    # Move obstacles
    for obstacle in obstacles:
        obstacle.z -= 0.1  # Move obstacle toward the player

        # Check collision
        if abs(obstacle.x - player.x) < 0.5 and abs(obstacle.z - player.z) < 0.5:
            print("Game Over!")
            # application.quit()

        # Remove obstacles that go off-screen
        if obstacle.z < -20:
            obstacles.remove(obstacle)
            destroy(obstacle)

    # Move lane blocks to simulate infinite scrolling
    for block in lane_blocks:
        block.z -= 0.1
        if block.z < -40:
            block.z += 120  # Recycle block to the end


# Schedule periodic obstacle creation
obstacle_timer = Sequence(
    Func(create_obstacle), Wait(2),  # Increase delay between obstacles
    loop=True
)
obstacle_timer.start()

# Camera settings
camera.position = (0, 4, -30)  # Move camera higher and further back
camera.rotation_x = 20    # Tilt camera downward to show the road ahead

app.run()
