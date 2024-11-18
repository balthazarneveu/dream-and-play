from ursina import *
from random import choice, randint

app = Ursina()

# Player setup
PLAYER_STANDARD_SCALE = 0.5
GROUND = -2.5  # Ground level
player = Entity(model='cube', color=color.orange,
                scale=(0.5, PLAYER_STANDARD_SCALE, 0.5), position=(0, 5, -15))
player_velocity = -0.1  # For initial fall
is_falling = True  # Start the game with a fall

# Ground setup
lane_positions = [-1, 0, 1]  # Three lanes: left, middle, right
lane_blocks = []

# Score setup
score = Text(text="Score: 0", position=(-0.85, 0.45), scale=1.5)
points = 0  # Player score

# Function to create lanes with gaps


def create_lanes():
    z_pos = -15  # Start lanes closer to the player
    for _ in range(40):  # Create 40 blocks initially (longer lanes)
        for lane in lane_positions:
            if randint(0, 7) > 0 or z_pos < -5:  # Ensure blocks exist at the start
                lane_thickness = 0.1
                lane_block = Entity(
                    model='cube',
                    color=color.azure,
                    # Extend the length of each lane block
                    scale=(0.97, lane_thickness, 3),
                    position=(lane, GROUND - lane_thickness/2, z_pos)
                )
                lane_blocks.append(lane_block)
        z_pos += 3  # Increase distance between blocks to give time for jumps


# Call to create lanes initially
create_lanes()

# Obstacle setup
obstacles = []


# Obstacle setup
# Obstacle setup
def create_obstacle():
    lane = choice(lane_positions)
    size = choice(['small', 'tall', 'low'])  # Add low obstacles
    if size == 'small':
        height = 0.5  # Small obstacles (can be jumped over)
        offset = 0  # No offset for small obstacles
    elif size == 'tall':
        height = 2.5  # Tall obstacles (cannot be jumped over)
        offset = 0  # No offset for small obstacles
    elif size == 'low':
        height = 3.  # Low obstacles (require crouching)
        offset = 0.3  # No offset for small obstacles

    obstacle = Entity(
        model='cube',
        # Different colors
        color=color.magenta if size == 'small' else color.red if size == 'tall' else color.violet,
        scale=(0.5, height, 0.5),  # Adjust height
        # Position the obstacle properly on the ground
        position=(lane, GROUND + offset + height / 2, 20)
    )
    obstacle.size = size  # Store the size for collision logic
    obstacles.append(obstacle)


# Reset player after falling
def reset_player():
    global is_falling, player_velocity
    print("Returning player to the nearest valid lane.")

    # Find the nearest lane that also has a block beneath
    valid_lanes = [
        lane for lane in lane_positions
        if any(
            abs(block.x - lane) < 0.8
            and abs(block.z - player.z) < 1.5
            for block in lane_blocks
        )
    ]

    if valid_lanes:
        nearest_lane = min(valid_lanes, key=lambda lane: abs(player.x - lane))
    else:
        nearest_lane = 0  # Default to the middle lane if no valid lanes are found

    # Reset player position to the nearest valid lane
    player.position = (nearest_lane, GROUND + player.scale_y / 2, player.z)
    player_velocity = 0  # Reset velocity
    is_falling = False  # Stop falling

# Movement logic


def update():
    global player_velocity, points, is_falling

    if is_falling:
        # If the player is falling, continue to move them downward
        player.y += player_velocity
        player_velocity -= 0.01  # Gravity effect

        # Check if they've landed on the track
        if player.y <= GROUND + player.scale_y / 2:
            # Check if they are on a lane block
            is_on_lane = any(
                abs(block.x - player.x) < 0.8
                and abs(block.z - player.z) < 1.5
                for block in lane_blocks
            )
            if is_on_lane:
                player.y = GROUND + player.scale_y / 2  # Adjust player to be on the ground
                player_velocity = 0
                is_falling = False  # Stop falling once landed
            else:
                # Reset after falling for a bit
                invoke(reset_player, delay=0.5)
        return  # Skip further updates while falling

    # Player movement between lanes
    if held_keys['left arrow']:
        player.x = max(player.x - 0.1, -1)

    if held_keys['right arrow']:
        player.x = min(player.x + 0.1, 1)

    # Jumping
    if held_keys['space'] and player.y <= -2:  # Allow jump only if player is on the ground
        player_velocity = 0.25  # Stronger upward velocity for longer jumps

    # Crouching
    # if held_keys['down arrow']:

    #     player.scale_y = 0.25  # Shrink player for crouch
    # else:
    #     player.scale_y = 0.5  # Reset player to normal size

    # Crouching
    if held_keys['down arrow']:
        new_scale_y = 0.25  # Crouching scale
        if player.scale_y != new_scale_y:  # Only adjust if the scale changes
            # Adjust y-position to keep feet on the ground
            player.y -= (player.scale_y - new_scale_y) / 2
        player.scale_y = new_scale_y  # Shrink player for crouch
    else:
        new_scale_y = 0.5  # Normal scale
        if player.scale_y != new_scale_y:  # Only adjust if the scale changes
            # Adjust y-position to keep feet on the ground
            player.y += (new_scale_y - player.scale_y) / 2
        player.scale_y = new_scale_y  # Reset player to normal size

    # Apply gravity
    player.y += player_velocity
    player_velocity -= 0.01  # Gravity effect

    # Prevent falling through ground
    if player.y < -2.5 + player.scale_y / 2:
        # Check if the player is on or near a lane block
        is_on_lane = any(
            abs(block.x - player.x) < 0.8
            and abs(block.z - player.z) < 1.5
            for block in lane_blocks
        )
        if is_on_lane:
            player.y = -2.5 + player.scale_y / 2  # Adjust player to be on the ground
            player_velocity = 0
        else:
            print("Player falling off!")
            is_falling = True  # Enable falling
            player_velocity = -0.1  # Start falling down

    # Move obstacles and check for collisions
    for obstacle in obstacles:
        obstacle.z -= 0.1  # Move obstacle toward the player

        # Check collision based on obstacle height
        if (
            abs(obstacle.x - player.x) < 0.5 and  # Horizontal proximity
            abs(obstacle.z - player.z) < 0.5 and  # Depth proximity
            
            player.y < obstacle.y + obstacle.scale_y / 2 and  # Vertical proximity
            # Vertical proximity under
            player.y + player.scale_y / 2 > obstacle.y - obstacle.scale_y / 2
            # Check crouch for low obstacles
        ):
            print(
                f"Hit an obstacle! (Height: {obstacle.scale_y}) Losing points.")
            points -= 1
            score.text = f"Score: {points}"

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
