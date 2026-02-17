#!/usr/bin/env python3
"""
A simple script to play ALE (Arcade Learning Environment) games using Gymnasium.

Controls:
-   Movement: Arrows / WASD
-   Fire: Space / Enter
-   Reset: R
-   Quit: ESCAPE

Requires `gymnasium[atari]`, `ale-py`, `pygame`, and `numpy`.
"""

import argparse
import sys
import pygame
import numpy as np
import gymnasium as gym
import ale_py


# --- Constants ---
UPSCALE_FACTOR = 4
NATIVE_H, NATIVE_W = 210, 160
SCALED_W = NATIVE_W * UPSCALE_FACTOR
SCALED_H = NATIVE_H * UPSCALE_FACTOR

COLOR_WHITE = (255, 255, 255)
COLOR_BG = (20, 20, 20)


def setup_ale_env(game_name: str, seed: int) -> gym.Env:
    """Initializes the Gymnasium ALE environment."""
    print(f"Initializing ALE env: 'ALE/{game_name}-v5'")
    env = gym.make(
        f"ALE/{game_name}-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="rgb_array",
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def get_pygame_action(ale_env: gym.Env) -> int:
    """
    Reads keyboard input and returns the corresponding ALE action integer.
    """
    keys = pygame.key.get_pressed()
    
    # Get the meaning of each action
    action_meanings = ale_env.unwrapped.get_action_meanings()
    
    # Create reverse mapping: meaning -> action_id
    meaning_to_action = {meaning: i for i, meaning in enumerate(action_meanings)}
    
    # Determine which action to take based on keys
    fire = keys[pygame.K_SPACE] or keys[pygame.K_RETURN]
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    
    # Build action string based on inputs
    action_str = ""
    if up and right and fire:
        action_str = "UPRIGHTFIRE"
    elif up and left and fire:
        action_str = "UPLEFTFIRE"
    elif down and right and fire:
        action_str = "DOWNRIGHTFIRE"
    elif down and left and fire:
        action_str = "DOWNLEFTFIRE"
    elif up and right:
        action_str = "UPRIGHT"
    elif up and left:
        action_str = "UPLEFT"
    elif down and right:
        action_str = "DOWNRIGHT"
    elif down and left:
        action_str = "DOWNLEFT"
    elif up and fire:
        action_str = "UPFIRE"
    elif down and fire:
        action_str = "DOWNFIRE"
    elif left and fire:
        action_str = "LEFTFIRE"
    elif right and fire:
        action_str = "RIGHTFIRE"
    elif fire:
        action_str = "FIRE"
    elif up:
        action_str = "UP"
    elif down:
        action_str = "DOWN"
    elif left:
        action_str = "LEFT"
    elif right:
        action_str = "RIGHT"
    else:
        action_str = "NOOP"
    
    # Return the corresponding action ID
    return meaning_to_action.get(action_str, 0)


def main():
    parser = argparse.ArgumentParser(description="Play an ALE game.")
    parser.add_argument(
        "-g", "--game",
        type=str,
        required=True,
        help="Name of the game (e.g., 'Pong', 'Breakout', 'Darkchambers')."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the environment."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for playback."
    )
    args = parser.parse_args()
    
    # Capitalize game name for ALE
    game_name = args.game.capitalize()
    
    # Setup
    pygame.init()
    screen = pygame.display.set_mode((SCALED_W, SCALED_H))
    pygame.display.set_caption(f"ALE - {game_name}")
    clock = pygame.time.Clock()
    
    # Initialize ALE environment
    try:
        ale_env = setup_ale_env(game_name, args.seed)
    except Exception as e:
        print(f"Error initializing ALE environment: {e}")
        sys.exit(1)
    
    # Print available actions
    print(f"\nAvailable actions for {game_name}:")
    for i, meaning in enumerate(ale_env.unwrapped.get_action_meanings()):
        print(f"  {i}: {meaning}")
    print()
    
    # Game loop
    running = True
    obs, info = ale_env.reset()
    total_reward = 0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = ale_env.reset()
                    total_reward = 0
                    print("Environment reset")
        
        # Get action from keyboard
        action = get_pygame_action(ale_env)
        
        # Step environment
        obs, reward, terminated, truncated, info = ale_env.step(action)
        total_reward += reward
        
        # Reset if episode ended
        if terminated or truncated:
            print(f"Episode ended. Total reward: {total_reward}")
            obs, info = ale_env.reset()
            total_reward = 0
        
        # Render
        frame = obs  # obs is already the RGB array
        # Convert to pygame surface
        frame_surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )
        # Scale up
        scaled_surface = pygame.transform.scale(
            frame_surface, (SCALED_W, SCALED_H)
        )
        
        # Display
        screen.fill(COLOR_BG)
        screen.blit(scaled_surface, (0, 0))
        
        # Display info
        font = pygame.font.Font(None, 24)
        info_text = font.render(
            f"Reward: {total_reward:.0f}", True, COLOR_WHITE
        )
        screen.blit(info_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(args.fps)
    
    ale_env.close()
    pygame.quit()
    print("Game closed.")


if __name__ == "__main__":
    main()
