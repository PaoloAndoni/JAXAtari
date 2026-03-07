#!/usr/bin/env python3
"""
A script to play an ALE (Gymnasium) Atari game interactively via pygame.

Controls:
-   Movement: Arrows / WASD
-   Fire: Space / Enter
-   Pause: P
-   Frame-by-Frame Toggle: F
-   Next Frame: N (when frame-by-frame is on)
-   Reset: R
-   Quit: ESCAPE

Requires: gymnasium[atari], ale-py, pygame, numpy.
"""

import argparse
import sys
import time

import pygame
import numpy as np
import gymnasium as gym
import ale_py  # noqa: F401 – required to register ALE envs


# --- Constants ---
UPSCALE_FACTOR = 3
NATIVE_H, NATIVE_W = 210, 160
SCALED_W = NATIVE_W * UPSCALE_FACTOR
SCALED_H = NATIVE_H * UPSCALE_FACTOR

COLOR_WHITE = (255, 255, 255)
COLOR_BG = (20, 20, 20)
COLOR_PAUSE = (255, 255, 0)
COLOR_PLAY = (0, 200, 0)


# --- Key → semantic action mapping (same as gameplay_comparison.py) ---

ALL_SEMANTIC_ACTIONS = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]


def get_semantic_action_from_keys(pressed_keys: pygame.key.ScancodeWrapper) -> str:
    up    = pressed_keys[pygame.K_UP]    or pressed_keys[pygame.K_w]
    down  = pressed_keys[pygame.K_DOWN]  or pressed_keys[pygame.K_s]
    left  = pressed_keys[pygame.K_LEFT]  or pressed_keys[pygame.K_a]
    right = pressed_keys[pygame.K_RIGHT] or pressed_keys[pygame.K_d]
    fire  = pressed_keys[pygame.K_SPACE] or pressed_keys[pygame.K_RETURN]

    if up and right and fire:  return "UPRIGHTFIRE"
    if up and left  and fire:  return "UPLEFTFIRE"
    if down and right and fire: return "DOWNRIGHTFIRE"
    if down and left  and fire: return "DOWNLEFTFIRE"

    if up   and fire: return "UPFIRE"
    if down and fire: return "DOWNFIRE"
    if left and fire: return "LEFTFIRE"
    if right and fire: return "RIGHTFIRE"

    if up and right:   return "UPRIGHT"
    if up and left:    return "UPLEFT"
    if down and right: return "DOWNRIGHT"
    if down and left:  return "DOWNLEFT"

    if fire:  return "FIRE"
    if up:    return "UP"
    if down:  return "DOWN"
    if left:  return "LEFT"
    if right: return "RIGHT"

    return "NOOP"


# --- Environment Setup ---

def setup_ale_env(game_name: str, seed: int) -> gym.Env:
    """Initializes the Gymnasium ALE environment."""
    print(f"Initializing ALE env: 'ALE/{game_name}-v5'")
    try:
        env = gym.make(
            f"ALE/{game_name}-v5",
            render_mode="rgb_array",
            frameskip=1,
            repeat_action_probability=0.0,  # deterministic
        )
        env.reset(seed=seed)
        print("ALE environment initialized.")
        return env
    except Exception as e:
        print(f"Error creating ALE environment: {e}")
        print("Ensure ROMs are installed: pip install gymnasium[accept-rom-license]")
        sys.exit(1)


def build_ale_action_map(env: gym.Env) -> dict:
    """Maps semantic action names → ALE integer action indices."""
    try:
        meanings = env.unwrapped.get_action_meanings()
        action_map = {name: i for i, name in enumerate(meanings)}
        print("\n--- ALE ACTION MAP ---")
        for name in ALL_SEMANTIC_ACTIONS:
            idx = action_map.get(name, "N/A")
            print(f"  {name:<18} -> {idx}")
        print()
        return action_map
    except Exception as e:
        print(f"Warning: could not get ALE action meanings: {e}. Defaulting to NOOP.")
        return {"NOOP": 0}


# --- Main game loop ---

def run(env: gym.Env, action_map: dict, fps: int, seed: int) -> None:
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 18)

    screen = pygame.display.set_mode((SCALED_W, SCALED_H))
    pygame.display.set_caption("ALE Player")
    clock = pygame.time.Clock()

    scaled_size = (SCALED_W, SCALED_H)
    frame = env.render()

    running = True
    pause = False
    frame_by_frame = False
    next_frame_asked = False

    while running:
        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    pause = not pause
                    print(f"Paused: {pause}")
                elif event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
                    pause = False
                    print(f"Frame-by-frame: {frame_by_frame}")
                elif event.key == pygame.K_n:
                    next_frame_asked = True
                elif event.key == pygame.K_r:
                    print("Resetting…")
                    env.reset(seed=seed)
                    frame = env.render()

        # --- Pause / frame-by-frame ---
        if pause or (frame_by_frame and not next_frame_asked):
            label = "PAUSED" if pause else "FRAME-BY-FRAME (N = next)"
            color = COLOR_PAUSE
        else:
            # --- Step ---
            pressed_keys = pygame.key.get_pressed()
            semantic = get_semantic_action_from_keys(pressed_keys)
            ale_action = action_map.get(semantic, 0)

            _obs, _reward, terminated, truncated, _info = env.step(ale_action)
            frame = env.render()

            if terminated or truncated:
                print("Episode ended – resetting…")
                env.reset()
                frame = env.render()

            label = "ALE"
            color = COLOR_PLAY

        if next_frame_asked:
            next_frame_asked = False

        # --- Draw ---
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf_scaled = pygame.transform.scale(surf, scaled_size)
        screen.fill(COLOR_BG)
        screen.blit(surf_scaled, (0, 0))

        text = font.render(label, True, color)
        text_rect = text.get_rect(center=(SCALED_W // 2, SCALED_H - 20))
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    env.close()


# --- Entry point ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Play an ALE Atari game interactively.")
    parser.add_argument(
        "-g", "--game",
        type=str,
        required=True,
        help="Game name, e.g. 'darkchambers', 'pong'. Capitalised automatically for ALE.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fps",  type=int, default=30, help="Playback frame rate.")
    args = parser.parse_args()

    ale_game_name = args.game.capitalize()
    env = setup_ale_env(ale_game_name, args.seed)
    action_map = build_ale_action_map(env)

    print(f"Starting ALE '{ale_game_name}' at {args.fps} FPS (seed={args.seed})…")
    run(env, action_map, args.fps, args.seed)


if __name__ == "__main__":
    main()
