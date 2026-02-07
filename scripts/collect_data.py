"""Collect demonstration data from Meta-World MT1 environments using expert policies"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import time
import numpy as np
import gymnasium as gym
import metaworld
from metaworld.policies import ENV_POLICY_MAP
from utils.tokenizer import SimpleTokenizer

import yaml
import imageio
from tqdm import trange 


def save_video(images, output_path, fps=30):
    imageio.mimsave(output_path, images, fps=fps) 
    print(f"Saved video to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_collection.yaml",
                        help="Path to the config file")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    with open(args.config, "r") as f:
        cfg = yaml.full_load(f)
    
    mw_args = cfg["meta_world_args"]
    # Extract top-level args
    episodes = cfg["episodes"]
    max_steps = cfg["max_steps"]
    output_path = cfg["output_path"]
    sleep = cfg.get("sleep", 0.0)
    instruction = cfg["instruction"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    env = gym.make(
        cfg["benchmark"],
        **mw_args
    )

    obs, info = env.reset(seed=mw_args["seed"])
    policy = ENV_POLICY_MAP[mw_args["env_name"]]()

    images = []
    states = []
    actions = []
    texts = []

    for ep in trange(episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # expert policy action on raw obs
            action = policy.get_action(obs)  # shape (action_dim,)

            # log current transition
            img = env.render() # (H, W, 3) uint8
            state = np.asarray(obs, dtype=np.float32).ravel() # (state_dim,)

            images.append(img.copy())
            states.append(state.copy())
            actions.append(np.asarray(action, dtype=np.float32).copy())
            texts.append(instruction)

            # step env
            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1

            if sleep > 0:
                time.sleep(sleep)
    env.close()

    # stack arrays
    images = np.stack(images, axis=0)   # (N, H, W, 3)
    states = np.stack(states, axis=0)   # (N, state_dim)
    actions = np.stack(actions, axis=0) # (N, action_dim)

    # tokenize instructions
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids[i, :len(seq)] = np.array(seq, dtype=np.int64)

    np.savez_compressed(
        output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=tokenizer.vocab,
    )

    print("Saved Meta-World push dataset to", output_path)
    print("  images:", images.shape)
    print("  states:", states.shape)
    print("  actions:", actions.shape)
    print("  text_ids:", text_ids.shape)

    save_video(images, output_path.replace(".npz", ".mp4"))


if __name__ == "__main__":
    main()
