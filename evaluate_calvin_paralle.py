# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Code to evaluate Calvin."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import copy
# from moviepy.editor import ImageSequenceClip

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

import tianshou
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRawGymnasium

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
# from pytorch_lightning import seed_everything
# from termcolor import colored
import torch
from tqdm.auto import tqdm

from evaluation.calvin_evaluation import DummyCalvinEvaluation, GR1CalvinEvaluation

from utils.calvin_utils import print_and_save

logger = logging.getLogger(__name__)

# Path to calvin
CALVIN_ROOT = os.environ['CALVIN_ROOT']

EP_LEN = 360
NUM_SEQUENCES = 50


def make_env(dataset_path, observation_space, device_id, env_idx=-1):
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    device = torch.device('cuda', device_id)
    env = CalvinEnvWrapperRawGymnasium(val_folder, observation_space, device, env_idx=env_idx)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, eval_dir=None, debug=False):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    num_envs = env.env_num
    sequence_i = 0
    sequence_i_pack = []
    initial_state_pack = []
    eval_sequence_pack = []
    for initial_state, eval_sequence in eval_sequences:
        # result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i)
        # results.append(result)
        sequence_i_pack.append(sequence_i)
        initial_state_pack.append(initial_state)
        eval_sequence_pack.append(eval_sequence)

        if len(sequence_i_pack) != num_envs:
            sequence_i += 1
            continue

        result = evaluate_sequences(env, model, task_oracle, initial_state_pack, eval_sequence_pack, val_annotations, debug, eval_dir, sequence_i_pack)
        results += result
        sequence_i_pack = []
        initial_state_pack = []
        eval_sequence_pack = []
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{NUM_SEQUENCES}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequences(env, model, task_checker, initial_states, eval_sequences, val_annotations, debug, eval_dir, sequence_is):
    initial_obs = [get_env_state_for_initial_condition(initial_state) for initial_state in initial_states]
    robot_obses, scene_obses = [item[0] for item in initial_obs], [item[1] for item in initial_obs]
    obs, current_info = env.reset(robot_obs=robot_obses, scene_obs=scene_obses)
    obs = tianshou.data.Batch(obs)
    success_counter = [0 for _ in initial_states]
    terminate_flag = [False for _ in initial_states]
    if debug:
        time.sleep(1)
        print()
        print()
        for eval_sequence in eval_sequences:
            print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtasks in enumerate(zip(*eval_sequences)):

        if debug:
            for subtask in subtasks:
                print(f"{subtask} ", end="")
            time.sleep(0.5)

        start_info = current_info
        lang_annotations = [val_annotations[subtask][0] for subtask in subtasks]
        model.reset()
        assert env.env_num == len(subtasks)
        num_envs = env.env_num
        done = [False] * num_envs
        successes = [False] * num_envs

        if debug:
            img_lists = [[] for _ in range(num_envs)]

        for step in range(EP_LEN):
            actions = model.step(obs, lang_annotations)
            obs, _, _, _, current_info = env.step(actions)
            obs = tianshou.data.Batch(obs)

            for i in range(num_envs):
                if not done[i]:
                    current_task_info = task_checker.get_task_info_for_set(start_info[i], current_info[i], {subtasks[i]})
                    if len(current_task_info) > 0:
                        successes[i] = True
                        done[i] = True
                    if debug:
                        img_copy = copy.deepcopy(obs[i]['rgb_obs']['rgb_static'])
                        img_lists[i].append(img_copy)
            if all(done):
                break

        for i, success in enumerate(successes):
            if not terminate_flag[i]:
                if success:
                    success_counter[i] += 1
                else:
                    terminate_flag[i] = True
        if all(terminate_flag):
            break
    return success_counter


def main():
    # seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_dir", default='fake_dataset/', type=str, help="Dataset directory.")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument('--eval_dir', default='logs', type=str, help="Directory to save evaluation results")
    parser.add_argument('--policy_ckpt_path', type=str, help="Path to the evaluating checkpoint")
    parser.add_argument('--mae_ckpt_path', type=str, help="Path to the MAE checkpoint")
    parser.add_argument('--configs_path', type=str, help="Path to the config json file")
    parser.add_argument('--device', default=0, type=int, help="CUDA device")
    parser.add_argument('--parallel', default=2, type=int, help="Environment number for parallel")
    args = parser.parse_args()

    if args.configs_path and args.mae_ckpt_path and args.policy_ckpt_path:
        with open(args.configs_path, "r") as f:
            variant = json.load(f)
        device = torch.device('cuda', args.device)
        model = GR1CalvinEvaluation(
            mae_ckpt=args.mae_ckpt_path,
            policy_ckpt=args.policy_ckpt_path,
            variant=variant,
            device=device)
    else:
        model = DummyCalvinEvaluation()
        print("\n\n\n\n\nWarning: Using Dummy Model\n\n\n\n\n")

    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language']}

    def env_initializer(dataset_path, observation_space, device_id, env_idx):
        def make_single_env():
            env = make_env(dataset_path, observation_space, device_id, env_idx)
            return env

        return make_single_env

    def make_env_initializers(dataset_path, observation_space, num_envs, start_device_id=0):
        envs = []
        for i in range(num_envs):
            envs.append(env_initializer(dataset_path, observation_space, start_device_id + 0, i))
        return envs

    num_parallel_envs = args.parallel
    assert NUM_SEQUENCES % num_parallel_envs == 0
    initializers = make_env_initializers(args.dataset_dir, observation_space, num_parallel_envs)
    env = SubprocVectorEnv(initializers)
    # env = make_env(args.dataset_dir, observation_space, args.device)

    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir, exist_ok=True)
    eval_sr_path = os.path.join(args.eval_dir, "success_rate.txt")
    eval_result_path = os.path.join(args.eval_dir, "result.txt")
    evaluate_policy(
        model, 
        env,
        eval_sr_path, 
        eval_result_path, 
        args.eval_dir,
        debug=args.debug)


if __name__ == "__main__":
    main()