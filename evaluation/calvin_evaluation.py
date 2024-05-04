# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class for evaluating GR-1 on Calvin Benchmark."""
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image

import clip

import models.vision_transformer as vits
from models.gr1 import GR1

from calvin_agent.models.calvin_base_model import CalvinBaseModel


class DummyCalvinEvaluation(CalvinBaseModel):
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def step(self, obs, goal):
        B = len(obs)
        act_min_bound = [-0.432188, -0.545456, 0.293439, -3.141593, -0.811348, -3.141573, -1.]
        act_max_bound = [0.42977, 0.139396, 0.796262, 3.141592, 0.638583, 3.141551, 1.]
        act_min_bound = torch.tensor(act_min_bound)
        act_max_bound = torch.tensor(act_max_bound)
        random_actions = act_min_bound + (act_max_bound - act_min_bound) * torch.rand(7)
        expanded_random_actions = random_actions.expand(B, -1)  # Expand to the batch size B
        return expanded_random_actions


class GR1CalvinEvaluation(CalvinBaseModel):
    parallel_num = 99
    def __init__(self,
                 mae_ckpt,
                 policy_ckpt,
                 variant,
                 device
    ):
        """Constructor."""
        self.tokenizer = clip.tokenize
        self.variant = variant
        self.seq_len = variant['seq_len']
        self.use_hand_rgb = variant['use_hand_rgb']
        self.act_dim = variant['act_dim']
        self.state_dim = variant['state_dim']
        self.device = device

        # Preprocess
        input_size = (224, 224)
        rgb_mean = (0.485, 0.456, 0.406)
        rgb_std = (0.229, 0.224, 0.225)
        self.preprocess = T.Compose([
            T.Resize(input_size, interpolation=Image.BICUBIC),
            T.Normalize(rgb_mean, rgb_std)])

        # CLIP
        model_clip, _ = clip.load(variant['clip_backbone'], device=self.device)
        
        # MAE
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        model_mae.to(self.device)
        checkpoint = torch.load(mae_ckpt, map_location='cpu')
        model_mae.load_state_dict(checkpoint['model'], strict=False)

        # Resampler hparams
        resampler_params = dict()
        resampler_params['depth'] = variant['resampler_depth']
        resampler_params['dim_head'] = variant['resampler_dim_head']
        resampler_params['heads'] = variant['resampler_heads']
        resampler_params['num_latents'] = variant['resampler_num_latents']
        resampler_params['num_media_embeds'] = variant['resampler_num_media_embeds']
        variant['resampler_params'] = resampler_params

        # GR-1 policy
        self.policy = GR1(
            model_clip=model_clip,
            model_mae=model_mae,
            state_dim=variant['state_dim'],
            act_dim=variant['act_dim'],
            hidden_size=variant['embed_dim'],
            sequence_length=variant['seq_len'],
            training_target=['act_pred'],
            img_feat_dim=variant['img_feat_dim'],
            lang_feat_dim=variant['lang_feat_dim'],
            patch_feat_dim=variant['patch_feat_dim'],
            resampler_params=variant['resampler_params'],
            without_norm_pix_loss=variant['without_norm_pix_loss'],
            use_hand_rgb=variant['use_hand_rgb'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=variant['n_positions'],
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'])
        print(f"loading state dict: {policy_ckpt}...")
        payload = torch.load(policy_ckpt)
        state_dict = payload['state_dict']
        msg = self.policy.load_state_dict(state_dict, strict=False)
        self.policy.to(self.device)
        self.policy.eval()
        self.rgb_list = [[] for _ in range(self.parallel_num)]
        self.hand_rgb_list = [[] for _ in range(self.parallel_num)]
        self.state_list = [[] for _ in range(self.parallel_num)]
        self.rollout_step_counter = 0

    def reset(self, env_idx=None):
        """Reset function."""
        if env_idx is None:
            self.rgb_list = [[] for _ in range(self.parallel_num)]
            self.hand_rgb_list = [[] for _ in range(self.parallel_num)]
            self.state_list = [[] for _ in range(self.parallel_num)]
            self.rollout_step_counter = 0
        else:
            self.rgb_list[env_idx] = []
            self.hand_rgb_list[env_idx] = []
            self.state_list[env_idx] = []

    def step(self, obs, goal):
        """Step function."""
        # Language
        text = goal
        tokenized_text = self.tokenizer(text)

        # RGB
        rgb = obs['rgb_obs']['rgb_static']
        if len(rgb.shape) > 3:
            tmp = []
            rgbs = rgb
            for i, rgb in enumerate(rgbs):
                rgb = Image.fromarray(rgb)
                rgb = T.ToTensor()(rgb.convert("RGB"))
                rgb = self.preprocess(rgb)
                tmp.append(rgb)
                self.rgb_list[i].append(rgb)
        else:
            rgb = Image.fromarray(rgb)
            rgb = T.ToTensor()(rgb.convert("RGB"))
            rgb = self.preprocess(rgb)
            self.rgb_list[0].append(rgb)

        hand_rgb = obs['rgb_obs']['rgb_gripper']
        if len(hand_rgb.shape) > 3:
            tmp = []
            hand_rgbs = hand_rgb
            for i, hand_rgb in enumerate(hand_rgbs):
                hand_rgb = Image.fromarray(hand_rgb)
                hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
                hand_rgb = self.preprocess(hand_rgb)
                tmp.append(hand_rgb)
                self.hand_rgb_list[i].append(rgb)
        else:
            hand_rgb = Image.fromarray(hand_rgb)
            hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
            hand_rgb = self.preprocess(hand_rgb)
            self.hand_rgb_list[0].append(hand_rgb)

        # State
        no_batch = False
        state = torch.from_numpy(obs['robot_obs'])
        if state.dim() == 1:
            state = state.unsqueeze(0)
            no_batch = True
        arm_state = state[:, :6]
        gripper_state = state[:, -1].unsqueeze(1)
        state = torch.cat((arm_state, gripper_state), dim=-1)
        if len(state.shape) > 1:
            b = state.size(0)
            for i, s in enumerate(state):
                self.state_list[i].append(s)
        else:
            b = 1
            self.state_list[0].append(state)

        # Buffer
        for i in range(len(state)):
            buffer_len = len(self.rgb_list[i])
            if buffer_len > self.seq_len:
                self.rgb_list[i].pop(0)
                self.hand_rgb_list[i].pop(0)
                self.state_list[i].pop(0)

                assert len(self.rgb_list[i]) == self.seq_len
                assert len(self.hand_rgb_list[i]) == self.seq_len
                assert len(self.state_list[i]) == self.seq_len

        # Static RGB
        c, h, w = rgb.shape[-3:]
        rgb_data = torch.zeros((b, self.seq_len, c, h, w))
        for i in range(b):
            buffer_len = len(self.rgb_list[i])
            rgb_tensor = torch.stack(self.rgb_list[i], dim=0)  # (b, l, c, h, w)
            rgb_data[i, :buffer_len] = rgb_tensor[:buffer_len]

        # Hand RGB
        c, h, w = hand_rgb.shape[-3:]
        hand_rgb_data = torch.zeros((b, self.seq_len, c, h, w))
        for i in range(b):
            buffer_len = len(self.hand_rgb_list[i])
            hand_rgb_tensor = torch.stack(self.hand_rgb_list[i], dim=0)  # (b, l, c, h, w)
            hand_rgb_data[i, :buffer_len] = hand_rgb_tensor[:buffer_len]

        # State
        # b = state_tensor.shape[0]
        gripper_state_data = -torch.ones((b, self.seq_len)).float()
        for i in range(b):
            buffer_len = len(self.state_list[i])
            gripper_state_data[i, :buffer_len] = torch.stack(self.state_list[i], dim=0)[:, 6]
        gripper_state_data = (gripper_state_data + 1.0) / 2
        gripper_state_data = gripper_state_data.long()
        gripper_state_data = F.one_hot(gripper_state_data, num_classes=2).float()
        arm_state_data = torch.zeros((b, self.seq_len, self.act_dim - 1)).float()  # (b, l, act_dim - 1)
        for i in range(b):
            buffer_len = len(self.state_list[i])
            arm_state_data[i, :buffer_len] = torch.stack(self.state_list[i], dim=0)[:, :6]

        # Attention mask
        attention_mask = torch.zeros(b, self.seq_len, device=self.device).long()
        for i in range(b):
            buffer_len = len(self.state_list[i])
            attention_mask[:, :buffer_len] = 1

        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        rgb_data = rgb_data.to(self.device)
        hand_rgb_data = hand_rgb_data.to(self.device)
        arm_state_data = arm_state_data.to(self.device)
        gripper_state_data = gripper_state_data.to(self.device)
        state_data = {'arm': arm_state_data, 'gripper': gripper_state_data}
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            prediction = self.policy(
                rgb=rgb_data, 
                hand_rgb=hand_rgb_data,
                state=state_data,
                language=tokenized_text,
                attention_mask=attention_mask
        )

        # Arm action
        arm_action_preds = prediction['arm_action_preds']  # (b, l, act_dim - 1)
        arm_action_preds = arm_action_preds[attention_mask > 0].view(-1, self.act_dim - 1)

        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']  # (b, l, 1)
        gripper_action_preds = gripper_action_preds[attention_mask > 0]

        # Use the last action
        arm_action_pred = arm_action_preds.view(b, -1, arm_action_preds.shape[-1])[..., -1, :]
        gripper_action_pred = gripper_action_preds.view(b, -1, gripper_action_preds.shape[-1])[..., -1, :]

        gripper_action_pred = torch.nn.Sigmoid()(gripper_action_pred)
        gripper_action_pred = gripper_action_pred > 0.5
        gripper_action_pred = gripper_action_pred.int().float()
        gripper_action_pred = gripper_action_pred * 2.0 - 1.0
        action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (b, act_dim,)
        if no_batch:
            action_pred = action_pred.squeeze(0)
        action_pred = action_pred.detach().cpu()

        self.rollout_step_counter += 1
    
        return action_pred
