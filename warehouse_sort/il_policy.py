"""Imitation-learning policy entrypoints for eval.py / the judge.

Each function satisfies the policy contract:
    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

Wire one in via the config `policy` field:
    pixi run python eval.py difficulty=easy \\
        policy=warehouse_sort.il_policy:load_dp_rgb \\
        checkpoint=<path> eval_config=conf/eval/default.yaml
"""

from collections import deque

import torch


def _add_baseline_path(rel):
    import os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel))
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# RGB Diffusion Policy — image + robot proprioception, NO privileged state.
# Same fixed image input shape at every difficulty; same checkpoint runs across configs.
# Template only — image IL is not yet solving this task.
# --------------------------------------------------------------------------- #
class _DPRgbPolicy:
    """Receding-horizon Diffusion Policy executor.

    This mirrors EXACTLY how the model is run during training-time evaluation
    (diffusion_policy/evaluate.py + the FrameStack wrapper), which is what selects the
    "best" checkpoint:

      * a true rolling history of the last ``obs_horizon`` observations is frame-stacked
        and fed to the model (not a faked/duplicated previous frame), and
      * each diffusion call produces ``act_horizon`` actions that are executed open-loop
        before re-planning (temporal action chunking) instead of replanning every step and
        throwing away 7 of every 8 predicted actions.

    Deploying the policy the same way it was validated is what turns a trained checkpoint
    from "near 0%" into a working sorter, and it is also ~``act_horizon``x faster at eval.
    """

    def __init__(self, agent, obs_horizon, act_horizon, device, num_inference_steps=16):
        self.agent = agent.to(device).eval()
        self.agent.noise_scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.device = device
        self._hist = None      # rolling window of the last obs_horizon obs dicts
        self._queue = []       # not-yet-executed actions from the last plan
        self._batch = None     # current batch size (used to detect a new rollout)

    def reset(self):
        self._hist = None
        self._queue = []
        self._batch = None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = obs["state"].float().to(self.device)
        rgb = obs["rgb"].to(self.device)
        cur = {"state": state, "rgb": rgb}
        b = state.shape[0]

        # New rollout (first call, or the eval harness moved to a fresh batch of envs):
        # start the history fresh and drop any stale planned actions.
        if self._hist is None or self._batch != b:
            self._hist = deque([cur] * self.obs_horizon, maxlen=self.obs_horizon)
            self._queue = []
            self._batch = b
        else:
            self._hist.append(cur)

        if not self._queue:
            obs_seq = {
                "state": torch.stack([h["state"] for h in self._hist], dim=1),
                "rgb": torch.stack([h["rgb"] for h in self._hist], dim=1),
            }
            aseq = self.agent.get_action(obs_seq).clamp(-1.0, 1.0)  # (b, act_horizon, act_dim)
            self._queue = list(aseq.unbind(dim=1))

        return self._queue.pop(0)


def load_dp_rgb(checkpoint, sample_obs, action_space, device,
                obs_horizon=2, act_horizon=8, pred_horizon=16,
                diffusion_step_embed_dim=64, unet_dims=(64, 128, 256), n_groups=8,
                num_inference_steps=16, visual_encoder="resnet18", num_kp=32):
    """Load an RGB Diffusion Policy checkpoint (vendored train_rgbd; uses EMA weights).

    Template implementation — image IL is not yet solving this task.
    """
    import types
    import numpy as np
    import gymnasium.spaces as spaces
    _add_baseline_path("diffusion_policy")
    from train_rgbd import Agent

    h, w, c = sample_obs["rgb"].shape[1:]
    state_dim = sample_obs["state"].shape[1]
    stub = types.SimpleNamespace(
        single_observation_space=spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (obs_horizon, state_dim), np.float32),
            "rgb": spaces.Box(0, 255, (obs_horizon, h, w, c), np.uint8),
        }),
        single_action_space=spaces.Box(-1.0, 1.0, (action_space.shape[0],), np.float32),
    )
    args = types.SimpleNamespace(
        obs_horizon=obs_horizon, act_horizon=act_horizon, pred_horizon=pred_horizon,
        diffusion_step_embed_dim=diffusion_step_embed_dim, unet_dims=list(unet_dims),
        n_groups=n_groups, visual_encoder=visual_encoder, num_kp=num_kp,
    )
    agent = Agent(stub, args)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt.get("ema_agent", ckpt.get("agent")))
    return _DPRgbPolicy(agent, obs_horizon, act_horizon, device,
                        num_inference_steps=num_inference_steps)
