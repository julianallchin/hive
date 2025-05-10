madrona_escape_room_learn/__init__.py
---
from madrona_escape_room_learn.train import train
from madrona_escape_room_learn.learning_state import LearningState
from madrona_escape_room_learn.cfg import TrainConfig, PPOConfig, SimInterface
from madrona_escape_room_learn.action import DiscreteActionDistributions
from madrona_escape_room_learn.actor_critic import (
        ActorCritic, DiscreteActor, Critic,
        BackboneEncoder, RecurrentBackboneEncoder,
        Backbone, BackboneShared, BackboneSeparate,
    )
from madrona_escape_room_learn.profile import profile
import madrona_escape_room_learn.models
import madrona_escape_room_learn.rnn

__all__ = [
        "train", "LearningState", "models", "rnn",
        "TrainConfig", "PPOConfig", "SimInterface",
        "DiscreteActionDistributions",
        "ActorCritic", "DiscreteActor", "Critic",
        "BackboneEncoder", "RecurrentBackboneEncoder",
        "Backbone", "BackboneShared", "BackboneSeparate",
    ]


---
madrona_escape_room_learn/action.py
---
import torch
from torch.distributions.categorical import Categorical

class DiscreteActionDistributions:
    def __init__(self, actions_num_buckets, logits = None):
        self.actions_num_buckets = actions_num_buckets

        self.dists = []
        cur_bucket_offset = 0

        for num_buckets in self.actions_num_buckets:
            self.dists.append(Categorical(logits = logits[
                :, cur_bucket_offset:cur_bucket_offset + num_buckets],
                validate_args=False))
            cur_bucket_offset += num_buckets

    def best(self, out):
        actions = [dist.probs.argmax(dim=-1) for dist in self.dists]
        torch.stack(actions, dim=1, out=out)

    def sample(self, actions_out, log_probs_out):
        actions = [dist.sample() for dist in self.dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(self.dists, actions)]

        torch.stack(actions, dim=1, out=actions_out)
        torch.stack(log_probs, dim=1, out=log_probs_out)

    def action_stats(self, actions):
        log_probs = []
        entropies = []
        for i, dist in enumerate(self.dists):
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)

    def probs(self):
        return [dist.probs for dist in self.dists]


---
madrona_escape_room_learn/actor_critic.py
---
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .action import DiscreteActionDistributions
from .profile import profile

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def _flatten_obs_sequence(self, obs):
        return [o.view(-1, *o.shape[2:]) for o in obs]

    def forward(self, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        raise NotImplementedError


class DiscreteActor(nn.Module):
    def __init__(self, actions_num_buckets, impl):
        super().__init__()

        self.actions_num_buckets = actions_num_buckets
        self.impl = impl

    def forward(self, features_in):
        logits = self.impl(features_in)

        return DiscreteActionDistributions(
                self.actions_num_buckets, logits=logits)


class Critic(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.impl = impl 

    def forward(self, features_in):
        return self.impl(features_in)


class RecurrentStateConfig:
    def __init__(self, shapes):
        self.shapes = shapes
    

class ActorCritic(nn.Module):
    def __init__(self, backbone, actor, critic):
        super().__init__()

        self.backbone = backbone 
        self.recurrent_cfg = backbone.recurrent_cfg
        self.actor = actor
        self.critic = critic

    # Direct call intended for debugging only, should use below
    # specialized functions
    def forward(self, rnn_states, *obs):
        actor_features, critic_features, new_rnn_states = self.backbone(
            rnn_states, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        return action_dists, values, new_rnn_states

    def fwd_actor(self, actions_out, rnn_states_out, rnn_states_in, *obs_in):
        actor_features = self.backbone.fwd_actor_only(
            rnn_states_out, rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        action_dists.best(out=actions_out)

    def fwd_critic(self, values_out, rnn_states_out, rnn_states_in, *obs_in):
        features = self.backbone.fwd_critic_only(
            rnn_states_out, rnn_states_in, *obs_in)
        values_out[...] = self.critic(features)

    def fwd_rollout(self, actions_out, log_probs_out, values_out,
                      rnn_states_out, rnn_states_in, *obs_in):
        actor_features, critic_features = self.backbone.fwd_rollout(
            rnn_states_out, rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        action_dists.sample(actions_out, log_probs_out)
        values_out[...] = values

    def fwd_update(self, rnn_states, sequence_breaks, rollout_actions, *obs):
        actor_features, critic_features = self.backbone.fwd_sequence(
            rnn_states, sequence_breaks, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        T, N = rollout_actions.shape[0:2]
        flattened_actions = rollout_actions.view(
            T * N, *rollout_actions.shape[2:])

        log_probs, entropies = action_dists.action_stats(flattened_actions)

        log_probs = log_probs.view(T, N, *log_probs.shape[1:])
        entropies = entropies.view(T, N, *entropies.shape[1:])
        values = values.view(T, N, *values.shape[1:])

        return log_probs, entropies, values

class BackboneEncoder(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.rnn_state_shape = None

    def forward(self, rnn_states, *inputs):
        features = self.net(*inputs)
        return features, None

    def fwd_inplace(self, rnn_states_out, rnn_states_in, *inputs):
        return self.net(*inputs)

    # *inputs come in pre-flattened
    def fwd_sequence(self, rnn_start_states,
                     sequence_breaks, *flattened_inputs):
        return self.net(*flattened_inputs)

class RecurrentBackboneEncoder(nn.Module):
    def __init__(self, net, rnn):
        super().__init__()
        self.net = net
        self.rnn = rnn
        self.rnn_state_shape = rnn.hidden_shape

    def forward(self, rnn_states_in, *inputs):
        features = self.net(*inputs)

        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        return rnn_out, new_rnn_states

    def fwd_inplace(self, rnn_states_out, rnn_states_in, *inputs):
        features = self.net(*inputs)
        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        # FIXME: proper inplace
        if rnn_states_out != None:
            rnn_states_out[...] = rnn_states_in

        return rnn_out

    # *inputs come in pre-flattened
    def fwd_sequence(self, rnn_start_states, sequence_breaks,
                     *flattened_inputs):
        features = self.net(*flattened_inputs)
        features_seq = features.view(
            *sequence_breaks.shape[0:2], *features.shape[1:])

        with profile('rnn.fwd_sequence'):
            rnn_out_seq = self.rnn.fwd_sequence(
                features_seq, rnn_start_states, sequence_breaks)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])
        return rnn_out_flattened

class BackboneShared(Backbone):
    def __init__(self, process_obs, encoder):
        super().__init__()
        self.process_obs = process_obs
        self.encoder = encoder 

        if encoder.rnn_state_shape:
            self.recurrent_cfg = RecurrentStateConfig([encoder.rnn_state_shape])
            self.extract_rnn_state = lambda x: x[0] if x != None else None
            self.package_rnn_state = lambda x: (x,)
        else:
            self.recurrent_cfg = RecurrentStateConfig([])
            self.extract_rnn_state = lambda x: None
            self.package_rnn_state = lambda x: ()

    def forward(self, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        features, new_rnn_states = self.encoder(
            self.extract_rnn_state(rnn_states_in), processed_obs)
        return features, features, self.package_rnn_state(new_rnn_states)

    def _rollout_common(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.encoder.fwd_inplace(
            self.extract_rnn_state(rnn_states_out),
            self.extract_rnn_state(rnn_states_in),
            processed_obs,
        )

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        features = self._rollout_common(
            rnn_states_out, rnn_states_in, *obs_in)

        return features, features

    def fwd_sequence(self, rnn_start_states, sequence_breaks, *obs_in):
        with torch.no_grad():
            flattened_obs = self._flatten_obs_sequence(obs_in)
            processed_obs = self.process_obs(*flattened_obs)
        
        features = self.encoder.fwd_sequence(
            self.extract_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        return features, features


class BackboneSeparate(Backbone):
    def __init__(self, process_obs, actor_encoder, critic_encoder):
        super().__init__()
        self.process_obs = process_obs
        self.actor_encoder = actor_encoder
        self.critic_encoder = critic_encoder

        rnn_state_shapes = []

        if actor_encoder.rnn_state_shape == None:
            self.extract_actor_rnn_state = lambda rnn_states: None
        else:
            actor_rnn_idx = len(rnn_state_shapes)
            rnn_state_shapes.append(actor_encoder.rnn_state_shape)
            self.extract_actor_rnn_state = \
                lambda rnn_states: rnn_states[actor_rnn_idx]

        if critic_encoder.rnn_state_shape == None:
            self.extract_critic_rnn_state = lambda rnn_states: None
        else:
            critic_rnn_idx = len(rnn_state_shapes)
            rnn_state_shapes.append(critic_encoder.rnn_state_shape)
            self.extract_critic_rnn_state = \
                lambda rnn_states: rnn_states[critic_rnn_idx]

        if (actor_encoder.rnn_state_shape and
                critic_encoder.rnn_state_shape):
            self.package_rnn_states = lambda a, c: (a, c)
        elif actor_encoder.rnn_state_shape:
            self.package_rnn_states = lambda a, c: (a,)
        elif critic_encoder.rnn_state_shape:
            self.package_rnn_states = lambda a, c: (c,)
        else:
            self.package_rnn_states = lambda a, c: ()

        self.recurrent_cfg = RecurrentStateConfig(rnn_state_shapes)

    def forward(self, rnn_states, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_features, new_actor_rnn_states = self.actor_encoder(
            self.extract_actor_rnn_state(rnn_states),
            processed_obs)
        critic_features, new_critic_rnn_states = self.critic_encoder(
            self.extract_critic_rnn_state(rnn_states),
            processed_obs)

        return actor_features, critic_features, self.package_rnn_states(
            new_actor_rnn_states, new_critic_rnn_states)

    def _rollout_common(self, rnn_states_out, rnn_states_in,
                        *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.encoder.fwd_inplace(
            rnn_states_out, rnn_states_in, processed_obs)

    def fwd_actor_only(self, rnn_states_out, rnn_states_in,
                       *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.actor_encoder.fwd_inplace(
            self.extract_actor_rnn_state(rnn_states_out) if rnn_states_out else None,
            self.extract_actor_rnn_state(rnn_states_in),
            processed_obs)

    def fwd_critic_only(self, rnn_states_out, rnn_states_in,
                        *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.critic_encoder.fwd_inplace(
            self.extract_critic_rnn_state(rnn_states_out) if rnn_states_out else None,
            self.extract_critic_rnn_state(rnn_states_in),
            processed_obs)

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_features = self.actor_encoder.fwd_inplace(
            self.extract_actor_rnn_state(rnn_states_out),
            self.extract_actor_rnn_state(rnn_states_in),
            processed_obs)

        critic_features = self.critic_encoder.fwd_inplace(
            self.extract_critic_rnn_state(rnn_states_out),
            self.extract_critic_rnn_state(rnn_states_in),
            processed_obs)

        return actor_features, critic_features

    def fwd_sequence(self, rnn_start_states, sequence_breaks, *obs_in):
        with torch.no_grad():
            flattened_obs = self._flatten_obs_sequence(obs_in)
            processed_obs = self.process_obs(*flattened_obs)
        
        actor_features = self.actor_encoder.fwd_sequence(
            self.extract_actor_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        critic_features = self.critic_encoder.fwd_sequence(
            self.extract_critic_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        return actor_features, critic_features


---
madrona_escape_room_learn/amp.py
---
import torch
from typing import Optional
from contextlib import contextmanager
from dataclasses import dataclass

@dataclass(init=False)
class AMPState:
    device_type: str
    enabled: bool
    compute_dtype: torch.dtype
    scaler: Optional[torch.cuda.amp.GradScaler]

    def __init__(self, dev, enable_mixed_precision):
        self.device_type = dev.type

        if enable_mixed_precision:
            self.enabled = True

            if dev.type == 'cuda':
                self.compute_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.compute_dtype = torch.bfloat16
                self.scaler = None
        else:
            self.enabled = False
            self.compute_dtype = torch.float32
            self.scaler = None

    @contextmanager
    def enable(self):
        if not self.enabled:
            try:
                yield
            finally:
                pass
        else:
            with torch.autocast(self.device_type, dtype=self.compute_dtype):
                try:
                    yield
                finally:
                    pass

    @contextmanager
    def disable(self):
        if not self.enabled:
            try:
                yield
            finally:
                pass
        else:
            with torch.autocast(self.device_type, enabled=False):
                try:
                    yield
                finally:
                    pass


---
madrona_escape_room_learn/cfg.py
---
from dataclasses import dataclass
from typing import Callable, List

import torch

@dataclass(frozen=True)
class PPOConfig:
    num_mini_batches: int
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    num_epochs: int = 1
    clip_value_loss: bool = False
    adaptive_entropy: bool = True

@dataclass(frozen=True)
class TrainConfig:
    num_updates: int
    steps_per_update: int
    num_bptt_chunks: int
    lr: float
    gamma: float
    ppo: PPOConfig
    gae_lambda: float = 1.0
    normalize_advantages: bool = True
    normalize_values : bool = True
    value_normalizer_decay : float = 0.99999
    mixed_precision : bool = False

    def __repr__(self):
        rep = "TrainConfig:"

        for k, v in self.__dict__.items():
            if k == 'ppo':
                rep += f"\n  ppo:"
                for ppo_k, ppo_v in self.ppo.__dict__.items():
                    rep += f"\n    {ppo_k}: {ppo_v}"
            else:
                rep += f"\n  {k}: {v}" 

        return rep

@dataclass(frozen=True)
class SimInterface:
    step: Callable
    obs: List[torch.Tensor]
    actions: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


---
madrona_escape_room_learn/learning_state.py
---
import torch
from dataclasses import dataclass
from typing import Optional

from .amp import AMPState
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer

@dataclass
class LearningState:
    policy: ActorCritic
    optimizer : torch.optim.Optimizer
    scheduler : Optional[torch.optim.lr_scheduler.LRScheduler]
    value_normalizer: EMANormalizer
    amp: AMPState

    def save(self, update_idx, path):
        if self.scheduler != None:
            scheduler_state_dict = self.scheduler.state_dict()
        else:
            scheduler_state_dict = None

        if self.amp.scaler != None:
            scaler_state_dict = self.amp.scaler.state_dict()
        else:
            scaler_state_dict = None

        torch.save({
            'next_update': update_idx + 1,
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': scheduler_state_dict,
            'value_normalizer': self.value_normalizer.state_dict(),
            'amp': {
                'device_type': self.amp.device_type,
                'enabled': self.amp.enabled,
                'compute_dtype': self.amp.compute_dtype,
                'scaler': scaler_state_dict,
            },
        }, path)

    def load(self, path):
        loaded = torch.load(path)

        self.policy.load_state_dict(loaded['policy'])
        self.optimizer.load_state_dict(loaded['optimizer'])

        if self.scheduler:
            self.scheduler.load_state_dict(loaded['scheduler'])
        else:
            assert(loaded['scheduler'] == None)

        self.value_normalizer.load_state_dict(loaded['value_normalizer'])

        amp_dict = loaded['amp']
        if self.amp.scaler:
            self.amp.scaler.load_state_dict(amp_dict['scaler'])
        else:
            assert(amp_dict['scaler'] == None)
        assert(
            self.amp.device_type == amp_dict['device_type'] and
            self.amp.enabled == amp_dict['enabled'] and
            self.amp.compute_dtype == amp_dict['compute_dtype'])

        return loaded['next_update']

    @staticmethod
    def load_policy_weights(path):
        loaded = torch.load(path)
        return loaded['policy']



---
madrona_escape_room_learn/models.py
---
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, DiscreteActor, Critic

class MLP(nn.Module):
    def __init__(self, input_dim, num_channels, num_layers):
        super().__init__()

        layers = [
            nn.Linear(input_dim, num_channels),
            nn.LayerNorm(num_channels),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(num_channels, num_channels))
            layers.append(nn.LayerNorm(num_channels))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, inputs):
        return self.net(inputs)

class LinearLayerDiscreteActor(DiscreteActor):
    def __init__(self, actions_num_buckets, in_channels):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Linear(in_channels, total_action_dim)

        super().__init__(actions_num_buckets, impl)

        nn.init.orthogonal_(self.impl.weight, gain=0.01)
        nn.init.constant_(self.impl.bias, 0)

class LinearLayerCritic(Critic):
    def __init__(self, in_channels):
        super().__init__(nn.Linear(in_channels, 1))

        nn.init.orthogonal_(self.impl.weight)
        nn.init.constant_(self.impl.bias, 0)


---
madrona_escape_room_learn/moving_avg.py
---
import torch
import torch.nn as nn

from .amp import AMPState

# Exponential Moving Average mean and variance estimator for
# values and observations
class EMANormalizer(nn.Module):
    def __init__(self, decay, eps=1e-5, disable=False):
        super().__init__()

        self.disable = disable
        if disable:
            return

        self.eps = eps

        # Current parameter estimates
        self.register_buffer("mu", torch.zeros([], dtype=torch.float32))
        self.register_buffer("inv_sigma", torch.zeros([], dtype=torch.float32))
        self.register_buffer("sigma", torch.zeros([], dtype=torch.float32))

        # Intermediate values used to compute the moving average
        # decay and one_minus_decay don't strictly need to be tensors, but it's
        # critically important for floating point precision that
        # one_minus_decay is computed in fp32 rather than fp64 to 
        # match the bias_correction computation below
        self.register_buffer("decay",
                             torch.tensor(decay, dtype=torch.float32))
        self.register_buffer("one_minus_decay", 1 - self.decay)

        self.register_buffer("mu_biased",
                             torch.zeros([], dtype=torch.float32))
        self.register_buffer("sigma_sq_biased",
                             torch.zeros([], dtype=torch.float32))
        self.register_buffer("N",
                             torch.zeros([], dtype=torch.int64))

        nn.init.constant_(self.mu , 0)
        nn.init.constant_(self.inv_sigma, 0)
        nn.init.constant_(self.sigma, 0)

        nn.init.constant_(self.mu_biased, 0)
        nn.init.constant_(self.sigma_sq_biased, 0)
        nn.init.constant_(self.N, 0)

    def forward(self, amp, x):
        if self.disable:
            return x 

        with amp.disable():
            if self.training:
                x_f32 = x.to(dtype=torch.float32)

                self.N.add_(1)
                bias_correction = -torch.expm1(self.N * torch.log(self.decay))

                self.mu_biased.mul_(self.decay).addcmul_(
                    x_f32.mean(),
                    self.one_minus_decay)

                new_mu = self.mu_biased / bias_correction

                # prev_mu needs to be unbiased (bias_correction only accounts
                # for the initial EMA with 0), since otherwise variance would
                # be off by a squared factor.
                # On the first iteration, simply treat x's variance as the 
                # full estimate of variance
                if self.N == 1:
                    prev_mu = new_mu
                else:
                    prev_mu = self.mu

                sigma_sq_new = torch.mean((x_f32 - prev_mu) * (x_f32 - new_mu))

                self.sigma_sq_biased.mul_(self.decay).addcmul_(
                    sigma_sq_new,
                    self.one_minus_decay)

                sigma_sq = self.sigma_sq_biased / bias_correction

                # Write out new unbiased params
                self.mu = new_mu
                self.inv_sigma = torch.rsqrt(torch.clamp(sigma_sq, min=self.eps))
                self.sigma = torch.reciprocal(self.inv_sigma)

            return torch.addcmul(
                -self.mu * self.inv_sigma,
                x,
                self.inv_sigma,
            ).to(dtype=x.dtype)

    def invert(self, amp, normalized_x):
        if self.disable:
            return normalized_x

        with amp.disable():
            return torch.addcmul(
                self.mu,
                normalized_x.to(dtype=torch.float32),
                self.sigma,
            ).to(dtype=normalized_x.dtype)


---
madrona_escape_room_learn/profile.py
---
from contextlib import contextmanager
from time import time
import torch

__all__ = [ "profile" ]

class DummyGPUEvent:
    def __init__(self, enable_timing):
        pass

    def record(self):
        pass

    def elapsed_time(self, e):
        return 0.0

    def synchronize(self):
        pass

if torch.cuda.is_available():
    GPUTimingEvent = torch.cuda.Event
else:
    GPUTimingEvent = DummyGPUEvent


class Timer:
    def __init__(self, name):
        self.name = name
        self.cur_sum = 0
        self.cpu_mean = 0
        self.N = 0
        self.children = {}

    def start(self):
        self.cpu_start = time()

    def end(self):
        end = time()

        diff = end - self.cpu_start
        self.cur_sum += diff

    def reset(self):
        self.cur_sum = 0
        self.cpu_mean = 0
        self.N = 0

    def gpu_measure(self, sync):
        pass

    def commit(self):
        self.N += 1
        self.cpu_mean += (self.cur_sum - self.cpu_mean) / self.N
        self.cur_sum = 0

    def __repr__(self):
        return f"CPU: {self.cpu_mean:.3f}"


class GPUTimer(Timer):
    def __init__(self, name):
        super().__init__(name)

        self.gpu_mean = 0
        self.gpu_sum = 0
        self.cur_event_idx = 0
        self.start_events = []
        self.end_events = []

    def start(self):
        super().start()

        if self.cur_event_idx >= len(self.start_events):
            self.start_events.append(GPUTimingEvent(enable_timing=True))
            self.end_events.append(GPUTimingEvent(enable_timing=True))

        self.start_events[self.cur_event_idx].record()

    def end(self):
        super().end()
        self.end_events[self.cur_event_idx].record()
        self.cur_event_idx += 1

    def reset(self):
        super().reset()
        self.gpu_mean = 0
        self.gpu_sum = 0
        self.cur_event_idx = 0

    def gpu_measure(self, sync):
        self.cur_event_idx = 0

        for start, end in zip(self.start_events, self.end_events):
            if sync:
                end.synchronize()
            self.gpu_sum += start.elapsed_time(end) / 1000

    def commit(self):
        super().commit()

        assert(self.cur_event_idx == 0)

        self.gpu_mean += (self.gpu_sum - self.gpu_mean) / self.N
        self.gpu_sum = 0

    def __repr__(self):
        return f"CPU: {self.cpu_mean:.3f}, GPU: {self.gpu_mean:.3f}"


class Profiler:
    def __init__(self):
        self.top = {}
        self.parents = []
        self.iter_stack = []
        self.disabled = False

    @contextmanager
    def __call__(self, name, gpu=False):
        if self.disabled:
            try:
                yield
            finally:
                pass
            return

        if len(self.parents) > 0:
            cur_timers = self.parents[-1].children
        else:
            cur_timers = self.top

        try:
            timer = cur_timers[name]
        except KeyError:
            if gpu:
                timer = GPUTimer(name)
            else:
                timer = Timer(name)
            cur_timers[name] = timer

        self.parents.append(timer)

        try:
            timer.start()
            yield
        finally:
            timer.end()
            self.parents.pop()

    def _iter_timers(self, fn):
        if len(self.parents) == 0:
            start = self.top
            starting_depth = 0
        else:
            start = self.parents[-1].children
            starting_depth = len(self.parents)

        for timer in reversed(start.values()):
            self.iter_stack.append((timer, starting_depth))

        while len(self.iter_stack) > 0:
            cur, depth = self.iter_stack.pop()
            fn(cur, depth)
            for child in reversed(cur.children.values()):
                self.iter_stack.append((child, depth + 1))

    def gpu_measure(self, sync=False):
        def measure_timer(timer, depth):
            timer.gpu_measure(sync)

        self._iter_timers(measure_timer)

    def commit(self):
        assert(len(self.parents) == 0)
        self._iter_timers(lambda x, d: x.commit())

    def reset(self):
        assert(len(self.parents) == 0)
        self._iter_timers(lambda x, d: x.reset())

    def clear(self):
        assert(len(self.parents) == 0)
        self.top.clear()

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False

    def report(self, base_indent='    ', depth_indent='  '):
        assert(len(self.parents) == 0)

        def pad(depth):
            return f"{base_indent}{depth_indent * depth}"

        max_len = 0
        def compute_max_len(timer, depth):
            nonlocal max_len

            prefix_len = len(f"{pad(depth)}{timer.name}")
            if prefix_len > max_len:
                max_len = prefix_len

        self._iter_timers(compute_max_len)

        def print_timer(timer, depth):
            prefix = f"{pad(depth)}{timer.name}"
            right_pad_amount = max_len - len(prefix)

            print(f"{pad(depth)}{timer.name}{' ' * right_pad_amount} => {timer}")

        self._iter_timers(print_timer)


profile = Profiler()


---
madrona_escape_room_learn/rnn.py
---
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LSTM"]

class LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=False)

        for name, param in self.lstm.named_parameters():
            # LSTM parameters are named weight_* and bias_*
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        self.num_layers = num_layers
        self.hidden_shape = (2, self.num_layers, hidden_channels)

    def forward(self, in_features, cur_hidden):
        in_features = in_features.view(1, *in_features.shape)

        out, (new_h, new_c) = self.lstm(in_features,
                                        (cur_hidden[0], cur_hidden[1]))

        new_hidden = torch.stack([new_h, new_c], dim=0)

        return out.view(*out.shape[1:]), new_hidden

    def _get_lstm_params(self, layer_idx):
        weight_ih = getattr(self.lstm, f'weight_ih_l{layer_idx}')
        bias_ih = getattr(self.lstm, f'bias_ih_l{layer_idx}')

        weight_hh = getattr(self.lstm, f'weight_hh_l{layer_idx}')
        bias_hh = getattr(self.lstm, f'bias_hh_l{layer_idx}')

        return weight_ih, bias_ih, weight_hh, bias_hh

    def lstm_iter_slow(self, layer_idx, in_features, cur_hidden, breaks):
        weight_ih, bias_ih, weight_hh, bias_hh = self._get_lstm_params(
                layer_idx)

        ifgo = (
            F.linear(in_features, weight_ih, bias_ih) +
            F.linear(cur_hidden[0, :, :], weight_hh, bias_hh)
        )

        hs = self.hidden_shape[-1] # hidden feature size

        c = (F.sigmoid(ifgo[:, hs:2*hs]) * cur_hidden[1, :, :] +
            F.sigmoid(ifgo[:, 0:hs]) * F.tanh(ifgo[:, 2*hs:3*hs]))

        o = ifgo[:, 3*hs:4*hs]

        h = o * F.tanh(c)

        new_hidden = torch.stack([h, c], dim=0)

        return o, new_hidden

    # Manually written LSTM implementation, doesn't work
    def fwd_sequence_slow(self, in_sequences, start_hidden, sequence_breaks):
        seq_len = in_sequences.shape[0]

        hidden_dim_per_layer = start_hidden.shape[-1]

        zero_hidden = torch.zeros((2, self.num_layers, 1,
                                   hidden_dim_per_layer),
                                  device=start_hidden.device,
                                  dtype=start_hidden.dtype)

        out_sequences = []

        cur_hidden = start_hidden
        for i in range(seq_len):
            cur_features = in_sequences[i]
            cur_breaks = sequence_breaks[i]

            new_hiddens = []
            for layer_idx in range(self.num_layers):
                cur_features, new_hidden = self.lstm_iter_slow(
                    layer_idx, cur_features, cur_hidden[:, layer_idx, :, :],
                    sequence_breaks[i])

                new_hiddens.append(new_hidden)
                out_sequences.append(cur_features)

            cur_hidden = torch.stack(new_hiddens, dim=1)

            cur_hidden = torch.where(
                cur_breaks.bool(), zero_hidden, cur_hidden)

        return torch.stack(out_sequences, dim=0)

    # Just call forward repeatedly
    def fwd_sequence_default(self, in_sequences, start_hidden, sequence_breaks):
        seq_len = in_sequences.shape[0]

        hidden_dim_per_layer = start_hidden.shape[-1]

        zero_hidden = torch.zeros((2, self.num_layers, 1,
                                   hidden_dim_per_layer),
                                  device=start_hidden.device,
                                  dtype=start_hidden.dtype)

        out_sequences = []

        cur_hidden = start_hidden
        for i in range(seq_len):
            cur_features = in_sequences[i]
            cur_breaks = sequence_breaks[i]

            out, new_hidden = self.forward(cur_features, cur_hidden)
            out_sequences.append(out)

            cur_hidden = torch.where(
                cur_breaks.bool(), zero_hidden, new_hidden)

        return torch.stack(out_sequences, dim=0)

    fwd_sequence = fwd_sequence_default


---
madrona_escape_room_learn/rollouts.py
---
import torch
from time import time
from dataclasses import dataclass
from typing import List, Optional
from .amp import AMPState
from .cfg import SimInterface
from .actor_critic import ActorCritic, RecurrentStateConfig
from .profile import profile
from .moving_avg import EMANormalizer

@dataclass(frozen = True)
class Rollouts:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    bootstrap_values: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]

class RolloutManager:
    def __init__(
            self,
            dev : torch.device,
            sim : SimInterface,
            steps_per_update : int,
            num_bptt_chunks : int,
            amp : AMPState,
            recurrent_cfg : RecurrentStateConfig,
        ):
        self.dev = dev
        self.steps_per_update = steps_per_update
        self.num_bptt_chunks = num_bptt_chunks
        assert(steps_per_update % num_bptt_chunks == 0)
        num_bptt_steps = steps_per_update // num_bptt_chunks
        self.num_bptt_steps = num_bptt_steps

        self.need_obs_copy = sim.obs[0].device != dev

        if dev.type == 'cuda':
            float_storage_type = torch.float16
        else:
            float_storage_type = torch.bfloat16

        self.actions = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.actions.shape),
            dtype=sim.actions.dtype, device=dev)

        self.log_probs = torch.zeros(
            self.actions.shape,
            dtype=float_storage_type, device=dev)

        self.dones = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.dones.shape),
            dtype=torch.bool, device=dev)

        self.rewards = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
            dtype=float_storage_type, device=dev)

        self.values = torch.zeros(
            (num_bptt_chunks, num_bptt_steps, *sim.rewards.shape),
            dtype=float_storage_type, device=dev)

        self.bootstrap_values = torch.zeros(
            sim.rewards.shape, dtype=amp.compute_dtype, device=dev)

        self.obs = []

        for obs_tensor in sim.obs:
            self.obs.append(torch.zeros(
                (num_bptt_chunks, num_bptt_steps, *obs_tensor.shape),
                dtype=obs_tensor.dtype, device=dev))

        if self.need_obs_copy:
            self.final_obs = []

            for obs_tensor in sim.obs:
                self.final_obs.append(torch.zeros(
                    obs_tensor.shape, dtype=obs_tensor.dtype, device=dev))

        self.rnn_end_states = []
        self.rnn_alt_states = []
        self.rnn_start_states = []
        for rnn_state_shape in recurrent_cfg.shapes:
            # expand shape to batch size
            batched_state_shape = (*rnn_state_shape[0:2],
                sim.actions.shape[0], rnn_state_shape[2])

            rnn_end_state = torch.zeros(
                batched_state_shape, dtype=amp.compute_dtype, device=dev)
            rnn_alt_state = torch.zeros_like(rnn_end_state)

            self.rnn_end_states.append(rnn_end_state)
            self.rnn_alt_states.append(rnn_alt_state)

            bptt_starts_shape = (num_bptt_chunks, *batched_state_shape)

            rnn_start_state = torch.zeros(
                bptt_starts_shape, dtype=amp.compute_dtype, device=dev)

            self.rnn_start_states.append(rnn_start_state)

        self.rnn_end_states = tuple(self.rnn_end_states)
        self.rnn_alt_states = tuple(self.rnn_alt_states)
        self.rnn_start_states = tuple(self.rnn_start_states)

    def collect(
            self,
            amp : AMPState,
            sim : SimInterface,
            actor_critic : ActorCritic,
            value_normalizer : EMANormalizer,
        ):
        rnn_states_cur_in = self.rnn_end_states
        rnn_states_cur_out = self.rnn_alt_states

        for bptt_chunk in range(0, self.num_bptt_chunks):
            with profile("Cache RNN state"):
                # Cache starting RNN state for this chunk
                for start_state, end_state in zip(
                        self.rnn_start_states, rnn_states_cur_in):
                    start_state[bptt_chunk].copy_(end_state)

            for slot in range(0, self.num_bptt_steps):
                cur_obs_buffers = [obs[bptt_chunk, slot] for obs in self.obs]

                with profile('Policy Infer', gpu=True):
                    for obs_idx, step_obs in enumerate(sim.obs):
                        cur_obs_buffers[obs_idx].copy_(step_obs, non_blocking=True)

                    cur_actions_store = self.actions[bptt_chunk, slot]

                    with amp.enable():
                        actor_critic.fwd_rollout(
                            cur_actions_store,
                            self.log_probs[bptt_chunk, slot],
                            self.values[bptt_chunk, slot],
                            rnn_states_cur_out,
                            rnn_states_cur_in,
                            *cur_obs_buffers,
                        )

                    # Invert normalized values
                    self.values[bptt_chunk, slot] = value_normalizer.invert(amp, self.values[bptt_chunk, slot])


                    rnn_states_cur_in, rnn_states_cur_out = \
                        rnn_states_cur_out, rnn_states_cur_in

                    # This isn't non-blocking because if the sim is running in
                    # CPU mode, the copy needs to be finished before sim.step()
                    # FIXME: proper pytorch <-> madrona cuda stream integration

                    # For now, the Policy Infer profile block ends here to get
                    # a CPU synchronization
                    sim.actions.copy_(cur_actions_store)

                with profile('Simulator Step'):
                    sim.step()

                with profile('Post Step Copy'):
                    self.rewards[bptt_chunk, slot].copy_(
                        sim.rewards, non_blocking=True)

                    cur_dones_store = self.dones[bptt_chunk, slot]
                    cur_dones_store.copy_(
                        sim.dones, non_blocking=True)

                    for rnn_states in rnn_states_cur_in:
                        rnn_states.masked_fill_(cur_dones_store, 0)

                profile.gpu_measure(sync=True)

        if self.need_obs_copy:
            final_obs = self.final_obs
            for obs_idx, step_obs in enumerate(sim.obs):
                final_obs[obs_idx].copy_(step_obs, non_blocking=True)
        else:
            final_obs = sim.obs

        # rnn_hidden_cur_in and rnn_hidden_cur_out are flipped after each
        # iter so rnn_hidden_cur_in is the final output
        self.rnn_end_states = rnn_states_cur_in
        self.rnn_alt_states = rnn_states_cur_out

        with amp.enable(), profile("Bootstrap Values"):
            actor_critic.fwd_critic(
                self.bootstrap_values, None, self.rnn_end_states, *final_obs)
            self.bootstrap_values = value_normalizer.invert(amp, self.bootstrap_values)

        # Right now this just returns the rollout manager's pointers,
        # but in the future could return only one set of buffers from a
        # double buffered store, etc

        return Rollouts(
            obs = self.obs,
            actions = self.actions,
            log_probs = self.log_probs,
            dones = self.dones,
            rewards = self.rewards,
            values = self.values,
            bootstrap_values = self.bootstrap_values,
            rnn_start_states = self.rnn_start_states,
        )


---
madrona_escape_room_learn/train.py
---
import torch
from torch import nn
import torch.nn.functional as F
import torch._dynamo
from torch import optim
from torch.func import vmap
from os import environ as env_vars
from typing import Callable
from dataclasses import dataclass
from typing import List, Optional, Dict
from .profile import profile
from time import time
from pathlib import Path

from .cfg import TrainConfig, SimInterface
from .rollouts import RolloutManager, Rollouts
from .amp import AMPState
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer
from .learning_state import LearningState

@dataclass(frozen = True)
class MiniBatch:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]


@dataclass
class PPOStats:
    loss : float = 0
    action_loss : float = 0
    value_loss : float = 0
    entropy_loss : float = 0
    returns_mean : float = 0
    returns_stddev : float = 0


@dataclass(frozen = True)
class UpdateResult:
    actions : torch.Tensor
    rewards : torch.Tensor
    values : torch.Tensor
    advantages : torch.Tensor
    bootstrap_values : torch.Tensor
    ppo_stats : PPOStats


def _mb_slice(tensor, inds):
    # Tensors come from the rollout manager as (C, T, N, ...)
    # Want to select mb from C * N and keep sequences of length T

    return tensor.transpose(0, 1).reshape(
        tensor.shape[1], tensor.shape[0] * tensor.shape[2], *tensor.shape[3:])[:, inds, ...]

def _mb_slice_rnn(rnn_state, inds):
    # RNN state comes from the rollout manager as (C, :, :, N, :)
    # Want to select minibatch from C * N and keep sequences of length T

    reshaped = rnn_state.permute(1, 2, 0, 3, 4).reshape(
        rnn_state.shape[1], rnn_state.shape[2], -1, rnn_state.shape[4])

    return reshaped[:, :, inds, :] 

def _gather_minibatch(rollouts : Rollouts,
                      advantages : torch.Tensor,
                      inds : torch.Tensor,
                      amp : AMPState):
    obs_slice = tuple(_mb_slice(obs, inds) for obs in rollouts.obs)
    
    actions_slice = _mb_slice(rollouts.actions, inds)
    log_probs_slice = _mb_slice(rollouts.log_probs, inds).to(
        dtype=amp.compute_dtype)
    dones_slice = _mb_slice(rollouts.dones, inds)
    rewards_slice = _mb_slice(rollouts.rewards, inds).to(
        dtype=amp.compute_dtype)
    values_slice = _mb_slice(rollouts.values, inds).to(
        dtype=amp.compute_dtype)
    advantages_slice = _mb_slice(advantages, inds).to(
        dtype=amp.compute_dtype)

    rnn_starts_slice = tuple(
        _mb_slice_rnn(state, inds) for state in rollouts.rnn_start_states)

    return MiniBatch(
        obs=obs_slice,
        actions=actions_slice,
        log_probs=log_probs_slice,
        dones=dones_slice,
        rewards=rewards_slice,
        values=values_slice,
        advantages=advantages_slice,
        rnn_start_states=rnn_starts_slice,
    )

def _compute_advantages(cfg : TrainConfig,
                        amp : AMPState,
                        value_normalizer : EMANormalizer,
                        advantages_out : torch.Tensor,
                        rollouts : Rollouts):
    # This function is going to be operating in fp16 mode completely
    # when mixed precision is enabled since amp.compute_dtype is fp16
    # even though there is no autocast here. Unclear if this is desirable or
    # even beneficial for performance.

    num_chunks, steps_per_chunk, N = rollouts.dones.shape[0:3]
    T = num_chunks * steps_per_chunk

    seq_dones = rollouts.dones.view(T, N, 1)
    seq_rewards = rollouts.rewards.view(T, N, 1)
    seq_values = rollouts.values.view(T, N, 1)
    seq_advantages_out = advantages_out.view(T, N, 1)

    next_advantage = 0.0
    next_values = rollouts.bootstrap_values
    for i in reversed(range(cfg.steps_per_update)):
        cur_dones = seq_dones[i].to(dtype=amp.compute_dtype)
        cur_rewards = seq_rewards[i].to(dtype=amp.compute_dtype)
        cur_values = seq_values[i].to(dtype=amp.compute_dtype)

        next_valid = 1.0 - cur_dones

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        td_err = (cur_rewards + 
            cfg.gamma * next_valid * next_values - cur_values)

        # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
        #     = delta_t + gamma * lambda * A_t+1
        cur_advantage = (td_err +
            cfg.gamma * cfg.gae_lambda * next_valid * next_advantage)

        seq_advantages_out[i] = cur_advantage

        next_advantage = cur_advantage
        next_values = cur_values

def _compute_action_scores(cfg, amp, advantages):
    if not cfg.normalize_advantages:
        return advantages
    else:
        # Unclear from docs if var_mean is safe under autocast
        with amp.disable():
            var, mean = torch.var_mean(advantages.to(dtype=torch.float32))
            action_scores = advantages - mean
            action_scores.mul_(torch.rsqrt(var.clamp(min=1e-5)))

            return action_scores.to(dtype=amp.compute_dtype)

def _ppo_update(cfg : TrainConfig,
                amp : AMPState,
                mb : MiniBatch,
                actor_critic : ActorCritic,
                optimizer : torch.optim.Optimizer,
                value_normalizer : EMANormalizer,
            ):
    with amp.enable():
        with profile('AC Forward', gpu=True):
            new_log_probs, entropies, new_values = actor_critic.fwd_update(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs)

        with torch.no_grad():
            action_scores = _compute_action_scores(cfg, amp, mb.advantages)

        ratio = torch.exp(new_log_probs - mb.log_probs)
        surr1 = action_scores * ratio
        surr2 = action_scores * (
            torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef))

        action_obj = torch.min(surr1, surr2)

        returns = mb.advantages + mb.values

        if cfg.ppo.clip_value_loss:
            with torch.no_grad():
                low = mb.values - cfg.ppo.clip_coef
                high = mb.values + cfg.ppo.clip_coef

            new_values = torch.clamp(new_values, low, high)

        normalized_returns = value_normalizer(amp, returns)
        value_loss = 0.5 * F.mse_loss(
            new_values, normalized_returns, reduction='none')

        action_obj = torch.mean(action_obj)
        value_loss = torch.mean(value_loss)
        entropies = torch.mean(entropies)

        loss = (
            - action_obj # Maximize the action objective function
            + cfg.ppo.value_loss_coef * value_loss
            - cfg.ppo.entropy_coef * entropies # Maximize entropy
        )

    with profile('Optimize'):
        if amp.scaler is None:
            loss.backward()
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()
        else:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            amp.scaler.step(optimizer)
            amp.scaler.update()

        optimizer.zero_grad()

    with torch.no_grad():
        returns_var, returns_mean = torch.var_mean(normalized_returns)
        returns_stddev = torch.sqrt(returns_var)

        stats = PPOStats(
            loss = loss.cpu().float().item(),
            action_loss = -(action_obj.cpu().float().item()),
            value_loss = value_loss.cpu().float().item(),
            entropy_loss = -(entropies.cpu().float().item()),
            returns_mean = returns_mean.cpu().float().item(),
            returns_stddev = returns_stddev.cpu().float().item(),
        )

    return stats

def _update_iter(cfg : TrainConfig,
                 amp : AMPState,
                 num_train_seqs : int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 advantages : torch.Tensor,
                 actor_critic : ActorCritic,
                 optimizer : torch.optim.Optimizer,
                 scheduler : torch.optim.lr_scheduler.LRScheduler,
                 value_normalizer : EMANormalizer
            ):
    with torch.no_grad():
        actor_critic.eval()
        value_normalizer.eval()

        with profile('Collect Rollouts'):
            rollouts = rollout_mgr.collect(amp, sim, actor_critic, value_normalizer)
    
        # Engstrom et al suggest recomputing advantages after every epoch
        # but that's pretty annoying for a recurrent policy since values
        # need to be recomputed. https://arxiv.org/abs/2005.12729
        with profile('Compute Advantages'):
            _compute_advantages(cfg,
                                amp,
                                value_normalizer,
                                advantages,
                                rollouts)
    
    actor_critic.train()
    value_normalizer.train()

    with profile('PPO'):
        aggregate_stats = PPOStats()
        num_stats = 0

        for epoch in range(cfg.ppo.num_epochs):
            for inds in torch.randperm(num_train_seqs).chunk(
                    cfg.ppo.num_mini_batches):
                with torch.no_grad(), profile('Gather Minibatch', gpu=True):
                    mb = _gather_minibatch(rollouts, advantages, inds, amp)
                cur_stats = _ppo_update(cfg,
                                        amp,
                                        mb,
                                        actor_critic,
                                        optimizer,
                                        value_normalizer)

                with torch.no_grad():
                    num_stats += 1
                    aggregate_stats.loss += (cur_stats.loss - aggregate_stats.loss) / num_stats
                    aggregate_stats.action_loss += (
                        cur_stats.action_loss - aggregate_stats.action_loss) / num_stats
                    aggregate_stats.value_loss += (
                        cur_stats.value_loss - aggregate_stats.value_loss) / num_stats
                    aggregate_stats.entropy_loss += (
                        cur_stats.entropy_loss - aggregate_stats.entropy_loss) / num_stats
                    aggregate_stats.returns_mean += (
                        cur_stats.returns_mean - aggregate_stats.returns_mean) / num_stats
                    # FIXME
                    aggregate_stats.returns_stddev += (
                        cur_stats.returns_stddev - aggregate_stats.returns_stddev) / num_stats

    return UpdateResult(
        actions = rollouts.actions.view(-1, *rollouts.actions.shape[2:]),
        rewards = rollouts.rewards.view(-1, *rollouts.rewards.shape[2:]),
        values = rollouts.values.view(-1, *rollouts.values.shape[2:]),
        advantages = advantages.view(-1, *advantages.shape[2:]),
        bootstrap_values = rollouts.bootstrap_values,
        ppo_stats = aggregate_stats,
    )

def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 user_cb : Callable,
                 cfg : TrainConfig,
                 num_agents: int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 learning_state : LearningState,
                 start_update_idx : int):
    num_train_seqs = num_agents * cfg.num_bptt_chunks
    assert(num_train_seqs % cfg.ppo.num_mini_batches == 0)

    advantages = torch.zeros_like(rollout_mgr.rewards)

    for update_idx in range(start_update_idx, cfg.num_updates):
        update_start_time  = time()

        with profile("Update Iter Timing"):
            update_result = update_iter_fn(
                cfg,
                learning_state.amp,
                num_train_seqs,
                sim,
                rollout_mgr,
                advantages,
                learning_state.policy,
                learning_state.optimizer,
                learning_state.scheduler,
                learning_state.value_normalizer,
            )

            gpu_sync_fn()

        profile.gpu_measure()
        profile.commit()

        update_end_time = time()
        update_time = update_end_time - update_start_time
        user_cb(update_idx, update_time, update_result, learning_state)

def train(dev, sim, cfg, actor_critic, update_cb, restore_ckpt=None):
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_agents = sim.actions.shape[0]

    actor_critic = actor_critic.to(dev)

    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)

    amp = AMPState(dev, cfg.mixed_precision)

    value_normalizer = EMANormalizer(cfg.value_normalizer_decay,
                                     disable=not cfg.normalize_values)
    value_normalizer = value_normalizer.to(dev)

    learning_state = LearningState(
        policy = actor_critic,
        optimizer = optimizer,
        scheduler = None,
        value_normalizer = value_normalizer,
        amp = amp,
    )

    if restore_ckpt != None:
        start_update_idx = learning_state.load(restore_ckpt)
    else:
        start_update_idx = 0

    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        cfg.num_bptt_chunks, amp, actor_critic.recurrent_cfg)

    if dev.type == 'cuda':
        def gpu_sync_fn():
            torch.cuda.synchronize()
    else:
        def gpu_sync_fn():
            pass

    _update_loop(
        update_iter_fn=_update_iter,
        gpu_sync_fn=gpu_sync_fn,
        user_cb=update_cb,
        cfg=cfg,
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_mgr,
        learning_state=learning_state,
        start_update_idx=start_update_idx,
    )

    return actor_critic.cpu()


---
