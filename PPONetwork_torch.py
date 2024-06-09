import numpy as np
from typing import Dict, List, Tuple, Type, Union, Optional

import torch as th
from torch import nn
from torch.nn import functional as F
from distributions import MaskableDistribution, MaskableCategoricalDistribution


def thresholded(logits, regrets, threshold=2.5):
    """Zeros out `regrets` where `logits` are too negative or too large."""
    can_decrease = th.gt(logits, -threshold).float()
    can_increase = th.lt(logits, threshold).float()
    regrets_negative = th.minimum(regrets, th.Tensor([0.0]).to(regrets.device))
    regrets_positive = th.maximum(regrets, th.Tensor([0.0]).to(regrets.device))
    return can_decrease * regrets_negative + can_increase * regrets_positive


def compute_baseline(policy, action_values):
    # V = pi * Q, backprop through pi but not Q.
    return th.sum(th.mul(policy, action_values.detach()), dim=1)


def compute_advantages(
    policy_logits,
    legal_pi,
    action_values,
    action_mask,
    use_relu=False,
    threshold_fn=None,
):
    """Compute advantages using pi and Q."""
    # Compute advantage.
    # Avoid computing gradients for action_values.
    action_values = action_values.detach()  # Q

    baseline = compute_baseline(legal_pi, action_values)  # v

    advantages = action_values - th.unsqueeze(baseline, 1)
    if use_relu:
        advantages = F.relu(advantages)

    if threshold_fn:
        # Compute thresholded advanteges weighted by policy logits for NeuRD.
        policy_logits = policy_logits - (policy_logits * action_mask).mean(
            -1, keepdim=True
        )
        advantages = threshold_fn(policy_logits, advantages)
        policy_advantages = -action_mask * th.mul(policy_logits, advantages.detach())
    else:
        # Compute advantage weighted by policy.
        policy_advantages = -th.mul(legal_pi, advantages.detach())
    return th.sum(policy_advantages, dim=1)


class PPONetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super(PPONetwork, self).__init__()
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = 1024
        last_layer_dim_vf = 1024

        self.shared_linear = nn.Linear(feature_dim, 1024).to(device)
        self.shared_linear2 = nn.Linear(1024, 1024).to(device)
        self.shared_linear3 = nn.Linear(1024, 1024).to(device)
        self.shared_linear4 = nn.Linear(1024, 1024).to(device)
        self.shared_linear5 = nn.Linear(1024, 1024).to(device)

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        policy_net.append(nn.Linear(last_layer_dim_pi, action_dim))
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim
        value_net.append(nn.Linear(last_layer_dim_vf, 1))

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        self.action_dist = MaskableCategoricalDistribution(action_dim)
        self.device = device
        self.reset_parameters()

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def reset_parameters(self) -> None:
        module_gains = {
            self.shared_linear: np.sqrt(2),
            self.shared_linear2: np.sqrt(2),
            self.shared_linear3: np.sqrt(2),
            self.shared_linear4: np.sqrt(2),
            self.shared_linear5: np.sqrt(2),
            self.policy_net: 0.01,
            self.value_net: 1,
        }
        for module, gain in module_gains.items():
            self.init_weights(module, gain=gain)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        h = self.forward_shared(features)
        logits = self.policy_net(h)
        v = self.value_net(h)
        return logits, v

    def forward_shared(self, features: th.Tensor) -> th.Tensor:
        h1 = F.relu(self.shared_linear(features))
        h2 = F.relu(self.shared_linear3(F.relu(self.shared_linear2(h1))) + h1)
        h3 = F.relu(self.shared_linear5(F.relu(self.shared_linear4(h2))) + h2)
        return h3

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        h = self.forward_shared(features)
        return self.policy_net(h)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        h = self.forward_shared(features)
        return self.value_net(h)

    def step(self, obs, action_mask, action_feats):
        """
        obs is a tensor of shape (batch_size, obs_dim)
        availAcs is a tensor of shape (batch_size, n_actions) with 1 for available actions and 0 for unavailable actions
        """

        logits, v = self.forward(
            obs
        )  # logits is a tensor of shape (batch_size, n_actions), v is a tensor of shape (batch_size, 1)
        distribution = self._get_action_dist_from_logits(logits)
        distribution.apply_masking(action_mask)
        a = distribution.get_actions(deterministic=False)
        neglogpac = -distribution.log_prob(a)

        return a, v, neglogpac

    def neglogp(self, obs, action_mask, actions):  # for GUI
        logits = self.forward_actor(obs)
        distribution = self._get_action_dist_from_logits(logits)
        distribution.apply_masking(action_mask)
        neglogpac = -distribution.log_prob(actions)
        return neglogpac

    def value(self, obs, availableActions):
        return self.forward_critic(obs)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.array] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        logits, v = self.forward(obs)
        distribution = self._get_action_dist_from_logits(logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        neglog_prob = -log_prob
        entropy = distribution.entropy()
        return neglog_prob, v, entropy

    def get_masked_entropy_from_logits(
        self, logits: th.Tensor, action_masks: Optional[np.array] = None
    ) -> th.Tensor:
        distribution = self._get_mased_distribution_from_logits(logits, action_masks)
        return distribution.entropy()

    def _get_mased_distribution_from_logits(
        self, logits: th.Tensor, action_masks: Optional[np.array] = None
    ) -> MaskableDistribution:
        distribution = self._get_action_dist_from_logits(logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def _get_action_dist_from_logits(
        self, action_logits: th.Tensor
    ) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        return self.action_dist.proba_distribution(action_logits=action_logits)


class NeuRDNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super(NeuRDNetwork, self).__init__()
        policy_net: List[nn.Module] = []
        action_value_net: List[nn.Module] = []
        last_layer_dim_pi = 512
        last_layer_dim_vf = 512

        self.shared_linear = nn.Linear(feature_dim, 512).to(device)
        self.shared_linear2 = nn.Linear(512, 512).to(device)
        self.shared_linear3 = nn.Linear(512, 512).to(device)
        self.shared_linear4 = nn.Linear(512, 512).to(device)
        self.shared_linear5 = nn.Linear(512, 512).to(device)

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        policy_net.append(nn.Linear(last_layer_dim_pi, action_dim))
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            action_value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            action_value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim
        action_value_net.append(nn.Linear(last_layer_dim_vf, action_dim))  # Q values

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.action_value_net = nn.Sequential(*action_value_net).to(device)
        self.action_dist = MaskableCategoricalDistribution(action_dim)
        self.device = device
        self.reset_parameters()

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def reset_parameters(self) -> None:
        module_gains = {
            self.shared_linear: np.sqrt(2),
            self.shared_linear2: np.sqrt(2),
            self.shared_linear3: np.sqrt(2),
            self.shared_linear4: np.sqrt(2),
            self.shared_linear5: np.sqrt(2),
            self.policy_net: 0.01,
            self.action_value_net: 1,
        }
        for module, gain in module_gains.items():
            self.init_weights(module, gain=gain)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        h = self.forward_shared(features)
        logits = self.policy_net(h)
        q = self.action_value_net(h)
        return logits, q

    def forward_shared(self, features: th.Tensor) -> th.Tensor:
        h1 = F.relu(self.shared_linear(features))
        h2 = F.relu(self.shared_linear3(F.relu(self.shared_linear2(h1))) + h1)
        h3 = F.relu(self.shared_linear5(F.relu(self.shared_linear4(h2))) + h2 + h1)
        return h3

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        h = self.forward_shared(features)
        return self.policy_net(h)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        h = self.forward_shared(features)
        return self.action_value_net(h)

    def step(self, obs, action_mask, action_feats):
        """
        obs is a tensor of shape (batch_size, obs_dim)
        availAcs is a tensor of shape (batch_size, n_actions) with 1 for available actions and 0 for unavailable actions
        """

        logits, q = self.forward(
            obs
        )  # logits is a tensor of shape (batch_size, n_actions), q is a tensor of shape (batch_size, n_actions)
        distribution = self.get_legal_dist_from_logits(logits, action_mask)
        a = distribution.get_actions(deterministic=False)
        neglogpac = -distribution.log_prob(a)

        # if q is one dimension,
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        # q value of the action taken
        q_a = q[th.arange(q.shape[0]), a]

        return a, q_a, neglogpac

    def value(self, obs, availableActions):
        logits_pred, q = self.forward(obs)
        distribution = self.get_legal_dist_from_logits(logits_pred, availableActions)
        legal_pi = distribution.probs()
        v = compute_baseline(legal_pi, q)
        return v

    def neglogp(self, obs, action_mask, actions):  # for GUI
        logits = self.forward_actor(obs)
        distribution = self.get_legal_dist_from_logits(logits, action_mask)
        neglogpac = -distribution.log_prob(actions)
        return neglogpac

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.array] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        logits, q = self.forward(obs)
        distribution = self.get_legal_dist_from_logits(logits, action_masks)
        log_prob = distribution.log_prob(actions)
        neglog_prob = -log_prob
        entropy = distribution.entropy()
        return neglog_prob, q, entropy

    def get_legal_dist_from_logits(
        self, action_logits: th.Tensor, action_masks: th.Tensor
    ) -> MaskableDistribution:
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        distribution.apply_masking(action_masks)
        return distribution


class NeuRDSequentialNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super(NeuRDSequentialNetwork, self).__init__()
        policy_net: List[nn.Module] = []
        action_value_net: List[nn.Module] = []
        last_layer_dim_pi = 512
        last_layer_dim_vf = 512

        self.policy_linear = nn.Linear(feature_dim, 512).to(device)
        self.policy_linear2 = nn.Linear(512, 512).to(device)
        self.policy_linear3 = nn.Linear(512, 512).to(device)
        self.policy_linear4 = nn.Linear(512, 512).to(device)
        self.policy_linear5 = nn.Linear(512, 512).to(device)
        
        self.q_linear = nn.Linear(feature_dim, 512).to(device)
        self.q_linear2 = nn.Linear(512, 512).to(device)
        self.q_linear3 = nn.Linear(512, 512).to(device)
        self.q_linear4 = nn.Linear(512, 512).to(device)
        self.q_linear5 = nn.Linear(512, 512).to(device)
        self.action_embed = nn.Embedding(action_dim, 512).to(device)

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        policy_net.append(nn.Linear(last_layer_dim_pi, 1))
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            action_value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            action_value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim
        action_value_net.append(nn.Linear(last_layer_dim_vf, 1))  # Q values

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.action_value_net = nn.Sequential(*action_value_net).to(device)
        self.action_dist = MaskableCategoricalDistribution(action_dim)
        self.device = device
        self.reset_parameters()

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        if isinstance(module, (nn.Embedding)):
            nn.init.orthogonal_(module.weight, gain=gain)

    def reset_parameters(self) -> None:
        module_gains = {
            self.policy_linear: np.sqrt(2),
            self.policy_linear2: np.sqrt(2),
            self.policy_linear3: np.sqrt(2),
            self.policy_linear4: np.sqrt(2),
            self.policy_linear5: np.sqrt(2),
            self.q_linear: np.sqrt(2),
            self.q_linear2: np.sqrt(2),
            self.q_linear3: np.sqrt(2),
            self.q_linear4: np.sqrt(2),
            self.q_linear5: np.sqrt(2),
            self.action_embed: np.sqrt(2),
            self.policy_net: 0.01,
            self.action_value_net: 1,
        }
        for module, gain in module_gains.items():
            self.init_weights(module, gain=gain)

    def forward(
        self, features: th.Tensor, action_mask: th.Tensor, action_feats: List[th.Tensor]
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        features: (batch_size, feature_dim)
        action_mask: (batch_size, n_actions)
        action_feats: (batch_size, n_actions, action_feat_dim)
        """
        batch_size = features.shape[0]
        n_actions = action_mask.shape[-1]
        logits = th.zeros(batch_size, n_actions, device=self.device)
        q = th.zeros(batch_size, n_actions, device=self.device)

        # action_feats = th.cat(action_feats, dim=0)  # (batch_size * n_availActions of each batch, action_feat_dim)
        list_n_availActions = action_mask.sum(dim=-1).int()
        action_id = th.nonzero(action_mask, as_tuple=True)[1]  # (batch_size * (n_availActions of each batch))
        features = features.type(th.LongTensor)
        f = features.repeat_interleave(list_n_availActions, dim=0).float() # (batch_size * (n_availActions of each batch), feature_dim)
        a_m = action_mask.long().repeat_interleave(list_n_availActions.long(), dim=0).float() # (batch_size * n_availActions of each batch, n_availActions)
        print(action_feats)
        a_f = th.cat(action_feats, dim=0) # (n_availActions, action_feat_dim)

        # h = self._forward_shared(
        #     features.repeat_interleave(list_n_availActions, dim=0), # (batch_size * (n_availActions of each batch), feature_dim)
        #     action_mask.repeat_interleave(list_n_availActions, dim=0), # (batch_size * n_availActions of each batch, n_availActions)
        #     th.cat(action_feats, dim=0) , # (n_availActions, action_feat_dim)
        #     action_id, # (batch_size * (n_availActions of each batch))
        # )  # (n_availActions, last_layer_dim)
        
        print(f.shape)
        print(a_m.shape)
        print(a_f.shape)
        _policy_h1 = F.relu(self.policy_linear(th.cat((f, a_m, a_f), -1)) + self.action_embed(action_id))
        _policy_h2 = F.relu(self.policy_linear3(F.relu(self.policy_linear2(_policy_h1))) + _policy_h1)
        policy_h = F.relu(self.policy_linear5(F.relu(self.policy_linear4(_policy_h2))) + _policy_h2 + _policy_h1)
        logits[action_mask.bool()] = self.policy_net(policy_h).flatten()


        _q_h1 = F.relu(self.q_linear(th.cat((f, a_m, a_f), -1)) + self.action_embed(action_id))
        _q_h2 = F.relu(self.q_linear3(F.relu(self.q_linear2(_q_h1))) + _q_h1)
        q_h = F.relu(self.q_linear5(F.relu(self.q_linear4(_q_h2))) + _q_h2 + _q_h1)
        q[action_mask.bool()] = self.action_value_net(q_h).flatten()
        return logits, q  # (batch_size, n_actions), (batch_size, n_actions)

    # def _forward_shared(
    #     self, features: th.Tensor, action_mask: th.Tensor, action_feats: th.Tensor, action_id: th.Tensor
    # ) -> th.Tensor:
    #     h1 = F.relu(self.policy_linear(th.cat((features, action_mask, action_feats), -1)) + self.action_embed(action_id))
    #     h2 = F.relu(self.policy_linear3(F.relu(self.policy_linear2(h1))) + h1)
    #     h3 = F.relu(self.policy_linear5(F.relu(self.policy_linear4(h2))) + h2 + h1)
    #     return h3

    def step(self, obs, action_mask, action_feats):
        """
        obs is a tensor of shape (batch_size, obs_dim)
        availAcs is a tensor of shape (batch_size, n_actions) with 1 for available actions and 0 for unavailable actions
        action_feats is a tensor of shape (batch_size, n_actions, action_feats_dim)
        """

        logits, q = self.forward(
            obs, action_mask, action_feats
        )  # logits is a tensor of shape (batch_size, n_actions), q is a tensor of shape (batch_size, n_actions)
        distribution = self.get_legal_dist_from_logits(logits, action_mask)
        a = distribution.get_actions(deterministic=False)
        neglogpac = -distribution.log_prob(a)

        # if q is one dimension,
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        # q value of the action taken
        q_a = q[th.arange(q.shape[0]), a]

        return a, q_a, neglogpac

    def value(self, obs, action_mask, action_feats):
        logits_pred, q = self.forward(obs, action_mask, action_feats)
        distribution = self.get_legal_dist_from_logits(logits_pred, action_mask)
        legal_pi = distribution.probs()
        v = compute_baseline(legal_pi, q)
        return v

    def neglogp(self, obs, action_mask, action_feats, actions):  # for GUI
        logits, q = self.forward(obs, action_mask, action_feats)
        distribution = self.get_legal_dist_from_logits(logits, action_mask)
        neglogpac = -distribution.log_prob(actions)
        return neglogpac

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.array] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        logits, q = self.forward(obs)
        distribution = self.get_legal_dist_from_logits(logits, action_masks)
        log_prob = distribution.log_prob(actions)
        neglog_prob = -log_prob
        entropy = distribution.entropy()
        return neglog_prob, q, entropy

    def get_legal_dist_from_logits(
        self, action_logits: th.Tensor, action_masks: th.Tensor
    ) -> MaskableDistribution:
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        distribution.apply_masking(action_masks)
        return distribution


class PPOModel(nn.Module):
    def __init__(
        self, network, inputDim, actDim, ent_coef, vf_coef, max_grad_norm, l2_coef
    ):
        super(PPOModel, self).__init__()

        self.network = network
        print(f"==>> self.network: {self.network}")
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.l2_coef = l2_coef

        self.optimizer = th.optim.Adam(
            network.parameters(), betas=(0.0, 0.999), lr=0.00001
        )

        self.max_grad_norm = max_grad_norm

    def train(
        self,
        lr,
        cliprange,
        vfcliprange,
        observations,
        availableActions,
        returns,
        actions,
        values,
        neglogpacs,
    ):
        # Change learning rate
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        observations = th.tensor(observations, dtype=th.float32, device="cuda")
        availableActions = th.tensor(availableActions, dtype=th.float32, device="cuda")
        returns = th.tensor(returns, dtype=th.float32, device="cuda")
        actions = th.tensor(actions, dtype=th.int64, device="cuda")
        values = th.tensor(values, dtype=th.float32, device="cuda")
        neglogpacs = th.tensor(neglogpacs, dtype=th.float32, device="cuda")

        advs = returns - values
        # advs = (advs-advs.mean()) / (advs.std() + 1e-8)
        advs = advs - advs.mean()

        neglogpac_pred, v_pred, entropy = self.network.evaluate_actions(
            observations, actions, availableActions
        )

        entropyLoss = -th.mean(entropy)

        # value loss
        # v_pred = self.network.forward_critic(observations)
        v_pred = v_pred.flatten()
        # Clip the different between old and new value
        # NOTE: this depends on the reward scaling
        v_pred = values + th.clamp(v_pred - values, -vfcliprange, vfcliprange)
        # Value loss using the TD(gae_lambda) target
        vf_loss = F.mse_loss(returns, v_pred)

        # Policy gradient loss

        # neglogpac_pred = -th.log(p[th.arange(p.shape[0]), actions])  # (batch_size)
        # ratio between old and new policy, should be one at the first iteration
        prob_ratio = th.exp(neglogpacs - neglogpac_pred)  # (batch_size)
        # clipped surrogate loss
        policy_loss_1 = advs * prob_ratio
        policy_loss_2 = advs * th.clamp(prob_ratio, 1 - cliprange, 1 + cliprange)
        pg_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * entropyLoss
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return pg_loss.item(), vf_loss.item(), entropyLoss.item()

    def train_neurd(
        self,
        lr,
        cliprange,
        vfcliprange,
        observations,
        availableActions,
        actionFeats,
        returns,
        actions,
        values,
        neglogpacs,
    ):
        # train the model based on Neural Replicator Dynamics
        # Change learning rate
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        observations = th.tensor(observations, dtype=th.float32, device="cuda")
        availableActions = th.tensor(availableActions, dtype=th.float32, device="cuda")
        actionFeats = [th.tensor(actionFeat, dtype=th.float32, device="cuda") for actionFeat in actionFeats]
        returns = th.tensor(returns, dtype=th.float32, device="cuda")
        actions = th.tensor(actions, dtype=th.int64, device="cuda")
        values = th.tensor(values, dtype=th.float32, device="cuda")
        neglogpacs = th.tensor(neglogpacs, dtype=th.float32, device="cuda")

        logits_pred, action_values = self.network.forward(observations, availableActions, actionFeats)

        non_single_action_mask = availableActions.sum(axis=-1) > 1
        logits_pred = logits_pred[non_single_action_mask]
        availableActions = availableActions[non_single_action_mask]

        distribution = self.network.get_legal_dist_from_logits(
            logits_pred, availableActions
        )
        legal_pi = distribution.probs()
        entropy = distribution.entropy()

        advantages = compute_advantages(
            logits_pred,
            legal_pi,
            action_values[non_single_action_mask],
            availableActions,
            threshold_fn=thresholded,
        )
        pg_loss = advantages.mean(axis=0)
        print(f"==>> pg_loss: {pg_loss}")

        # critic loss for q-values
        value_predictions = action_values[th.arange(action_values.shape[0]), actions]
        critic_loss = F.mse_loss(returns, value_predictions)
        print(f"==>> critic_loss: {critic_loss}")

        entropy_loss = -th.mean(entropy)
        print(f"==>> entropy_loss: {entropy_loss}")

        loss = pg_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return pg_loss.item(), critic_loss.item(), entropy_loss.item()
