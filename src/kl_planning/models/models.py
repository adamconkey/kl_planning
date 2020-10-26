import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List
from dataclasses import dataclass


def get_activation(activation_name):
    # TODO should make these consistent, I'm not sure if they should be from functional
    # or not. Might be easier to configure if from nn, but if you just pass kwargs then
    # maybe functional is better?
    if activation_name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
        # return nn.functional.leaky_relu
    elif activation_name == 'prelu':
        return nn.PReLU()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation type: {activation_name}")


def get_normalization(normalization_name, **kwargs):
    if normalization_name == 'batch_norm':
        return nn.BatchNorm2d(**kwargs)
    elif normalization_name == 'layer_norm':
        pass
    else:
        raise ValueError(f"Unknown normalization type: {normalization_name}")

    
class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes, layer_norm=True, activation='prelu'):
        super().__init__()

        sizes = [input_size] + hidden_sizes
        modules = []
        for i in range(1, len(sizes)):
            modules.append(nn.Linear(sizes[i-1], sizes[i]))
            modules.append(get_activation(activation))
            if layer_norm:
                modules.append(nn.LayerNorm((sizes[i],)))
        modules.append(nn.Linear(sizes[-1], output_size))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ImageEncoder(nn.Module):
  
    def __init__(self, channels, embedding_size, activation='relu'):
        super().__init__()
        self.act_fn = get_activation(activation)
        
        # TODO hacking this to see if it's effective, can make nicer if so
        self.out_sizes = [32, 32, 64, 64, 128, 128, 256, 256]
        self.norm_layers = [get_normalization('batch_norm', num_features=s) for s in self.out_sizes]
        self.norm_layers = nn.ModuleList(self.norm_layers)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2 * embedding_size)

        self.conv1 = nn.Conv2d(channels, 32, 4, stride=2)
        self.conv1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv3_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.conv4_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.act_fn(self.conv1(x))
        x = self.norm_layers[0](x)
        x = self.act_fn(self.conv1_1(x))
        x = self.norm_layers[1](x)
        x = self.act_fn(self.conv2(x))
        x = self.norm_layers[2](x)
        x = self.act_fn(self.conv2_1(x))
        x = self.norm_layers[3](x)
        x = self.act_fn(self.conv3(x))
        x = self.norm_layers[4](x)
        x = self.act_fn(self.conv3_1(x))
        x = self.norm_layers[5](x)
        x = self.act_fn(self.conv4(x))
        x = self.norm_layers[6](x)
        x = self.act_fn(self.conv4_1(x))
        x = self.norm_layers[7](x)

        x = x.view(batch_size, -1)
        x = self.act_fn(self.fc1(x))
        x = self.fc2(x)

        mu, log_std = torch.chunk(x, 2, dim=1)
        std = torch.exp(log_std)
        noise = torch.randn_like(mu)
        
        return mu + std * noise
  

class ImageDecoder(nn.Module):
  
    def __init__(self, channels, state_size, activation='relu', out_activation=None):
        super().__init__()
        self.act_fn = get_activation(activation)
        self.out_act_fn = get_activation(out_activation) if out_activation else None

        # TODO testing this
        self.out_sizes = [128, 64, 32]
        self.norm_layers = [get_normalization('batch_norm', num_features=s) for s in self.out_sizes]
        self.norm_layers = nn.ModuleList(self.norm_layers)
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        
        self.conv1 = nn.ConvTranspose2d(128, 128, 5, stride=2)
        # self.conv1_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        # self.conv2_1 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        # self.conv3_1 = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        # self.conv4_1 = nn.ConvTranspose2d(3, channels, 3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 128, 1, 1)
        
        x = self.act_fn(self.conv1(x))
        x = self.norm_layers[0](x)
        x = self.act_fn(self.conv2(x))
        x = self.norm_layers[1](x)
        x = self.act_fn(self.conv3(x))
        x = self.norm_layers[2](x)
        x = self.conv4(x)
        if self.out_act_fn:
            x = self.out_act_fn(x)
        
        return x

    
class VectorEncoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=[32, 32], layer_norm=True,
                 activation='prelu'):
        super().__init__()

        self.output_size = output_size
        
        modules = []
        sizes = [input_size] + hidden_sizes
        for i in range(1, len(sizes)):
            modules.append(nn.Linear(sizes[i-1], sizes[i]))
            modules.append(get_activation(activation))
            if layer_norm:
                modules.append(nn.LayerNorm((sizes[i],)))
        modules.append(nn.Linear(sizes[-1], output_size))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = self.model(x)
        return x

    
class VectorDecoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=[32, 32], layer_norm=True,
                 activation='prelu', out_activation='tanh'):
        super().__init__()

        modules = []
        sizes = [input_size] + hidden_sizes
        for i in range(1, len(sizes)):
            modules.append(nn.Linear(sizes[i-1], sizes[i]))
            modules.append(get_activation(activation))
            if layer_norm:
                modules.append(nn.LayerNorm((sizes[i],)))
        modules.append(nn.Linear(sizes[-1], output_size))
        self.model = nn.Sequential(*modules)

        self.out_activation = get_activation(out_activation) if out_activation else None

    def forward(self, x):
        x = self.model(x)
        if self.out_activation:
            x = self.out_activation(x)
        return x

    
class MultisensoryEncoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=[512, 256],
                 layer_norm=True, activation='prelu'):
        super().__init__()

        modules = []
        sizes = [input_size] + hidden_sizes
        for i in range(1, len(sizes)):
            modules.append(nn.Linear(sizes[i-1], sizes[i]))
            modules.append(get_activation(activation))
            if layer_norm:
                modules.append(nn.LayerNorm((sizes[i],)))
        modules.append(nn.Linear(sizes[-1], output_size))
        
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = self.model(x)
        return x

    
class MulticlassClassifier(nn.Module):

    def __init__(self, input_size, n_classes, hidden_sizes=[128, 128], layer_norm=True,
                 activation='prelu'):
        super().__init__()

        modules = []
        sizes = [input_size] + hidden_sizes
        for i in range(1, len(sizes)):
            modules.append(nn.Linear(sizes[i-1], sizes[i]))
            modules.append(get_activation(activation))
            if layer_norm:
                modules.append(nn.LayerNorm((sizes[i],)))
        modules.append(nn.Linear(sizes[-1], n_classes))
        # Using nn.CrossEntropyLoss which combines logsoftmax and NLL loss,
        # so no activation needed here since it assumes raw scores as outputs

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = self.model(x)
        return x


@dataclass
class _RSSMReturn:
    beliefs: torch.Tensor = None
    prior_states: torch.Tensor = None
    prior_means: torch.Tensor = None
    prior_std_devs: torch.Tensor = None
    posterior_states: torch.Tensor = None
    posterior_means: torch.Tensor = None
    posterior_std_devs: torch.Tensor = None


class RSSM(nn.Module):
    """
    This is the recurrent state-space model (RSSM) from PlaNet. This code is adapted directly from
    the pytorch port of PlaNet: https://github.com/Kaixhin/PlaNet/blob/master/models.py#L15
    """
    __constants__ = ['min_std_dev']

    def __init__(self,
                 belief_size,
                 state_size,
                 action_size,
                 hidden_size,
                 embedding_size,
                 activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size) # Output mean and std dev
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size) # Output mean and std dev
        
    def forward(self,
                prev_state:torch.Tensor,
                actions:torch.Tensor,
                prev_belief:torch.Tensor,
                observations:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        """
        Operates over (previous) state, (previous) actions, (previous) belief,
        and (current) encoded observation. Diagram of expected inputs and outputs for 
        T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
            t :  0  1  2  3  4  5
            o :    -X--X--X--X--X-
            a : -X--X--X--X--X-
            n : -X--X--X--X--X-
            pb: -X-
            ps: -X-
            b : -x--X--X--X--X--X-
            s : -x--X--X--X--X--X-
        """
        # Create lists for hidden states (cannot use single tensor as buffer because
        # autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs = [torch.empty(0)] * T
        prior_states = [torch.empty(0)] * T
        prior_means = [torch.empty(0)] * T
        prior_std_devs = [torch.empty(0)] * T
        posterior_states = [torch.empty(0)] * T
        posterior_means = [torch.empty(0)] * T
        posterior_std_devs = [torch.empty(0)] * T
        beliefs[0] = prev_belief
        prior_states[0] = prev_state
        posterior_states[0] = prev_state
        
        for t in range(T - 1):
            s_t = prior_states[t] if observations is None else posterior_states[t]
            a_t = actions[t]
            b_t = beliefs[t]

            # print("STATE", s_t.shape)  # (b, c)
            # print("ACT", a_t.shape)    # (b, c)
            # print("BELIEF", b_t.shape) # (b, c)
            
            # Update belief (deterministic RNN hidden state)
            rnn_in = self.act_fn(self.fc_embed_state_action(torch.cat([s_t, a_t], dim=-1)))
            b_tp1 = self.rnn(rnn_in, b_t)
            beliefs[t + 1] = b_tp1
            # Compute state prior by applying transition dynamics
            prior_out = self.fc_state_prior(self.act_fn(self.fc_embed_belief_prior(b_tp1)))
            prior_means[t + 1], log_prior_std_dev = torch.chunk(prior_out, 2, dim=1)
            prior_std_dev = torch.exp(log_prior_std_dev)
            prior_std_devs[t + 1] = F.softplus(prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] \
                                  * torch.randn_like(prior_means[t + 1])     
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                embed_in = torch.cat([b_tp1, observations[t]], dim=1)
                posterior_in = self.act_fn(self.fc_embed_belief_posterior(embed_in))
                posterior_out = self.fc_state_posterior(posterior_in)
                posterior_means[t + 1], log_posterior_std_dev = torch.chunk(posterior_out, 2, dim=1)
                posterior_std_dev = torch.exp(log_posterior_std_dev)
                posterior_std_devs[t + 1] = F.softplus(posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] \
                                          * torch.randn_like(posterior_means[t + 1])
        # Return new hidden states
        hidden = _RSSMReturn(beliefs=torch.stack(beliefs[1:], dim=0),
                             prior_states=torch.stack(prior_states[1:], dim=0),
                             prior_means=torch.stack(prior_means[1:], dim=0),
                             prior_std_devs=torch.stack(prior_std_devs[1:], dim=0))
        if observations is not None:
            hidden.posterior_states = torch.stack(posterior_states[1:], dim=0)
            hidden.posterior_means = torch.stack(posterior_means[1:], dim=0)
            hidden.posterior_std_devs = torch.stack(posterior_std_devs[1:], dim=0)
        return hidden

    
if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device('cpu')

    model = ImageEncoder(3, 200).to(device)
    summary(model, (3, 64, 64), device='cpu')

    model = ImageDecoder(1, 200).to(device)
    summary(model, (200,), device='cpu')
