import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn import GATConv
from dgl.nn import SAGEConv
from functools import partial



def extract(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()
        self.in_dim = 128
        self.num_hidden = 128
        self.num_layers = 2
        self.nhead = 4
        self.feat_drop = 0.1
        self.attn_drop = 0.1
        self.negative_slope = 0.2
        self.alpha_l = 2
        self.beta_schedule = 'linear'
        self.beta_1 = 0.0001
        self.beta_T = 0.02
        self.T = 1000
        
        beta = get_beta_schedule(self.beta_schedule, self.beta_1, self.beta_T, self.T)
        self.register_buffer(
                'betas', beta
                )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
                'sqrt_alphas_bar', torch.sqrt(alphas_bar)
                )
        self.register_buffer(
                'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
                )

        self.net = Denoising_GNN(in_dim=self.in_dim,
                                  num_hidden=self.num_hidden,
                                  out_dim=self.in_dim,
                                  num_layers=self.num_layers,
                                  nhead=self.nhead,
                                  feat_drop=self.feat_drop,
                                  attn_drop=self.attn_drop,
                                  negative_slope=self.negative_slope
                                  )

        self.time_embedding = nn.Embedding(self.T, self.num_hidden)
        self.norm_x = nn.LayerNorm(self.in_dim, elementwise_affine=False)

    def forward(self, g, x, y):
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1], ))

        t = torch.randint(self.T, size=(x.shape[0], ), device=x.device)
        x_t, time_embed, g  = self.sample_q(t, x, g)

        loss = self.node_denoising(y, x_t, time_embed, g)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def sample_q(self, t, x, g):
        noise = torch.randn_like(x, device=x.device)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x.shape) * x +
                extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
                )
        time_embed = self.time_embedding(t)
        return x_t, time_embed, g

    def node_denoising(self, y, x_t, time_embed, g):
        out, _ = self.net(g, x_t=x_t, time_embed=time_embed)
        loss = loss_fn(out, y, self.alpha_l)

        return loss

    def embed(self, g, x, T):
        t = torch.full((1, ), T, device=x.device)
        with torch.no_grad():
            x = F.layer_norm(x, (x.shape[-1], ))
        x_t, time_embed, g = self.sample_q(t, x, g)
        _, hidden = self.net(g, x_t=x_t, time_embed=time_embed)
        return hidden


def loss_fn(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)
    


    
class Denoising_GNN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 feat_drop,
                 attn_drop,
                 negative_slope
                 ):
        super(Denoising_GNN, self).__init__()
        self.in_dim=in_dim
        self.num_hidden=num_hidden
        self.out_dim = out_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.feat_drop=feat_drop
        self.attn_drop=attn_drop
        self.negative_slope=negative_slope
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.mlp_in_t = MlpBlock(in_dim=self.in_dim, hidden_dim=self.num_hidden*2, out_dim=self.num_hidden)

        self.mlp_middle = MlpBlock(self.num_hidden, self.num_hidden, self.num_hidden)

        self.mlp_out = MlpBlock(self.num_hidden, self.out_dim, self.out_dim)        

        self.down_layers.append(SAGEConv(num_hidden, num_hidden,'mean'))
        self.up_layers.append(SAGEConv(num_hidden, num_hidden,'mean'))



        for _ in range(1, self.num_layers):
            self.down_layers.append(SAGEConv(num_hidden, num_hidden,'mean'))
            self.up_layers.append(SAGEConv(num_hidden, num_hidden,'mean'))

        self.up_layers = self.up_layers[::-1]

    def forward(self, g, x_t, time_embed):
        h_t = self.mlp_in_t(x_t)
        down_hidden = []
        for l in range(self.num_layers):
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)
            else:
                pass
            h_t = self.down_layers[l](g, h_t)
            h_t = h_t.flatten(1)
            down_hidden.append(h_t)
        h_middle = self.mlp_middle(h_t)

        h_t = h_middle
        out_hidden = []
        for l in range(self.num_layers):
            h_t = h_t + down_hidden[self.num_layers - l - 1 ]
            if h_t.ndim > 2:
                h_t = h_t + time_embed.unsqueeze(1).repeat(1, h_t.shape[1], 1)
            else:
                pass
            h_t = self.up_layers[l](g, h_t)
            h_t = h_t.flatten(1)
            out_hidden.append(h_t)
        out = self.mlp_out(h_t)
        out_hidden = torch.cat(out_hidden, dim=-1)

        return out, out_hidden



class Residual(nn.Module):
    def __init__(self, fnc):
        super().__init__()
        self.fnc = fnc

    def forward(self, x, *args, **kwargs):
        return self.fnc(x, *args, **kwargs) + x


class MlpBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 norm: str = 'layernorm', activation: str = 'relu'):
        super(MlpBlock, self).__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_mlp = Residual(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              create_norm(norm)(hidden_dim),
                                              create_activation(activation),
                                              nn.Linear(hidden_dim, hidden_dim)))
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.act = create_activation(activation)
    def forward(self, x):
        x = self.in_proj(x)
        x = self.res_mlp(x)
        x = self.out_proj(x)
        x = self.act(x)
        return x    


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


    