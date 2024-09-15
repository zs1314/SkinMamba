import torch
import torch.nn as nn
import torch.nn.functional as F


# FFT
class FFML(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(FFML, self).__init__()
        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=1, stride=1)
        )
        self.norm = norm
        self.act=nn.Sigmoid()
    def forward(self, x):
        _, _, H, W = x.shape

        dim = 1

        y = torch.fft.rfft2(x, norm=self.norm)

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)

        y = self.main_fft(y_f)

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y=self.act(y)
        y=y*x
        return y


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# FC
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)

class FFMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.fft = FFML(dim)

        # Feedforward layer
        self.fc = FC(dim, ffn_scale)

    def forward(self, x):
        y = self.norm1(x)

        y = self.fft(y)

        y = self.fc(self.norm2(y)) + y
        return y

