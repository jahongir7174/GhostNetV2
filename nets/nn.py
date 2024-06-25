import math
import torch


def round_width(width):
    divisor = 4
    new_width = max(divisor, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return int(new_width)


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k, s, p, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class SE(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1709.01507.pdf]
    """

    def __init__(self, ch):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Conv2d(ch, round_width(ch // 4), kernel_size=1),
                                      torch.nn.ReLU(),
                                      torch.nn.Conv2d(round_width(ch // 4), ch, kernel_size=1),
                                      torch.nn.Hardsigmoid())

    def forward(self, x):
        return x * self.se(x)


class Block(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, attn=False):
        super().__init__()
        self.attn = attn
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        if self.attn:
            self.conv1 = Conv(in_ch, out_ch // 2, activation, k=1, s=1, p=0)
            self.conv2 = Conv(out_ch // 2, out_ch // 2, activation, k=3, s=1, p=1, g=out_ch // 2)
            self.conv3 = torch.nn.Sequential(Conv(in_ch, out_ch, torch.nn.Identity(), k=1, s=1, p=0),
                                             Conv(out_ch, out_ch, torch.nn.Identity(),
                                                  k=(1, 5), s=1, p=(0, 2), g=out_ch),
                                             Conv(out_ch, out_ch, torch.nn.Identity(),
                                                  k=(5, 1), s=1, p=(2, 0), g=out_ch))
        else:
            self.conv1 = Conv(in_ch, out_ch // 2, activation, k=1, s=1, p=0)
            self.conv2 = Conv(out_ch // 2, out_ch // 2, activation, k=3, s=1, p=1, g=out_ch // 2)

    def forward(self, x):
        if self.attn:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            out = torch.cat(tensors=[x1, x2], dim=1)
            return out * torch.nn.functional.interpolate(self.conv3(self.pool(x)).sigmoid(),
                                                         size=(out.shape[-2], out.shape[-1]))
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return torch.cat(tensors=[x1, x2], dim=1)


class Residual(torch.nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch,
                 k=3, s=1, se=False, attn=False):
        super().__init__()
        # Point-wise Expansion
        self.conv1 = Block(in_ch, mid_ch, torch.nn.ReLU(), attn)

        # Depth-wise Convolution
        if s == 1:
            self.conv2 = torch.nn.Identity()
        else:
            self.conv2 = Conv(mid_ch, mid_ch, torch.nn.Identity(), k, s, (k - 1) // 2, mid_ch)

        # Squeeze-and-Excitation
        self.se = SE(mid_ch) if se else torch.nn.Identity()

        self.conv3 = Block(mid_ch, out_ch, torch.nn.Identity(), attn=False)

        # Shortcut
        if in_ch == out_ch and s == 1:
            self.conv4 = torch.nn.Identity()
        else:
            self.conv4 = torch.nn.Sequential(Conv(in_ch, in_ch, torch.nn.Identity(), k, s, (k - 1) // 2, in_ch),
                                             Conv(in_ch, out_ch, torch.nn.Identity(), k=1, s=1, p=0, g=1))

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        y = self.conv3(y)
        return y + self.conv4(x)


class GhostNetV2(torch.nn.Module):
    """"
    GhostNetV2-1.0 -> width=1.0
    GhostNetV2-1.3 -> width=1.3
    GhostNetV2-1.6 -> width=1.6
    """

    def __init__(self, width=1.0):
        super().__init__()
        se = [True, False]
        out = [3, 16, 24, 40, 80, 112, 160, 960, 1280, 1000]
        mid = [16, 48, 72, 120, 240, 200, 184, 480, 672, 960]

        out[1:8] = [round_width(width * i) for i in out[1:8]]
        mid = [round_width(width * i) for i in mid]

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(out[0], out[1], torch.nn.ReLU(), k=3, s=2, p=1))
        self.p1.append(Residual(out[1], mid[0], out[1], k=3, s=1, se=se[1]))
        # p2/4
        self.p2.append(Residual(out[1], mid[1], out[2], k=3, s=2, se=se[1]))
        self.p2.append(Residual(out[2], mid[2], out[2], k=3, s=1, se=se[1], attn=True))
        # p3/8
        self.p3.append(Residual(out[2], mid[2], out[3], k=5, s=2, se=se[0], attn=True))
        self.p3.append(Residual(out[3], mid[3], out[3], k=5, s=1, se=se[0], attn=True))
        # p4/16
        self.p4.append(Residual(out[3], mid[4], out[4], k=3, s=2, se=se[1], attn=True))
        self.p4.append(Residual(out[4], mid[5], out[4], k=3, s=1, se=se[1], attn=True))
        self.p4.append(Residual(out[4], mid[6], out[4], k=3, s=1, se=se[1], attn=True))
        self.p4.append(Residual(out[4], mid[6], out[4], k=3, s=1, se=se[1], attn=True))
        self.p4.append(Residual(out[4], mid[7], out[5], k=3, s=1, se=se[0], attn=True))
        self.p4.append(Residual(out[5], mid[8], out[5], k=3, s=1, se=se[0], attn=True))
        # p5/32
        self.p5.append(Residual(out[5], mid[8], out[6], k=5, s=2, se=se[0], attn=True))
        self.p5.append(Residual(out[6], mid[9], out[6], k=5, s=1, se=se[1], attn=True))
        self.p5.append(Residual(out[6], mid[9], out[6], k=5, s=1, se=se[0], attn=True))
        self.p5.append(Residual(out[6], mid[9], out[6], k=5, s=1, se=se[1], attn=True))
        self.p5.append(Residual(out[6], mid[9], out[6], k=5, s=1, se=se[0], attn=True))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.fc = torch.nn.Sequential(Conv(out[6], out[7], torch.nn.ReLU(), k=1, s=1, p=0),
                                      torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Conv2d(out[7], out[8], kernel_size=1),
                                      torch.nn.ReLU(),
                                      torch.nn.Flatten(),
                                      torch.nn.Dropout(0.2),
                                      torch.nn.Linear(out[8], out[9]))

        # initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out = fan_out * m.dilation[0] * m.dilation[1] // m.groups
                torch.nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            if isinstance(m, torch.nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.size()[0])
                torch.nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        return self.fc(x)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
