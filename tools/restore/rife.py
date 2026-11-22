"""Frame interpolation with RIFE 4.25 (Practical-RIFE, MIT licence).

The network below is IFNet_HDv3 from hzwer/Practical-RIFE v4.25, trimmed to
inference; `warp` replaces the repo's model/warplayer.py. Weights:
RIFE_v4.25.zip -> train_log/flownet.pkl (loaded as plain tensors,
weights_only=True).

Huang et al., "Real-Time Intermediate Flow Estimation for Video Frame
Interpolation", ECCV 2022.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_grid_cache = {}


def warp(img, flow):
    key = (flow.shape, flow.dtype)
    if key not in _grid_cache:
        n, _, h, w = flow.shape
        gx = torch.linspace(-1.0, 1.0, w, dtype=flow.dtype).view(1, 1, 1, w).expand(n, -1, h, -1)
        gy = torch.linspace(-1.0, 1.0, h, dtype=flow.dtype).view(1, 1, h, 1).expand(n, -1, -1, w)
        _grid_cache[key] = torch.cat([gx, gy], 1)
    w_, h_ = img.shape[3], img.shape[2]
    f = torch.cat([flow[:, 0:1] / ((w_ - 1.0) / 2.0), flow[:, 1:2] / ((h_ - 1.0) / 2.0)], 1)
    g = (_grid_cache[key] + f).permute(0, 2, 3, 1)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)


def conv(cin, cout, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, cin, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(conv(cin, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(self, img0, img1, timestep, scale_list, flow_only=False):
        t = (img0[:, :1] * 0 + 1) * timestep
        f0, f1 = self.encode(img0), self.encode(img1)
        flow = mask = feat = None
        w0, w1 = img0, img1
        for i, blk in enumerate([self.block0, self.block1, self.block2, self.block3, self.block4]):
            if flow is None:
                flow, mask, feat = blk(torch.cat((img0, img1, f0, f1, t), 1), None, scale=scale_list[i])
            else:
                wf0, wf1 = warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4])
                fd, mask, feat = blk(torch.cat((w0, w1, wf0, wf1, t, mask, feat), 1), flow, scale=scale_list[i])
                flow = flow + fd
            w0, w1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        if flow_only:
            return flow, m
        return w0 * m + w1 * (1 - m)


class Rife:
    """Interpolate between two RGB frames (float tensors 1x3xHxW in [0,1])."""

    def __init__(self, weights, scale=1.0, threads=4):
        torch.set_num_threads(threads)
        self.net = IFNet()
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", ""): v for k, v in sd.items() if not k.startswith(("teacher", "caltime"))}
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        assert not missing, missing
        self.net.eval()
        self.scale = scale

    @torch.inference_mode()
    def __call__(self, img0, img1, t):
        _, _, h, w = img0.shape
        m = max(128, int(128 / self.scale))
        ph, pw = ((h - 1) // m + 1) * m, ((w - 1) // m + 1) * m
        pad = (0, pw - w, 0, ph - h)
        a, b = F.pad(img0, pad), F.pad(img1, pad)
        s = self.scale
        out = self.net(a, b, t, [16 / s, 8 / s, 4 / s, 2 / s, 1 / s])
        return out[:, :, :h, :w].clamp(0, 1)

    @torch.inference_mode()
    def hires(self, img0, img1, t, down=2):
        """Motion (flow + blend mask) from frames `down` times smaller, applied at full size.

        Same result as scale=1/down for the motion, but the encoder and the
        per-block warps run on the small frames; only the last warp and the
        blend touch the full-resolution pixels.
        """
        _, _, h, w = img0.shape
        a = F.interpolate(img0, scale_factor=1 / down, mode="area")
        b = F.interpolate(img1, scale_factor=1 / down, mode="area")
        _, _, hs, ws = a.shape
        m = 128
        ph, pw = ((hs - 1) // m + 1) * m, ((ws - 1) // m + 1) * m
        pad = (0, pw - ws, 0, ph - hs)
        flow, mask = self.net(F.pad(a, pad), F.pad(b, pad), t, [16, 8, 4, 2, 1], flow_only=True)
        flow = flow[:, :, :hs, :ws]
        mask = mask[:, :, :hs, :ws]
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False) * down
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        w0, w1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        return (w0 * mask + w1 * (1 - mask)).clamp(0, 1)
