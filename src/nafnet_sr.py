"""
NAFNet-SR: standalone single-file NAFNet adapted for joint denoise + 2x super-resolution
on grayscale semiconductor inspection images.

No BasicSR dependency. Runs on CUDA, Apple MPS, or CPU.

Architecture is Chen et al., "Simple Baselines for Image Restoration" (ECCV 2022),
https://github.com/megvii-research/NAFNet  -- MIT licensed.

Adaptations for the KLA challenge:
  * 1-channel grayscale input, optional log-domain second channel (speckle is
    multiplicative; log turns it additive)
  * PixelShuffle SR head so the body runs at LR resolution (cheap) and outputs HR
  * Global bicubic residual skip -- network predicts a correction, not the image,
    which holds up much better on out-of-distribution sources
  * Per-instance normalisation helpers using median/MAD (robust to speckle tails)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks (vendored from NAFNet, BasicSR imports removed)
# --------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Normalises across the channel dim at each spatial location.

    This is NAFNet's own LayerNorm2d, not GroupNorm(1, C) -- the difference
    matters. Because statistics are computed per-image, the network is partly
    invariant to global brightness/contrast shifts between data sources.
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)

        # Simplified Channel Attention -- no sigmoid, weights may exceed 1
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


# --------------------------------------------------------------------------
# NAFNet-SR
# --------------------------------------------------------------------------

class NAFNetSR(nn.Module):
    """NAFNet U-Net body at LR resolution + PixelShuffle SR head.

    Args:
        img_channel:    input channels before the log channel is added (1 = grayscale)
        width:          base width. 32 is the sane hackathon default; 64 if you have GPU budget
        enc_blk_nums:   NAFBlocks per encoder stage
        middle_blk_num: NAFBlocks at the bottleneck
        dec_blk_nums:   NAFBlocks per decoder stage
        scale:          upsampling factor (2 for 256->512 and 128->256)
        use_log_channel: concat log1p(x) as a second channel (recommended for speckle)
    """

    def __init__(self, img_channel=1, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2),
                 scale=2, use_log_channel=True):
        super().__init__()
        self.scale = scale
        self.use_log_channel = use_log_channel
        self.out_channel = img_channel

        in_ch = img_channel * 2 if use_log_channel else img_channel
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        # SR head: width -> img_channel * scale^2 -> PixelShuffle -> img_channel at HR
        self.sr_head = nn.Sequential(
            nn.Conv2d(chan, img_channel * (scale ** 2), 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
        )
        # Start as a near-identity correction so early training is stable
        nn.init.zeros_(self.sr_head[0].weight)
        nn.init.zeros_(self.sr_head[0].bias)

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        """inp: (B, img_channel, H, W) float, NOT clamped to [0,1].

        Speckle legitimately pushes values above the ground-truth range -- clamping
        the input throws that information away. Returns (B, img_channel, H*scale, W*scale).
        """
        B, C, H, W = inp.shape

        if self.use_log_channel:
            # SIGNED log. A plain log1p(clamp_min(0)) would flatten every negative
            # pixel to zero -- and the negatives are exactly where the additive
            # static shows itself, since multiplicative grain can never produce them.
            log_ch = torch.sign(inp) * torch.log1p(inp.abs())
            x_in = torch.cat([inp, log_ch], dim=1)
        else:
            x_in = inp

        x_in = self.check_image_size(x_in)
        x = self.intro(x_in)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        residual = self.sr_head(x)
        residual = residual[:, :, :H * self.scale, :W * self.scale]

        base = F.interpolate(inp, scale_factor=self.scale,
                             mode='bicubic', align_corners=False)
        return base + residual

    def check_image_size(self, x):
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # reflect padding beats zero padding at borders for restoration
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x


# --------------------------------------------------------------------------
# Robust per-instance normalisation (helps a lot on OOD sources)
# --------------------------------------------------------------------------

def robust_normalize(x, eps=1e-6):
    """Median/MAD normalisation per image. Returns (normalized, center, scale).

    Median and MAD are used instead of mean/std because speckle skews the tails
    badly. De-normalise the model output with the SAME center/scale.
    """
    B = x.shape[0]
    flat = x.reshape(B, -1)
    center = flat.median(dim=1, keepdim=True).values
    mad = (flat - center).abs().median(dim=1, keepdim=True).values
    scale = (mad * 1.4826).clamp_min(eps)  # 1.4826 * MAD ~= std for Gaussian
    center = center.view(B, 1, 1, 1)
    scale = scale.view(B, 1, 1, 1)
    return (x - center) / scale, center, scale


def robust_denormalize(y, center, scale):
    return y * scale + center


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Smooth L1. More robust than L2, sharper results than plain L1."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class FFTLoss(nn.Module):
    """L1 in the frequency domain.

    Directly penalises the "blur it to kill the noise" failure mode, which the
    challenge brief calls out explicitly. Weight it around 0.05-0.1.
    """

    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred.float(), norm='ortho')
        targ_f = torch.fft.rfft2(target.float(), norm='ortho')
        return (torch.view_as_real(pred_f) - torch.view_as_real(targ_f)).abs().mean()


class RestorationLoss(nn.Module):
    def __init__(self, fft_weight=0.05):
        super().__init__()
        self.char = CharbonnierLoss()
        self.fft = FFTLoss()
        self.fft_weight = fft_weight

    def forward(self, pred, target):
        return self.char(pred, target) + self.fft_weight * self.fft(pred, target)


# --------------------------------------------------------------------------

def get_device():
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(size='w32', scale=2):
    cfg = {
        'w16': dict(width=16, middle_blk_num=4, enc_blk_nums=(1, 1, 2, 4), dec_blk_nums=(1, 1, 1, 1)),
        'w32': dict(width=32, middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)),
        'w64': dict(width=64, middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)),
    }[size]
    return NAFNetSR(img_channel=1, scale=scale, **cfg)


if __name__ == '__main__':
    device = get_device()
    print(f'device: {device}')

    for size in ['w16', 'w32']:
        net = build_model(size).to(device)
        n_params = sum(p.numel() for p in net.parameters())

        # 128x128 LR -> 256x256 HR, with values deliberately exceeding 1.0
        x = torch.rand(2, 1, 128, 128, device=device) * 1.3
        with torch.no_grad():
            y = net(x)
        print(f'{size}: {n_params/1e6:.2f}M params | {tuple(x.shape)} -> {tuple(y.shape)}')

    # odd input size (OOD test images are rarely nice round numbers)
    net = build_model('w16').to(device)
    x = torch.rand(1, 1, 137, 91, device=device)
    with torch.no_grad():
        y = net(x)
    print(f'odd size: {tuple(x.shape)} -> {tuple(y.shape)}')

    # loss + backward
    net = build_model('w16').to(device)
    crit = RestorationLoss()
    x = torch.rand(2, 1, 64, 64, device=device) * 1.2
    gt = torch.rand(2, 1, 128, 128, device=device)
    loss = crit(net(x), gt)
    loss.backward()
    gnorm = sum(p.grad.norm().item() ** 2 for p in net.parameters() if p.grad is not None) ** 0.5
    print(f'loss: {loss.item():.4f} | grad norm: {gnorm:.4f}')

    # robust normalisation round-trip
    xn, c, s = robust_normalize(x)
    err = (robust_denormalize(xn, c, s) - x).abs().max().item()
    print(f'normalize round-trip max err: {err:.2e}')
