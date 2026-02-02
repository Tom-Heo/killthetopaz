from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


class Heo:
    class HeLU(nn.Module):
        """
        원본 HeLU: last-dim 기반 (..., dim) 입력용
        """

        def __init__(self, dim: int):
            super().__init__()

            self.alpha = nn.Parameter(torch.full((dim,), 0.9))
            self.beta = nn.Parameter(torch.full((dim,), -0.9))
            self.redweight = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=0.45))
            self.blueweight = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=0.45))
            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor):
            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            red = torch.tanh(sqrt(3.0) * self.redweight) + 1.0
            blue = torch.tanh(sqrt(3.0) * self.blueweight) + 1.0
            redx = rgx * red
            bluex = bgx * blue
            x = redx + bluex
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            y = (alpha * x + beta * raw) / 2
            return y

    class HeLU2d(nn.Module):
        """
        입력: (N,C,H,W)
        """

        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            # 원본 HeLU와 같은 파라미터 의미(채널별)
            self.alpha = nn.Parameter(torch.full((c,), 0.9))
            self.beta = nn.Parameter(torch.full((c,), -0.9))
            self.redweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.45))
            self.blueweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.45))

            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4:
                raise ValueError(
                    f"HeLU2d expects NCHW 4D tensor, got shape={tuple(x.shape)}"
                )
            if x.size(1) != self.channels:
                raise ValueError(
                    f"HeLU2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            # (C,) -> (1,C,1,1) broadcasting
            red = (torch.tanh(sqrt(3.0) * self.redweight) + 1.0).view(1, -1, 1, 1)
            blue = (torch.tanh(sqrt(3.0) * self.blueweight) + 1.0).view(1, -1, 1, 1)
            x = rgx * red + bgx * blue

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            y = (alpha * x + beta * raw) / 2
            return y

    class HeoGate(nn.Module):
        def __init__(self, dim: int):
            super().__init__()

            self.alpha = nn.Parameter(torch.full((dim,), 0.9))
            self.beta = nn.Parameter(torch.full((dim,), -0.9))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            return (alpha * x + beta * raw) / 2

    class HeoGate2d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            self.alpha = nn.Parameter(torch.full((c,), 0.9))
            self.beta = nn.Parameter(torch.full((c,), -0.9))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            if x.dim() != 4 or x.size(1) != self.channels:
                raise ValueError(
                    f"HeoGate2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            return (alpha * x + beta * raw) / 2

    class OklabtoBakedColor(nn.Module):
        def __init__(self):
            super().__init__()
            """
            Normalized Oklab (Lp, ap, bp)
            Lp = 2*L - 1
            ap = 2*a
            bp = 2*b
            
            Macbeth ColorChecker 24 Reference Points (Float32 Precision)
            Lp, ap, bp 좌표계의 절대 기준점
            """
            MACBETH_REFS = [
                [-0.0597942, 0.0715682, 0.0689456],
                [0.4183730, 0.0844511, 0.0862138],
                [0.1485260, -0.0260145, -0.1197130],
                [0.0060500, -0.0867117, 0.1012940],
                [0.2456150, 0.0471529, -0.1390040],
                [0.4774960, -0.1772700, 0.0070673],
                [0.3507240, 0.1477280, 0.2429970],
                [-0.0020556, 0.0205867, -0.2380310],
                [0.1933670, 0.2535120, 0.0753901],
                [-0.1681210, 0.1244790, -0.1221760],
                [0.4962120, -0.1630230, 0.2584340],
                [0.5099230, 0.0561745, 0.2807750],
                [-0.1797940, 0.0282334, -0.2856320],
                [0.1974370, -0.2180970, 0.1560710],
                [0.0268793, 0.2907220, 0.1168620],
                [0.6649640, -0.0396422, 0.3292740],
                [0.1965930, 0.2874640, -0.0811799],
                [0.1415920, -0.1571350, -0.1312720],
                [0.9277470, -0.0007701, 0.0023781],
                [0.6655870, -0.0000168, -0.0001434],
                [0.4115140, -0.0000142, -0.0001216],
                [0.1583850, -0.0008645, 0.0027884],
                [-0.1009340, -0.0000091, -0.0000774],
                [-0.3499260, -0.0000066, -0.0000560],
            ]

            # 모델의 State로 등록 (Device 이동 자동화)
            self.register_buffer(
                "macbeth_refs",
                torch.tensor(MACBETH_REFS, dtype=torch.float32).view(1, 24, 3, 1, 1),
            )

            # 2. RBF Coefficients (Geometric Progression)
            # 0.1 (Micro) -> 0.4 (Meso) -> 1.6 (Macro)
            self.coeffs = [0.1, 0.4, 1.6]
            self.conv1 = nn.Conv2d(96, 96, 1, 1)
            self.conv2 = nn.Conv2d(96, 96, 1, 1)
            self.conv3 = nn.Conv2d(96, 96, 1, 1)
            self.act1 = Heo.HeLU2d(96)
            self.act2 = Heo.HeLU2d(96)
            self.act3 = Heo.HeLU2d(96)
            self.gate1 = Heo.HeoGate2d(96)
            self.gate2 = Heo.HeoGate2d(96)
            self.gate3 = Heo.HeoGate2d(96)

        def forward(self, x):
            """
            Input:  x (B, 3, H, W) - Normalized Oklab (Lp, ap, bp)
            Lp = 2*L - 1
            ap = 2*a
            bp = 2*b
            Output: out (B, 96, H, W)
            """
            # --- Part 1: Signal Replication (24 Channels) ---
            # 원본 신호의 흐름을 강화하기 위해 8회 반복
            part1 = x.repeat(1, 8, 1, 1)

            # --- Part 2: RBF Global Context (72 Channels) ---
            # 1. Squared Euclidean Distance Calculation
            # (B, 1, 3, H, W) - (1, 24, 3, 1, 1) Broadcasting
            diff = x.unsqueeze(1) - self.macbeth_refs
            dist_sq = (diff**2).sum(dim=2)  # (B, 24, H, W)

            # 2. Gaussian RBF Application
            rbf_features = []
            for sigma in self.coeffs:
                # Gaussian Definition: exp( - d^2 / (2 * sigma^2) )
                gamma = 1.0 / (2.0 * (sigma**2))
                rbf = torch.exp(-dist_sq * gamma)
                rbf_features.append(rbf)

            part2 = torch.cat(rbf_features, dim=1)

            # --- Final Concatenation (96 Channels) ---
            features = torch.cat([part1, part2], dim=1)

            x1 = self.conv1(features)
            x1 = self.act1(x1)
            x1 = self.gate1(features, x1)

            x2 = self.conv2(x1)
            x2 = self.act2(x2)
            x2 = self.gate2(x1, x2)

            x3 = self.conv3(x2)
            x3 = self.act3(x3)
            x3 = self.gate3(x2, x3)

            return x3

    class sRGBtoOklab(nn.Module):
        """
        Oklab 변환 (sRGB in [0,1] 가정).
        출력 채널: Lp, ap, bp (네가 사용 중인 스케일):
          Lp = 2*L - 1
          ap = 2*a
          bp = 2*b
        """

        def __init__(self):
            super().__init__()

        @staticmethod
        def srgb_to_lsrgb(srgb: torch.Tensor) -> torch.Tensor:
            return torch.where(
                srgb <= 0.04045,
                srgb / 12.92,
                ((srgb + 0.055) / 1.055) ** 2.4,
            )

        @staticmethod
        def lsrgb_to_oklab(
            lsred: torch.Tensor, lsgreen: torch.Tensor, lsblue: torch.Tensor
        ):
            def cbrt(x: torch.Tensor) -> torch.Tensor:
                return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)

            l = 0.4122214708 * lsred + 0.5363325363 * lsgreen + 0.0514459929 * lsblue
            m = 0.2119034982 * lsred + 0.6806995451 * lsgreen + 0.1073969566 * lsblue
            s = 0.0883024619 * lsred + 0.2817188376 * lsgreen + 0.6299787005 * lsblue

            l = cbrt(l)
            m = cbrt(m)
            s = cbrt(s)

            oklab_L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
            oklab_a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
            oklab_b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s

            # 범위 확장 (네 스케일)
            Lp = (2 * oklab_L) - 1.0
            ap = 2 * oklab_a
            bp = 2 * oklab_b

            return Lp, ap, bp

        def forward(self, x: torch.Tensor):
            # x: (B,3,H,W) assumed in [0,1]
            if x.dim() != 4 or x.size(1) != 3:
                raise ValueError(f"Oklab expects (B,3,H,W), got shape={tuple(x.shape)}")

            x = x.permute(0, 2, 3, 1)  # NHWC
            srgb = x.clamp(0.0, 1.0)
            lsrgb = self.srgb_to_lsrgb(srgb)

            lsred = lsrgb[..., 0:1]
            lsgreen = lsrgb[..., 1:2]
            lsblue = lsrgb[..., 2:3]

            Lp, ap, bp = self.lsrgb_to_oklab(lsred, lsgreen, lsblue)
            nhwc = torch.cat([Lp, ap, bp], dim=-1)
            nchw = nhwc.permute(0, 3, 1, 2)
            return nchw

    class OklabtosRGB(nn.Module):
        """
        Oklab(Lp,ap,bp 스케일) -> sRGB [0,1]
        입력: (B,3,H,W)
          Lp∈[-1,1], ap,bp는 기존 Oklab 스케일의 2배
        출력: (B,3,H,W), sRGB [0,1]
        """

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4 or x.size(1) != 3:
                raise ValueError(f"RGB expects (B,3,H,W), got shape={tuple(x.shape)}")

            # 스케일 복원: L = (Lp+1)/2, a = ap/2, b = bp/2
            Lp, ap, bp = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            L = (Lp + 1.0) * 0.5
            a = ap * 0.5
            b = bp * 0.5

            # Oklab -> LMS^3
            l_ = L + 0.3963377774 * a + 0.2158037573 * b
            m_ = L - 0.1055613458 * a - 0.0638541728 * b
            s_ = L - 0.0894841775 * a - 1.2914855480 * b

            l = l_**3
            m = m_**3
            s = s_**3

            # LMS -> linear sRGB
            r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
            g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
            b_rgb = 0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
            srgb_lin = torch.cat([r, g, b_rgb], dim=1)

            # linear sRGB -> sRGB 감마, [0,1]
            threshold = 0.0031308
            srgb = torch.where(
                srgb_lin <= threshold,
                12.92 * srgb_lin,
                1.055 * torch.clamp(srgb_lin, min=0.0) ** (1.0 / 2.4) - 0.055,
            )
            return srgb.clamp(0.0, 1.0)

    class HeoTimeEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                Heo.HeLU(dim * 4),
                nn.Linear(dim * 4, dim * 4),
                Heo.HeLU(dim * 4),
                nn.Linear(dim * 4, dim),
            )

        def forward(self, time):
            # Sinusoidal embedding
            device = time.device
            half_dim = self.dim // 2
            embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = time[:, None] * embeddings[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
            return self.mlp(embeddings)

    class NeMO33(nn.Module):
        def __init__(self, dim: int):
            super().__init__()

            self.conv0 = nn.Conv2d(dim, dim, 3, 1, 1) # Padding added
            self.conv1 = nn.Conv2d(dim, dim, 1, 1)
            self.conv2 = nn.Conv2d(dim, dim, 1, 1)
            self.conv3 = nn.Conv2d(dim, dim, 1, 1)

            self.act0 = Heo.HeLU2d(dim)
            self.act1 = Heo.HeLU2d(dim)
            self.act2 = Heo.HeLU2d(dim)
            self.act3 = Heo.HeLU2d(dim)

            self.gate0 = Heo.HeoGate2d(dim)
            self.gate1 = Heo.HeoGate2d(dim)
            self.gate2 = Heo.HeoGate2d(dim)
            self.gate3 = Heo.HeoGate2d(dim)

        def forward(self, x: torch.Tensor):

            raw = x
            x = self.conv0(x)
            x = self.act0(x)
            x = self.gate0(x, raw)

            x0 = x
            x = self.conv1(x)
            x = self.act1(x)
            x = self.gate1(x0, x)

            x1 = x
            x = self.conv2(x)
            x = self.act2(x)
            x = self.gate2(x1, x)

            x2 = x
            x = self.conv3(x)
            x = self.act3(x)
            x = self.gate3(x2, x)

            return x

    class BakeLoss(nn.Module):
        def __init__(self, epsilon=0.001):
            super().__init__()
            self.epsilon = epsilon
            self.epsilon_char = 1e-8

        def forward(self, pred, target):
            pred = pred.float()  # FP32 강제
            target = target.float()

            diff = pred - target
            charbonnier = torch.sqrt(diff**2 + self.epsilon_char**2)

            loss = torch.log(1 + charbonnier / self.epsilon)

            return loss.mean()

    class BakeDDPM(nn.Module):
        """
        Cosine Schedule과 BakeLoss를 사용하는 SR용 DDPM Wrapper
        Prediction Target: x_0 (Clean Image)
        """

        def __init__(self, network, timesteps=1000, loss_epsilon=0.001):
            """
            Args:
                network: x_0를 예측하는 Network
                timesteps: 확산 단계 수 (기본 1000)
                loss_epsilon: BakeLoss의 epsilon (기본 0.001)
            """
            super().__init__()
            self.network = network
            self.timesteps = timesteps

            # 1. Loss Function 설정
            self.criterion = Heo.BakeLoss(epsilon=loss_epsilon)
            
            # 2. BakedColor Converter
            self.to_baked = Heo.OklabtoBakedColor()

            # 3. Beta Schedule 설정 (Cosine 고정)
            betas = self._get_cosine_schedule(timesteps)

            # 4. 확산 계수 등록 (Buffer)
            self.register_buffer("betas", betas)

            alphas = 1.0 - betas
            self.register_buffer("alphas", alphas)

            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.register_buffer("alphas_cumprod", alphas_cumprod)

            # 학습용 계수: q(x_t | x_0)
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer(
                "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
            )

            # 샘플링용 계수: Posterior Mean coefficients
            # mu_t = coef1 * x_0 + coef2 * x_t
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            
            self.register_buffer(
                "posterior_mean_coef1",
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            )
            self.register_buffer(
                "posterior_mean_coef2",
                (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
            )
            
            self.register_buffer(
                "posterior_variance",
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            )

        def _get_cosine_schedule(self, timesteps, s=0.008):
            """
            Cosine Beta Schedule 생성 함수
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0, 0.999)

        def forward(self, x_hr, x_lr):
            """
            Training Step:
            x_hr (Target), x_lr (Condition) -> BakeLoss on BakedColor space
            """
            batch_size, device = x_hr.shape[0], x_hr.device

            # 1. 랜덤 타임스텝 t
            t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

            # 2. 정답 노이즈 생성
            noise = torch.randn_like(x_hr)

            # 3. x_t (Noisy Image) 생성
            sqrt_ab = self.sqrt_alphas_cumprod[t][:, None, None, None]
            sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t][
                :, None, None, None
            ]

            x_noisy = sqrt_ab * x_hr + sqrt_one_minus_ab * noise

            # 4. 모델 예측 (Predict x_0)
            x0_pred = self.network(x_noisy, x_lr, t)

            # 5. BakedColor Space 변환 및 Loss 계산
            # x0_pred와 x_hr을 BakedColor 공간으로 변환
            pred_baked = self.to_baked(x0_pred)
            target_baked = self.to_baked(x_hr)
            
            loss = self.criterion(pred_baked, target_baked)
            return loss

        @torch.no_grad()
        def sample(self, x_lr, shape=None):
            """
            Inference Step: x_lr 조건으로 고해상도 이미지 생성 (predicting x_0)
            """
            device = x_lr.device
            B = x_lr.shape[0]

            # 출력 크기 설정 (shape 미지정 시 x_lr과 동일 - Pre-upsample 가정)
            if shape is None:
                C, H, W = x_lr.shape[1], x_lr.shape[2], x_lr.shape[3]
            else:
                C, H, W = shape

            # 1. 완전 노이즈 시작
            img = torch.randn((B, C, H, W), device=device)

            # 2. 역방향 루프
            for i in reversed(range(0, self.timesteps)):
                t = torch.full((B,), i, device=device, dtype=torch.long)

                # x_0 예측
                x0_pred = self.network(img, x_lr, t)
                
                # Posterior Mean 계산
                coef1 = self.posterior_mean_coef1[t][:, None, None, None]
                coef2 = self.posterior_mean_coef2[t][:, None, None, None]
                
                mean = coef1 * x0_pred + coef2 * img
                
                # 분산 추가 (마지막 단계 제외)
                if i > 0:
                    noise = torch.randn_like(img)
                    variance = self.posterior_variance[t][:, None, None, None]
                    sigma = torch.sqrt(variance)
                    img = mean + sigma * noise
                else:
                    img = mean

            return img

    class BakeTimeInjection(nn.Module):
        """
        BakeTimeInjection: Adaptive Time Injection Module
        Time Embedding을 사용해 피처맵을 정규화(Norm)하고, 스케일(Scale)과 시프트(Shift)를 조절합니다.
        """

        def __init__(self, dim):
            super().__init__()

            # 2. Time Embedding으로부터 Scale과 Shift를 예측하는 Act + Linear
            self.act1 = Heo.HeLU(dim)
            self.act2 = Heo.HeLU(dim * 2)
            self.linear1 = nn.Linear(dim, dim * 2)  # Output: (Scale, Shift)
            self.linear2 = nn.Linear(dim * 2, dim * 2)  # Output: (Scale, Shift)
            # 3. Zero Initialization
            # 학습 초기에는 입력(x)을 그대로 내보내도록 Scale/Shift를 0에 가깝게 초기화
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

        def forward(self, x, t_emb):
            """
            x: (B, C, H, W)
            t_emb: (B, C) - Time Embedding
            """
            # A. 시간 정보로부터 변조 계수 예측
            emb = self.linear1(self.act1(t_emb))
            emb = self.linear2(self.act2(emb))
            scale, shift = emb.chunk(2, dim=1)  # (B, C), (B, C)

            # Broadcasting을 위한 차원 확장 (B, C, 1, 1)
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]

            # B. Adaptive Modulation
            x = x * (torch.tanh(sqrt(3.0) * scale) + 1.0) + shift
            return x

    class BakeBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.time_embedding = Heo.HeoTimeEmbedding(dim)
            self.time_injector = Heo.BakeTimeInjection(dim)

            self.feature_gate = Heo.HeoGate2d(dim)
            self.residual_gate = Heo.HeoGate2d(dim)

            self.nemo1 = Heo.NeMO33(dim)
            self.nemo2 = Heo.NeMO33(dim)

        def forward(self, x, features, time):
            residual = x

            t_emb = self.time_embedding(time)
            h = self.time_injector(x, t_emb)
            h = self.nemo1(h)
            h = self.feature_gate(h, features)
            h = self.nemo2(h)
            h = self.residual_gate(h, residual)

            return h

    class BakeNet(nn.Module):
        def __init__(self, in_channels=3, dim=96, num_blocks=20):
            super().__init__()
            self.head = nn.Conv2d(in_channels, dim, 3, 1, 1)
            self.lr_encoder = nn.Conv2d(in_channels, dim, 3, 1, 1)
            
            self.body = nn.ModuleList([
                Heo.BakeBlock(dim) for _ in range(num_blocks)
            ])
            
            self.tail = nn.Conv2d(dim, in_channels, 3, 1, 1)
            
        def forward(self, x, x_lr, time):
            # x: Noisy Image (B, 3, H, W)
            # x_lr: Condition Image (B, 3, H, W) - Same resolution as x
            # time: Timestep (B,)
            
            # 1. Feature Extraction
            features = self.lr_encoder(x_lr) # (B, 96, H, W)
            h = self.head(x)                 # (B, 96, H, W)
            
            # 2. Body Processing
            for block in self.body:
                h = block(h, features, time)
                
            # 3. Output
            out = self.tail(h)
            return out
