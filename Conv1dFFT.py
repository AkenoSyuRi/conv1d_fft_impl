import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class OneSidedComplexFFT(nn.Module):
    def __init__(self, fft_len):
        super(OneSidedComplexFFT, self).__init__()
        self.fft_len = fft_len
        self.fft_bins = fft_len // 2 + 1  # 保留 one-sided 的频率分量

        # 生成正弦和余弦核
        n = np.arange(self.fft_len)
        k = np.arange(self.fft_bins).reshape(-1, 1)  # 仅生成 one-sided 部分
        cos_kernel = np.cos(2 * np.pi * k * n / self.fft_len)  # (fft_bins, fft_len)
        sin_kernel = -np.sin(2 * np.pi * k * n / self.fft_len)

        # 创建卷积核（实部和虚部）
        real_kernel = torch.tensor(cos_kernel, dtype=torch.float32).unsqueeze(1)  # (fft_bins, 1, fft_len)
        imag_kernel = torch.tensor(sin_kernel, dtype=torch.float32).unsqueeze(1)

        # 定义 Conv1d 层，设置权重
        self.real_conv = nn.Conv1d(1, self.fft_bins, kernel_size=self.fft_len, bias=False)
        self.imag_conv = nn.Conv1d(1, self.fft_bins, kernel_size=self.fft_len, bias=False)

        # 将核设置为固定值
        self.real_conv.weight.data = real_kernel
        self.imag_conv.weight.data = imag_kernel
        self.real_conv.weight.requires_grad = False
        self.imag_conv.weight.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, n_frames, fft_len)
        return: (batch_size, n_frames, freq_bins*2)
        """
        assert (
            x.ndim == 3 and x.shape[2] == self.fft_len
        ), "Input must be a 3D tensor with shape (batch_size, n_frames, fft_len)."
        batch_size, n_frames, _ = x.shape

        x = x.reshape(batch_size * n_frames, 1, self.fft_len)

        real_part = self.real_conv(x)  # 计算实部 (batch_size * n_frames, freq_bins, 1)
        imag_part = self.imag_conv(x)  # 计算虚部 (batch_size * n_frames, freq_bins, 1)

        # 组合为复数结果 (batch_size * n_frames, freq_bins, 1)
        out_spec = torch.stack([real_part, imag_part], dim=-1)  # (batch_size * n_frames, freq_bins, 1, 2)
        out_spec = out_spec.reshape(batch_size, n_frames, -1)  # (batch_size, n_frames, freq_bins * 2)
        return out_spec


class OneSidedComplexIFFT(nn.Module):
    def __init__(self, fft_len):
        super(OneSidedComplexIFFT, self).__init__()
        self.fft_len = fft_len
        self.fft_bins = fft_len // 2 + 1  # one-sided 频率分量
        self.spec_slice = slice(1, -1) if fft_len % 2 == 0 else slice(1, None)  # 保留 one-sided 部分

        # 生成正弦和余弦核
        n = np.arange(fft_len).reshape(-1, 1)
        k = np.arange(fft_len)  # one-sided 的频率索引
        cos_kernel = np.cos(2 * np.pi * k * n / fft_len)
        sin_kernel = np.sin(2 * np.pi * k * n / fft_len)

        # 创建卷积核（实部和虚部）
        real_kernel = torch.tensor(cos_kernel, dtype=torch.float32).unsqueeze(1) / self.fft_len
        imag_kernel = torch.tensor(sin_kernel, dtype=torch.float32).unsqueeze(1) / self.fft_len

        # 定义 Conv1d 层，设置权重
        self.real_conv = nn.Conv1d(1, fft_len, kernel_size=fft_len, bias=False)
        self.imag_conv = nn.Conv1d(1, fft_len, kernel_size=fft_len, bias=False)

        # 设置卷积核
        self.real_conv.weight.data = real_kernel
        self.imag_conv.weight.data = imag_kernel
        self.real_conv.weight.requires_grad = False
        self.imag_conv.weight.requires_grad = False

    def spec_cat(self, x: Tensor, conj=False) -> Tensor:
        """
        x: (batch_size*n_frames, 1, freq_bins)
        return: (batch_size*n_frames, 1, fft_len)
        """
        half_spec = x[..., self.spec_slice].flip([-1])  # (batch_size*n_frames, 1, fft_len)
        if conj:
            half_spec = -half_spec
        x = torch.cat([x, half_spec], dim=-1)  # (batch_size*n_frames, 1, fft_len)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (batch_size, n_frames, freq_bins*2)
        :return: (batch_size, n_frames, fft_len)
        """
        assert (
            x.ndim == 3 and x.shape[2] == self.fft_bins * 2
        ), "Input must be a 3D tensor with shape (batch_size, n_frames, freq_bins * 2)."
        batch_size, n_frames, _ = x.shape

        # 拆分为实部和虚部
        x = torch.reshape(x, (batch_size * n_frames, 1, self.fft_bins, 2))
        real_part, imag_part = x.unbind(-1)  # (batch_size*n_frames, 1, fft_bins)

        real_part = self.spec_cat(real_part)  # (batch_size*n_frames, 1, fft_len)
        imag_part = self.spec_cat(imag_part, conj=True)  # (batch_size*n_frames, 1, fft_len)

        # 计算实部和虚部的卷积
        out_real = self.real_conv(real_part) - self.imag_conv(imag_part)  # (batch_size*n_frames, fft_len, 1)
        # out_imag = self.real_conv(imag_part) + self.imag_conv(real_part)

        out_real = out_real.reshape(batch_size, n_frames, self.fft_len)

        # IFFT 的输出必须是实数，因此虚部应接近零
        return out_real
