import os
import sys
from pathlib import Path

import torch

from Conv1dFFT import OneSidedComplexFFT, OneSidedComplexIFFT


def main():
    # 测试
    batch_size, n_frames, fft_len = 3, 6, 513
    out_fft_onnx_path = Path("data/output/fft.onnx")
    out_ifft_onnx_path = Path("data/output/ifft.onnx")

    # 输入信号
    input_signal = torch.randn(batch_size, n_frames, fft_len)

    # 模拟 FFT
    fft_model = OneSidedComplexFFT(fft_len)
    out1_comp = torch.fft.rfft(input_signal)  # (batch_size, n_frames, freq_bins)
    out1 = torch.view_as_real(out1_comp).flatten(start_dim=-2)  # (batch_size, n_frames, freq_bins*2)
    out2 = fft_model(input_signal)
    torch.testing.assert_close(out1, out2, rtol=1e-4, atol=1e-4)

    # 模拟 IFFT
    ifft_model = OneSidedComplexIFFT(fft_len)
    tmp1 = torch.fft.irfft(out1_comp, n=fft_len)
    tmp2 = ifft_model(out1)
    torch.testing.assert_close(tmp1, tmp2)

    # 导出 ONNX
    out_fft_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        fft_model,
        (input_signal,),
        out_fft_onnx_path.as_posix(),
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
    )
    os.system(f"{sys.executable} -m onnxsim {out_fft_onnx_path} {out_fft_onnx_path.with_suffix('.sim.onnx')}")

    torch.onnx.export(
        ifft_model,
        (out1,),
        "data/output/ifft.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
    )
    os.system(f"{sys.executable} -m onnxsim {out_ifft_onnx_path} {out_ifft_onnx_path.with_suffix('.sim.onnx')}")
    ...


if __name__ == "__main__":
    main()
    ...
