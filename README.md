# conv1d_fft_impl

在DTLN模型导出onnx时，ifft与复数操作不受到支持，  
这里使用torch.nn.Conv1d函数实现了fft操作, 支持多帧多batch输入，  
与torch.fft.fft以及torch.fft.ifft函数的结果精度误差在1e-4以下。
