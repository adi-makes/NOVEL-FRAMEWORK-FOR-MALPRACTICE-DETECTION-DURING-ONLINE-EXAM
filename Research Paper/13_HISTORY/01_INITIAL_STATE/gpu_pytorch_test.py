import torch

def test_gpu_pytorch():
    cuda_available = torch.cuda.is_available()
    print('CUDA available:', cuda_available)
    if cuda_available:
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        z = x @ y
        torch.cuda.synchronize()
        print('GPU:', torch.cuda.get_device_name(0))
        assert z.device.type == 'cuda'