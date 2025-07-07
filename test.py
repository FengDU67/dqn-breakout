import torch

def test_gpu_memory_usage():
    """实际测试创建不同大小张量的显存使用"""
    
    print("=== 实际显存使用测试 ===")
    
    # 清空显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0)
        print(f"初始显存使用: {initial_memory / (1024**2):.1f} MB")
        
        state_shape = (4, 84, 84)
        device = torch.device("cuda")
        
        test_sizes = [3500, 10000, 25000, 50000]
        
        for size in test_sizes:
            try:
                # 创建状态张量
                states = torch.zeros((size, *state_shape), dtype=torch.float32, device=device)
                next_states = torch.zeros((size, *state_shape), dtype=torch.float32, device=device)
                actions = torch.zeros(size, dtype=torch.long, device=device)
                rewards = torch.zeros(size, dtype=torch.float32, device=device)
                dones = torch.zeros(size, dtype=torch.bool, device=device)
                
                current_memory = torch.cuda.memory_allocated(0)
                used_memory = (current_memory - initial_memory) / (1024**2)
                
                print(f"缓冲池大小 {size:5,}: 实际使用 {used_memory:.1f} MB")
                
                # 清理
                del states, next_states, actions, rewards, dones
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"缓冲池大小 {size:5,}: 显存不足 - {e}")
                break
    else:
        print("未检测到CUDA设备")

test_gpu_memory_usage()