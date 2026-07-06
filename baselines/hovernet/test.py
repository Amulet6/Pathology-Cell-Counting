import torch
import sys
import os

# 获取当前 test.py 所在的绝对目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将 official 文件夹动态加入系统路径
sys.path.append(os.path.join(current_dir, 'official'))

try:
    # 尝试从官方目录导入 HoVer-Net 模型
    from models.hovernet.net_desc import HoVerNet
    print("✅ 成功导入 HoVer-Net 模块！")

    # 实例化模型：nr_types=2 代表分类数为2（背景=0，细胞=1）
    # mode='original' 或 'fast'
    model = HoVerNet(nr_types=2, mode='original')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("✅ 模型实例化并成功加载至 GPU！")

    # original 模式需要 270x270 输入，fast 模式需要 256x256 输入
    input_size = 270 if model.mode == 'original' else 256
    dummy_input = torch.randn(2, 3, input_size, input_size, device=device)
    
    # 前向传播
    outputs = model(dummy_input)
    
    print("\n🎉 前向传播成功！模型输出的分支：")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f" - 分支 '{key}': 输出形状 {value.shape}")
        elif isinstance(value, list):
            print(f" - 分支 '{key}': 包含 {len(value)} 个特征层")

except Exception as e:
    print(f"❌ 测试失败，报错信息：\n{e}")