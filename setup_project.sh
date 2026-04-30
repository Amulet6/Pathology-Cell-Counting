#!/bin/bash
# setup_project.sh - 病理图像细胞计数项目初始化脚本

echo "🚀 开始创建项目框架..."

# 创建目录结构
mkdir -p configs datasets models utils train eval logs/checkpoints logs/tensorboard_logs results/predictions results/visualizations docs

# 创建 __init__.py 文件
touch configs/__init__.py datasets/__init__.py models/__init__.py utils/__init__.py train/__init__.py eval/__init__.py

# 创建 requirements.txt
cat > requirements.txt << 'EOF'
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
opencv-python>=4.5.0
scikit-image>=0.18.0
matplotlib>=3.3.0
pillow>=8.0.0
tqdm>=4.60.0
tensorboard>=2.5.0
scipy>=1.7.0
pandas>=1.3.0
albumentations>=1.1.0
EOF

# 创建配置文件
cat > configs/config.py << 'EOF'
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')
LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs')

DATASETS = {
    'bcdata': {'path': os.path.join(DATA_ROOT, 'BCData')},
    'conic': {'path': os.path.join(DATA_ROOT, 'CoNIC')},
    'monuseg': {'path': os.path.join(DATA_ROOT, 'MoNuSeg')},
}

TRAIN_CONFIG = {
    'batch_size': 16,
    'epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'seed': 42,
}
EOF

echo "✅ 项目框架创建完成！"
echo "📦 接下来运行：pip install -r requirements.txt"

