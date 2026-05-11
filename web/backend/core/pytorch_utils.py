"""
PyTorch环境初始化工具
统一管理PyTorch 2.x兼容性配置和monkey patch
"""

import os
import sys
import types


def setup_pytorch_compatibility():
    """
    设置PyTorch 2.x兼容性环境
    禁用weights_only安全检查，配置环境变量
    """
    # 禁用PyTorch 2.x的weights_only安全检查和新特性
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOADING"] = "False"
    os.environ["PYTORCH_JIT"] = "0"

    # 创建模拟的torch.sparse.semi_structured模块
    semi_structured_module = types.ModuleType("torch.sparse.semi_structured")
    semi_structured_module.SparseSemiStructuredTensor = None
    semi_structured_module.SparseSemiStructuredTensorBCSR = None
    semi_structured_module.SparseSemiStructuredTensorBCOO = None
    semi_structured_module.SparseSemiStructuredTensorCUSPARSELT = None
    semi_structured_module.SparseSemiStructuredTensorCUTLASS = None
    semi_structured_module.semi_structured_to_dense = lambda x: x
    semi_structured_module.dense_to_semi_structured = lambda x: x
    semi_structured_module.to_sparse_semi_structured = lambda x: x
    sys.modules["torch.sparse.semi_structured"] = semi_structured_module


def apply_torch_load_monkey_patch():
    """
    对torch.load进行monkey patch，默认使用weights_only=False
    """
    try:
        import torch
        _original_torch_load = torch.load

        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load
    except ImportError:
        print("警告: torch未安装，无法应用monkey patch")


def init_pytorch_env():
    """
    完整的PyTorch环境初始化
    按顺序调用所有初始化步骤
    """
    setup_pytorch_compatibility()
    apply_torch_load_monkey_patch()
