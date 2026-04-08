"""
配置管理模块
"""

import json
import os
from typing import Dict, Any


class Config:
    """配置类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        # 默认配置
        self.default_config = {
            "model": {
                "d_model": 256,
                "nhead": 8,
                "num_encoder_layers": 6,
                "num_associate_decoder_layers": 3,
                "num_filtering_decoder_layers": 6,
                "dim_feedforward_encoder": 2048,
                "dim_feedforward_associate": 1024,
                "dim_feedforward_filtering": 2048,
                "dropout": 0.1,
                "max_targets": 20
            },
            "data": {
                "task_type": 1,
                "num_train_scenarios": 800,
                "num_val_scenarios": 100,
                "num_test_scenarios": 100,
                "tau": 4,
                "max_measurements": 30,
                "batch_size": 16,
                "num_workers": 0
            },
            "training": {
                "num_steps": 800000,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "grad_clip": 1.0,
                "association_weight": 1.0,
                "filtering_weight": 1.0,
                "gamma": 1.0,
                "lr_scheduler": {
                    "type": "step",
                    "step_size": 200000,
                    "gamma": 0.5
                }
            },
            "logging": {
                "log_interval": 100,
                "val_interval": 5000,
                "save_interval": 10000,
                "save_dir": "checkpoints",
                "log_dir": "logs"
            }
        }
        
        # 加载配置
        if config_path is not None and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self.default_config.copy()
    
    def load_config(self, config_path: str):
        """从JSON文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        # 使用加载的配置更新默认配置
        self.config = self._deep_update(self.default_config.copy(), loaded_config)
        print(f"已从 {config_path} 加载配置")
    
    def save_config(self, save_path: str):
        """保存配置到JSON文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        print(f"配置已保存到 {save_path}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """深度更新字典"""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, *keys, default=None):
        """
        获取配置值
        
        Args:
            *keys: 配置键路径，例如 config.get('model', 'd_model')
            default: 默认值
        
        Returns:
            配置值
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys, value):
        """
        设置配置值
        
        Args:
            *keys: 配置键路径
            value: 要设置的值
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典"""
        return self.config.copy()
    
    def __repr__(self):
        return f"Config({json.dumps(self.config, indent=2, ensure_ascii=False)})"


def load_config(config_path: str = None) -> Config:
    """
    加载配置的便捷函数
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Config对象
    """
    return Config(config_path)
