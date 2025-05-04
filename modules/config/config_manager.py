"""
配置管理模块
负责加载和管理系统配置
"""

import os
import json
from modules.utils.exceptions import ConfigError
from modules.utils.logger import logger

class ConfigManager:
    """配置管理类，负责加载和管理系统配置"""
    
    def __init__(self, config_file=None):
        """初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为None，将使用默认配置文件
        """
        if config_file is None:
            # 使用默认配置文件
            self.config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
        else:
            self.config_file = config_file
            
        self.config = None
        self.current_config = None
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            logger.info(f"正在加载配置文件: {self.config_file}")
            if not os.path.exists(self.config_file):
                # 如果配置文件不存在，创建默认配置
                self._create_default_config()
                
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                
            # 选择一个活动的配置
            for c in self.config["config"]:
                if c.get("open", False):
                    self.current_config = c.copy()
                    logger.info("成功加载配置")
                    break
            else:
                raise ConfigError("没有设置有效的配置参数")
                
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            raise ConfigError(f"加载配置失败: {str(e)}")
    
    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            "config": [
                {
                    "open": 1,
                    "blur": 3,
                    "morphologyr": 4,
                    "morphologyc": 19,
                    "col_num_limit": 10,
                    "row_num_limit": 21
                },
                {
                    "open": 0,
                    "blur": 3,
                    "morphologyr": 5,
                    "morphologyc": 12,
                    "col_num_limit": 10,
                    "row_num_limit": 18
                }
            ]
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
            logger.info("已创建默认配置文件")
        except Exception as e:
            logger.error(f"创建默认配置文件失败: {str(e)}")
            raise ConfigError(f"创建默认配置文件失败: {str(e)}")
    
    def get_config(self):
        """获取当前活动的配置
        
        Returns:
            dict: 当前活动的配置
        """
        return self.current_config
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info("配置已保存")
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            raise ConfigError(f"保存配置失败: {str(e)}")

# 创建全局配置管理器实例
config_manager = ConfigManager() 