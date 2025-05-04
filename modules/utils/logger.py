import logging
import os
from datetime import datetime

class Logger:
    """日志系统类，提供应用程序的日志记录功能"""
    
    def __init__(self, log_level=logging.INFO):
        """初始化日志系统
        
        Args:
            log_level: 日志级别，默认为INFO
        """
        # 创建日志目录
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 设置日志文件名
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_filename = os.path.join(self.log_dir, f'car_license_{current_date}.log')
        
        # 配置日志格式
        self.logger = logging.getLogger('car_license')
        self.logger.setLevel(log_level)
        
        # 防止重复添加handler
        if not self.logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            file_handler.setLevel(log_level)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            
            # 日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def debug(self, message):
        """记录调试级别的日志"""
        self.logger.debug(message)
    
    def info(self, message):
        """记录信息级别的日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告级别的日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误级别的日志"""
        self.logger.error(message)
    
    def critical(self, message):
        """记录严重错误级别的日志"""
        self.logger.critical(message)

# 创建全局日志实例
logger = Logger() 