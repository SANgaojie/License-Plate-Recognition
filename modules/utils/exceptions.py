"""
车牌识别系统异常处理模块
定义了系统中各种可能出现的异常类型
"""

class CarLicenseException(Exception):
    """车牌识别系统基本异常类"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ConfigError(CarLicenseException):
    """配置文件错误"""
    pass

class ImageLoadError(CarLicenseException):
    """图像加载错误"""
    pass

class PlateDetectionError(CarLicenseException):
    """车牌检测错误"""
    pass

class CharacterSegmentationError(CarLicenseException):
    """字符分割错误"""
    pass

class CharacterRecognitionError(CarLicenseException):
    """字符识别错误"""
    pass

class ModelLoadError(CarLicenseException):
    """模型加载错误"""
    pass

class ModelTrainError(CarLicenseException):
    """模型训练错误"""
    pass 