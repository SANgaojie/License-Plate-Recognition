"""
车牌识别主控模块
整合图像处理、车牌定位、字符分割和字符识别等功能
"""

import cv2
import numpy as np
from modules.utils.exceptions import *
from modules.utils.logger import logger
from modules.image_processing.image_utils import imreadex
from modules.plate_location.plate_locator import PlateLocator
from modules.character_segmentation.char_segmenter import CharSegmenter
from modules.character_recognition.char_recognizer import CharRecognizer

class LicensePlateRecognizer:
    """车牌识别类，整合所有功能"""
    
    def __init__(self):
        """初始化车牌识别器"""
        logger.info("初始化车牌识别系统")
        self.plate_locator = PlateLocator()
        self.char_segmenter = CharSegmenter()
        self.char_recognizer = CharRecognizer()
        self.debug_mode = False
        self.debug_callback = None
    
    def set_debug_mode(self, debug_mode, callback=None):
        """设置调试模式
        
        Args:
            debug_mode: 是否开启调试模式
            callback: 调试信息回调函数
        """
        self.debug_mode = debug_mode
        self.debug_callback = callback
    
    def debug_show(self, img, title):
        """显示调试图像
        
        Args:
            img: 要显示的图像
            title: 图像标题
        """
        if self.debug_mode and self.debug_callback:
            self.debug_callback(img, title)
    
    def train_models(self):
        """训练识别模型"""
        try:
            logger.info("开始训练识别模型")
            self.char_recognizer.train_svm()
            logger.info("训练完成")
        except Exception as e:
            logger.error(f"训练模型失败: {str(e)}")
            raise ModelTrainError(f"训练模型失败: {str(e)}")
    
    def recognize_from_image(self, img_path):
        """从图像文件中识别车牌
        
        Args:
            img_path: 图像文件路径
            
        Returns:
            tuple: (识别结果, 车牌区域图像, 车牌颜色)
            
        Raises:
            CarLicenseException: 识别过程中出现错误时抛出
        """
        try:
            # 读取图像
            logger.info(f"从图像识别车牌: {img_path}")
            img = imreadex(img_path)
            if img is None:
                raise ImageLoadError(f"无法读取图像: {img_path}")
            
            return self.recognize(img)
            
        except Exception as e:
            if isinstance(e, CarLicenseException):
                raise e
            else:
                logger.error(f"从图像识别车牌失败: {str(e)}")
                raise CarLicenseException(f"从图像识别车牌失败: {str(e)}")
        
        return None, None, None
    
    def recognize(self, img):
        """识别图像中的车牌
        
        Args:
            img: 输入图像
            
        Returns:
            tuple: (识别结果, 车牌区域图像, 车牌颜色)
            
        Raises:
            CarLicenseException: 识别过程中出现错误时抛出
        """
        try:
            if img is None:
                raise ImageLoadError("输入图像为空")
            
            # 步骤1: 车牌定位
            logger.info("步骤1: 开始车牌定位")
            plate_img, plate_color = self.plate_locator.locate_plate(img)
            if plate_img is None:
                raise PlateDetectionError("未能定位到车牌")
            
            self.debug_show(plate_img, "车牌区域")
            logger.info(f"车牌颜色: {plate_color}")
            
            # 步骤2: 字符分割 - 传递车牌颜色信息
            logger.info("步骤2: 开始字符分割")
            char_imgs = self.char_segmenter.segment_chars(plate_img, plate_color)
            if not char_imgs or len(char_imgs) == 0:
                raise CharacterSegmentationError("未能分割字符")
            
            # 显示分割后的字符
            for i, char_img in enumerate(char_imgs):
                self.debug_show(char_img, f"字符{i+1}")
            
            logger.info(f"成功分割出 {len(char_imgs)} 个字符")
            
            # 步骤3: 字符识别
            logger.info("步骤3: 开始字符识别")
            plate_str = self.char_recognizer.recognize(char_imgs)
            if not plate_str:
                raise CharacterRecognitionError("未能识别字符")
            
            logger.info(f"识别结果: {plate_str}")
            return plate_str, plate_img, plate_color
            
        except Exception as e:
            if isinstance(e, CarLicenseException):
                logger.error(f"车牌识别失败: {e.message}")
                raise e
            else:
                logger.error(f"车牌识别失败: {str(e)}")
                raise CarLicenseException(f"车牌识别失败: {str(e)}")
        
        return None, None, None 