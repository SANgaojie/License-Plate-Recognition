"""
字符分割模块
负责将车牌图像分割为单个字符
"""

import cv2
import numpy as np
from modules.utils.exceptions import CharacterSegmentationError
from modules.utils.logger import logger

class CharSegmenter:
    """字符分割类，负责将车牌图像分割为单个字符"""
    
    def __init__(self):
        """初始化字符分割器"""
        pass
    
    def find_waves(self, threshold, histogram):
        """根据设定的阈值和图片直方图，找出波峰，用于分隔字符
        
        Args:
            threshold: 阈值
            histogram: 直方图
            
        Returns:
            list: 波峰的起始和结束位置列表
        """
        up_point = -1  # 上升点
        is_peak = False
        if histogram[0] > threshold:
            up_point = 0
            is_peak = True
        wave_peaks = []
        for i, x in enumerate(histogram):
            if is_peak and x < threshold:
                if i - up_point > 2:
                    is_peak = False
                    wave_peaks.append((up_point, i))
            elif not is_peak and x >= threshold:
                is_peak = True
                up_point = i
        if is_peak and up_point != -1 and i - up_point > 4:
            wave_peaks.append((up_point, i))
        return wave_peaks
    
    def segment_chars(self, plate_img):
        """将车牌图像分割为单个字符
        
        Args:
            plate_img: 车牌图像
            
        Returns:
            list: 分割后的字符图像列表
            
        Raises:
            CharacterSegmentationError: 字符分割失败时抛出
        """
        try:
            if plate_img is None:
                raise CharacterSegmentationError("输入车牌图像为空")
                
            logger.info("开始分割字符")
            
            # 灰度化
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            
            # 图像预处理
            ret, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 计算二值图像每列的像素和，构建直方图
            column_histogram = np.sum(binary_plate, axis=0)
            
            # 计算阈值，用于分割字符
            threshold = column_histogram.mean() * 0.5
            
            # 找出波峰
            wave_peaks = self.find_waves(threshold, column_histogram)
            
            # 过滤掉太近的波峰
            filtered_peaks = []
            for i, peak in enumerate(wave_peaks):
                # 宽度不能太小，这里设定最小宽度为10个像素
                if peak[1] - peak[0] >= 10:
                    filtered_peaks.append(peak)
            
            # 如果找不到足够的波峰，可能是阈值设置不当，尝试调整阈值
            if len(filtered_peaks) <= 6:
                threshold = column_histogram.mean() * 0.3
                wave_peaks = self.find_waves(threshold, column_histogram)
                
                filtered_peaks = []
                for i, peak in enumerate(wave_peaks):
                    if peak[1] - peak[0] >= 10:
                        filtered_peaks.append(peak)
            
            # 分割字符
            char_imgs = []
            for wave in filtered_peaks:
                char_img = binary_plate[:, wave[0]:wave[1]]
                # 去除上下空白
                char_rows = np.sum(char_img, axis=1)
                top = 0
                for i, row_sum in enumerate(char_rows):
                    if row_sum > 0:
                        top = i
                        break
                
                bottom = len(char_rows) - 1
                for i in range(len(char_rows) - 1, -1, -1):
                    if char_rows[i] > 0:
                        bottom = i
                        break
                
                # 裁剪字符图像
                if bottom > top:
                    char_img = char_img[top:bottom+1, :]
                    
                    # 调整大小为标准尺寸 (20x20)
                    char_img = cv2.resize(char_img, (20, 20), interpolation=cv2.INTER_AREA)
                    char_imgs.append(char_img)
            
            # 如果分割出的字符数量不合理（一般中国车牌有7个字符），记录警告
            if len(char_imgs) < 6 or len(char_imgs) > 8:
                logger.warning(f"分割出的字符数量不合理: {len(char_imgs)}")
            
            return char_imgs
            
        except Exception as e:
            if isinstance(e, CharacterSegmentationError):
                raise e
            else:
                logger.error(f"字符分割失败: {str(e)}")
                raise CharacterSegmentationError(f"字符分割失败: {str(e)}")
        
        return [] 