"""
车牌定位模块
负责从图像中定位车牌区域
"""

import cv2
import numpy as np
from modules.utils.exceptions import PlateDetectionError
from modules.utils.logger import logger
from modules.config.config_manager import config_manager
from modules.image_processing.image_utils import point_limit

class PlateLocator:
    """车牌定位类，负责从图像中定位车牌区域"""
    
    def __init__(self):
        """初始化车牌定位器"""
        self.config = config_manager.get_config()
        self.min_area = 1000  # 车牌区域允许最小面积
    
    def accurate_place(self, card_img_hsv, limit1, limit2, color):
        """定位车牌精确位置
        
        Args:
            card_img_hsv: HSV格式图像
            limit1: 颜色下限
            limit2: 颜色上限
            color: 车牌颜色
            
        Returns:
            tuple: 精确定位的车牌区域坐标 (x_min, x_max, y_min, y_max)
        """
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        
        # 获取配置参数
        row_num_limit = self.config["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        
        # 根据颜色阈值对图像进行二值化
        try:
            binary_img = cv2.inRange(card_img_hsv, limit1, limit2)
            
            # 查找车牌区域的精确范围
            for i in range(row_num):
                count = 0
                for j in range(col_num):
                    if binary_img[i, j] == 255:
                        count += 1
                if count > col_num_limit:
                    if yl > i:
                        yl = i
                    if yh < i:
                        yh = i
            
            for j in range(col_num):
                count = 0
                for i in range(row_num):
                    if binary_img[i, j] == 255:
                        count += 1
                if count > row_num_limit:
                    if xl > j:
                        xl = j
                    if xr < j:
                        xr = j
            
            # 确保坐标合法
            return xl, xr, yl, yh
            
        except Exception as e:
            logger.error(f"车牌精确定位失败: {str(e)}")
            raise PlateDetectionError(f"车牌精确定位失败: {str(e)}")
    
    def locate_plate(self, img):
        """
        定位车牌位置
        
        Args:
            img: 输入图像
            
        Returns:
            tuple: (车牌区域图像, 车牌颜色)
            
        Raises:
            PlateDetectionError: 车牌定位失败时抛出
        """
        try:
            if img is None:
                raise PlateDetectionError("输入图像为空")
                
            logger.info("开始定位车牌")
            
            # 获取配置参数
            blur_size = self.config["blur"]
            morph_size_width = self.config["morphologyc"]
            morph_size_height = self.config["morphologyr"]
            
            # 高斯模糊，减少图像噪声
            blur = self.config["blur"]
            if blur > 0:
                img = cv2.GaussianBlur(img, (blur, blur), 0)
                
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 不同颜色车牌的HSV阈值
            lower_blue = np.array([100, 43, 46])
            upper_blue = np.array([124, 255, 255])
            lower_yellow = np.array([15, 43, 46])
            upper_yellow = np.array([34, 255, 255])
            lower_green = np.array([35, 43, 46])
            upper_green = np.array([99, 255, 255])
            
            # 在HSV空间中提取出蓝色、黄色和绿色区域
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # 定义结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size_width, morph_size_height))
            
            # 形态学操作，去除噪声
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
            
            # 寻找轮廓
            contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 合并所有颜色的轮廓和对应颜色标签
            color_contours = []
            for contour in contours_blue:
                color_contours.append((contour, "blue"))
            for contour in contours_yellow:
                color_contours.append((contour, "yello"))  # 注意：这里沿用了原代码的拼写错误
            for contour in contours_green:
                color_contours.append((contour, "green"))
            
            # 根据面积排序
            color_contours.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
            
            # 遍历所有轮廓，寻找可能的车牌区域
            for contour, color in color_contours:
                rect = cv2.minAreaRect(contour)  # 最小外接矩形
                box = cv2.boxPoints(rect)  # 获取矩形的四个角点
                box = np.int0(box)  # 转换为整数
                
                # 计算矩形的高和宽
                height = abs(box[0][1] - box[2][1])
                width = abs(box[0][0] - box[2][0])
                
                # 根据车牌的比例和面积过滤
                if (height > width * 1.2 or width > height * 3) and width > 60 and height > 15 and width * height > self.min_area:
                    # 截取车牌区域
                    # 获取旋转角度
                    angle = rect[2]
                    if angle > 45:
                        angle = angle - 90
                    
                    # 旋转图像
                    (h, w) = img.shape[:2]
                    center = rect[0]
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    
                    # 计算旋转后的新矩形
                    new_rect = (rect[0], (width, height), angle)
                    box = cv2.boxPoints(new_rect)
                    box = np.int0(box)
                    
                    # 截取车牌区域
                    x_min = min(box[:, 0])
                    x_max = max(box[:, 0])
                    y_min = min(box[:, 1])
                    y_max = max(box[:, 1])
                    
                    # 确保坐标不越界
                    x_min = max(0, x_min - 5)
                    x_max = min(w - 1, x_max + 5)
                    y_min = max(0, y_min - 5)
                    y_max = min(h - 1, y_max + 5)
                    
                    # 裁剪车牌区域
                    plate_img = rotated[y_min:y_max, x_min:x_max]
                    
                    # 对车牌区域进行精确定位
                    plate_hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
                    
                    if color == "blue":
                        xl, xr, yl, yh = self.accurate_place(plate_hsv, lower_blue, upper_blue, color)
                    elif color == "yello":
                        xl, xr, yl, yh = self.accurate_place(plate_hsv, lower_yellow, upper_yellow, color)
                    elif color == "green":
                        xl, xr, yl, yh = self.accurate_place(plate_hsv, lower_green, upper_green, color)
                    
                    # 再次裁剪，得到精确的车牌区域
                    if xl < xr and yl < yh:
                        plate_img_accurate = plate_img[yl:yh, xl:xr]
                        return plate_img_accurate, color
            
            # 如果没有找到合适的车牌区域，抛出异常
            raise PlateDetectionError("未找到合适的车牌区域")
            
        except Exception as e:
            if isinstance(e, PlateDetectionError):
                raise e
            else:
                logger.error(f"车牌定位失败: {str(e)}")
                raise PlateDetectionError(f"车牌定位失败: {str(e)}")
            
        return None, None 