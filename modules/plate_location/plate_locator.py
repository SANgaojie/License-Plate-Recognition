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
        
        # 定义HSV颜色范围
        self.lower_blue = np.array([100, 40, 40])
        self.upper_blue = np.array([140, 255, 255])
        self.lower_yellow = np.array([15, 40, 40])
        self.upper_yellow = np.array([35, 255, 255])
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([100, 255, 255])
        
        # 定义车牌的宽高比范围
        self.min_ratio = 2.0  # 最小宽高比
        self.max_ratio = 5.5  # 最大宽高比
        
        # 调试模式（可通过配置文件开启）
        self.debug_mode = self.config.get("debug_mode", False)
    
    def preprocess_image(self, img):
        """图像预处理
        
        Args:
            img: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 调整图像大小，保持纵横比
        h, w = img.shape[:2]
        max_dimension = 1000
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 创建处理图像的副本
        processed = img.copy()
        
        # 转换为灰度图
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        blur_size = self.config.get("blur", 3)
        if blur_size % 2 == 0:  # 确保是奇数
            blur_size += 1
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # 应用自适应直方图均衡化提高对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 调试模式下保存预处理图像
        if self.debug_mode:
            cv2.imwrite("debug_preprocessed.jpg", gray)
        
        return processed, gray
    
    def detect_color(self, plate_img):
        """检测车牌颜色
        
        Args:
            plate_img: 车牌图像
            
        Returns:
            str: 车牌颜色 (blue, yellow, green)
        """
        # 转换为HSV
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        
        # 计算各种颜色的像素数量
        blue_count = cv2.countNonZero(cv2.inRange(hsv, self.lower_blue, self.upper_blue))
        yellow_count = cv2.countNonZero(cv2.inRange(hsv, self.lower_yellow, self.upper_yellow))
        green_count = cv2.countNonZero(cv2.inRange(hsv, self.lower_green, self.upper_green))
        
        # 确定颜色
        color_counts = {"blue": blue_count, "yello": yellow_count, "green": green_count}
        return max(color_counts, key=color_counts.get)
    
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
            
            # 图像预处理
            original_img, gray = self.preprocess_image(img)
            
            # 尝试几种不同的车牌定位方法
            for method_num in range(1, 5):
                try:
                    logger.info(f"尝试车牌定位方法 {method_num}")
                    
                    if method_num == 1:
                        # 方法1: 边缘检测 + Sobel算子 + 形态学操作
                        plate_img = self._locate_plate_method1(original_img, gray)
                    elif method_num == 2:
                        # 方法2: 垂直边缘检测 + 矩形筛选
                        plate_img = self._locate_plate_method2(original_img, gray)
                    elif method_num == 3:
                        # 方法3: 基于颜色的分割
                        plate_img = self._locate_plate_method3(original_img)
                    else:
                        # 方法4: 基于轮廓的方法
                        plate_img = self._locate_plate_method4(original_img, gray)
                    
                    # 如果成功找到车牌
                    if plate_img is not None and plate_img.size > 0:
                        # 确保车牌图像不会太小导致后续处理失败
                        if plate_img.shape[0] < 10 or plate_img.shape[1] < 30:
                            logger.warning(f"车牌太小，尺寸: {plate_img.shape}")
                            continue
                        
                        # 检测车牌颜色
                        color = self.detect_color(plate_img)
                        logger.info(f"成功定位车牌区域: 颜色={color}, 尺寸={plate_img.shape}")
                        
                        # 调试模式下保存找到的车牌图像
                        if self.debug_mode:
                            cv2.imwrite(f"debug_plate_method{method_num}.jpg", plate_img)
                        
                        return plate_img, color
                except Exception as e:
                    logger.debug(f"方法{method_num}失败: {str(e)}")
                    continue
            
            # 如果所有方法都失败，尝试返回近似车牌区域
            logger.warning("所有车牌定位方法都失败，尝试近似定位")
            
            # 最后的尝试：假设车牌在图像中心位置
            h, w = original_img.shape[:2]
            center_y = h // 2
            center_x = w // 2
            plate_h = h // 8  # 假设车牌高度为图像高度的1/8
            plate_w = int(plate_h * 3.5)  # 假设车牌宽高比为3.5
            
            # 截取图像中心区域
            y1 = max(0, center_y - plate_h // 2)
            y2 = min(h, center_y + plate_h // 2)
            x1 = max(0, center_x - plate_w // 2)
            x2 = min(w, center_x + plate_w // 2)
            
            # 确保区域有效
            if x1 < x2 and y1 < y2:
                plate_img = original_img[y1:y2, x1:x2]
                color = self.detect_color(plate_img)
                logger.warning(f"使用近似区域: 颜色={color}, 尺寸={plate_img.shape}")
                return plate_img, color
            
            raise PlateDetectionError("无法定位车牌区域")
            
        except Exception as e:
            if isinstance(e, PlateDetectionError):
                raise e
            else:
                logger.error(f"车牌定位失败: {str(e)}")
                raise PlateDetectionError(f"车牌定位失败: {str(e)}")
            
        return None, None
    
    def _locate_plate_method1(self, original_img, gray):
        """方法1: 边缘检测 + Sobel算子 + 形态学操作
        
        Args:
            original_img: 原始彩色图像
            gray: 预处理后的灰度图像
            
        Returns:
            车牌区域图像或None
        """
        # 计算Sobel梯度
        sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        
        # 组合梯度，注意：车牌边缘的水平梯度会更明显
        grad = cv2.addWeighted(abs_sobel_x, 0.8, abs_sobel_y, 0.2, 0)
        
        # 图像二值化
        ret, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作，突出车牌区域
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, rect_kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 调试模式下保存二值化图像
        if self.debug_mode:
            cv2.imwrite("debug_method1_binary.jpg", binary)
            debug_img = original_img.copy()
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite("debug_method1_contours.jpg", debug_img)
        
        # 筛选轮廓
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算宽高比
            ratio = float(w) / h if h > 0 else 0
            
            # 根据车牌特性筛选
            if self.min_ratio <= ratio <= self.max_ratio and w > 80 and h > 20 and w * h > self.min_area:
                # 提取车牌区域
                plate_img = original_img[y:y+h, x:x+w]
                return plate_img
        
        return None
    
    def _locate_plate_method2(self, original_img, gray):
        """方法2: 垂直边缘检测 + 矩形筛选
        
        Args:
            original_img: 原始彩色图像
            gray: 预处理后的灰度图像
            
        Returns:
            车牌区域图像或None
        """
        # Canny边缘检测
        edges = cv2.Canny(gray, 100, 200)
        
        # 使用闭操作连接边缘
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rect_kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 调试模式下保存边缘图像
        if self.debug_mode:
            cv2.imwrite("debug_method2_edges.jpg", edges)
            cv2.imwrite("debug_method2_closed.jpg", closed)
        
        # 筛选轮廓
        candidate_plates = []
        
        for contour in contours:
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            
            # 计算矩形的宽高比
            width, height = rect[1]
            if width < height:
                width, height = height, width
                
            ratio = width / height if height > 0 else 0
            
            # 根据车牌特性筛选
            if self.min_ratio <= ratio <= self.max_ratio and width > 80 and height > 20 and width * height > self.min_area:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # 计算倾斜角度
                angle = rect[2]
                if width < height:
                    angle = angle + 90
                    
                # 如果角度偏离水平或垂直太多，跳过
                if abs(angle) > 30 and abs(angle - 90) > 30:
                    continue
                
                # 获取矩形四个顶点的最小和最大坐标
                x_min = min(box[:, 0])
                x_max = max(box[:, 0])
                y_min = min(box[:, 1])
                y_max = max(box[:, 1])
                
                # 确保坐标不越界
                h, w = original_img.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w - 1, x_max)
                y_max = min(h - 1, y_max)
                
                # 如果矩形区域有效
                if x_min < x_max and y_min < y_max:
                    area = (x_max - x_min) * (y_max - y_min)
                    candidate_plates.append((x_min, y_min, x_max, y_max, area))
        
        # 根据面积排序，选择最大的几个候选区域
        candidate_plates.sort(key=lambda x: x[4], reverse=True)
        
        # 检查每个候选区域
        for x_min, y_min, x_max, y_max, _ in candidate_plates[:5]:  # 只检查前5个最大的
            plate_img = original_img[y_min:y_max, x_min:x_max]
            
            # 验证是否是车牌（可以添加更多的验证方法）
            if plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                return plate_img
        
        return None
    
    def _locate_plate_method3(self, original_img):
        """方法3: 基于颜色的分割
        
        Args:
            original_img: 原始彩色图像
            
        Returns:
            车牌区域图像或None
        """
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩码
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # 组合掩码
        combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, green_mask)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morphed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 调试模式下保存掩码图像
        if self.debug_mode:
            cv2.imwrite("debug_method3_mask.jpg", combined_mask)
            cv2.imwrite("debug_method3_morphed.jpg", morphed_mask)
        
        # 筛选轮廓
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算宽高比
            ratio = float(w) / h if h > 0 else 0
            
            # 根据车牌特性筛选
            if self.min_ratio <= ratio <= self.max_ratio and w > 80 and h > 20 and w * h > self.min_area:
                # 提取车牌区域
                plate_img = original_img[y:y+h, x:x+w]
                return plate_img
        
        return None
    
    def _locate_plate_method4(self, original_img, gray):
        """方法4: 基于轮廓的方法
        
        Args:
            original_img: 原始彩色图像
            gray: 预处理后的灰度图像
            
        Returns:
            车牌区域图像或None
        """
        # 计算图像梯度
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # 组合梯度，突出垂直边缘
        grad = cv2.addWeighted(abs_grad_x, 1.0, abs_grad_y, 0.0, 0)
        
        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 开运算，去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 垂直方向的闭运算，连接字符
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 水平方向的闭运算，连接字符
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 调试模式下保存二值化图像
        if self.debug_mode:
            cv2.imwrite("debug_method4_binary.jpg", binary)
        
        # 筛选候选区域
        candidate_plates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            # 获取最小外接矩形
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            # 确保宽度大于高度
            if width < height:
                width, height = height, width
                
            # 计算宽高比
            ratio = width / height if height > 0 else 0
            
            # 车牌宽高比筛选
            if self.min_ratio <= ratio <= self.max_ratio and width > 80 and height > 20:
                # 将矩形转换为四个顶点坐标
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # 计算轮廓的最小和最大坐标
                x_min = min(box[:, 0])
                x_max = max(box[:, 0])
                y_min = min(box[:, 1])
                y_max = max(box[:, 1])
                
                # 确保坐标不越界
                h, w = original_img.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w - 1, x_max)
                y_max = min(h - 1, y_max)
                
                # 如果矩形区域有效
                if x_min < x_max and y_min < y_max:
                    candidate_plates.append((x_min, y_min, x_max, y_max, area))
        
        # 根据面积排序
        candidate_plates.sort(key=lambda x: x[4], reverse=True)
        
        # 检查候选区域
        for i, (x_min, y_min, x_max, y_max, _) in enumerate(candidate_plates[:3]):  # 只检查前3个最大的
            plate_img = original_img[y_min:y_max, x_min:x_max]
            
            # 如果找到有效车牌区域
            if plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                # 调试模式下保存候选车牌图像
                if self.debug_mode:
                    cv2.imwrite(f"debug_method4_candidate{i}.jpg", plate_img)
                
                return plate_img
        
        return None 