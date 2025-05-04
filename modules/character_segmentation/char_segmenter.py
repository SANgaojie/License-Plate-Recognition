"""
字符分割模块
负责将车牌图像分割为单个字符
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from modules.utils.exceptions import CharacterSegmentationError
from modules.utils.logger import logger
from modules.config.config_manager import config_manager

class CharSegmenter:
    """字符分割类，负责将车牌图像分割为单个字符"""
    
    def __init__(self):
        """初始化字符分割器"""
        self.config = config_manager.get_config()
        self.debug_mode = self.config.get("debug_mode", False)
        
        # 设置标准字符数量
        self.std_plate_chars = 7  # 普通车牌7个字符
        self.green_plate_chars = 8  # 新能源车牌8个字符
        
        # 字符大小标准化参数
        self.char_w = 20  # 标准字符宽度
        self.char_h = 40  # 标准字符高度
        
        # 创建调试目录
        if self.debug_mode and not os.path.exists("debug_chars"):
            os.makedirs("debug_chars")
    
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
    
    def _preprocess_plate(self, plate_img):
        """预处理车牌图像
        
        Args:
            plate_img: 原始车牌图像
            
        Returns:
            tuple: (灰度图, 二值图列表)
        """
        # 保存原始图像用于调试
        if self.debug_mode:
            cv2.imwrite("debug_chars/01_plate_original.jpg", plate_img)
        
        # 调整车牌图像大小
        h, w = plate_img.shape[:2]
        aspect_ratio = float(w) / h
        
        # 标准车牌宽高比约为3.2:1
        if 2.7 <= aspect_ratio <= 4.5:
            # 保持原始宽高比调整大小
            new_h = 140  # 固定高度
            new_w = int(new_h * aspect_ratio)
            plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            logger.warning(f"车牌宽高比异常: {aspect_ratio:.2f}")
            # 强制调整为标准比例
            plate_img = cv2.resize(plate_img, (440, 140), interpolation=cv2.INTER_CUBIC)
        
        # 灰度化
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度（使用CLAHE）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 4))
        enhanced_gray = clahe.apply(gray_plate)
        
        if self.debug_mode:
            cv2.imwrite("debug_chars/02_gray_enhanced.jpg", enhanced_gray)
        
        # 创建多种二值化结果，以便后续选择最佳结果
        binary_results = []
        
        # 方法1：全局阈值二值化 (OTSU)
        _, binary_otsu = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_results.append(("OTSU", binary_otsu))
        
        # 方法2：自适应阈值二值化 (高斯)
        binary_adaptive1 = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        binary_results.append(("Adaptive1", binary_adaptive1))
        
        # 方法3：自适应阈值二值化 (均值)
        binary_adaptive2 = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        binary_results.append(("Adaptive2", binary_adaptive2))
        
        # 尝试不同的形态学操作改善二值图
        processed_binaries = []
        
        for name, binary in binary_results:
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            
            # 开运算：先腐蚀后膨胀，去除小噪点
            binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            processed_binaries.append((f"{name}_Open", binary_open))
            
            # 闭运算：先膨胀后腐蚀，填充小孔
            binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            processed_binaries.append((f"{name}_Close", binary_close))
            
            # 尝试更大的核进行闭运算，用于连接断开的字符
            kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary_close_large = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)
            processed_binaries.append((f"{name}_CloseLarge", binary_close_large))
        
        # 保存所有二值化结果用于调试
        if self.debug_mode:
            for i, (name, binary) in enumerate(processed_binaries):
                cv2.imwrite(f"debug_chars/03_binary_{name}.jpg", binary)
        
        return enhanced_gray, processed_binaries
    
    def _get_fixed_segments(self, binary_img, expected_chars=7):
        """基于车牌固定比例划分字符
        
        Args:
            binary_img: 二值化图像
            expected_chars: 期望的字符数量
            
        Returns:
            list: 字符区域列表 [(start, end), ...]
        """
        w = binary_img.shape[1]
        
        # 中国车牌字符间距基本均匀，可以按照固定比例分割
        if expected_chars == 7:  # 普通车牌
            # 省份字符约占10-15%宽度，字距基本相等
            segments = []
            
            # 省份字符（略宽）
            segments.append((int(w * 0.02), int(w * 0.15)))
            
            # 后6个字符
            char_width = (w * 0.98 - w * 0.15) / 6
            for i in range(6):
                start = int(w * 0.15 + char_width * i)
                end = int(w * 0.15 + char_width * (i + 1))
                segments.append((start, end))
        
        elif expected_chars == 8:  # 新能源车牌
            segments = []
            
            # 省份字符
            segments.append((int(w * 0.01), int(w * 0.12)))
            
            # 后7个字符
            char_width = (w * 0.99 - w * 0.12) / 7
            for i in range(7):
                start = int(w * 0.12 + char_width * i)
                end = int(w * 0.12 + char_width * (i + 1))
                segments.append((start, end))
        
        return segments
    
    def _segment_with_fixed_ratio(self, binary_plate, expected_chars=7):
        """使用固定比例分割字符
        
        Args:
            binary_plate: 二值化车牌图像
            expected_chars: 期望字符数
            
        Returns:
            list: 分割结果
        """
        segments = self._get_fixed_segments(binary_plate, expected_chars)
        
        # 保存分割线到调试图像
        if self.debug_mode:
            debug_img = cv2.cvtColor(binary_plate.copy(), cv2.COLOR_GRAY2BGR)
            h = binary_plate.shape[0]
            
            for start, end in segments:
                cv2.line(debug_img, (start, 0), (start, h), (0, 255, 0), 1)
                cv2.line(debug_img, (end, 0), (end, h), (0, 0, 255), 1)
            
            cv2.imwrite("debug_chars/05_fixed_segments.jpg", debug_img)
        
        return segments
    
    def _segment_with_projection(self, binary_plate, min_width=8, expected_chars=7):
        """使用投影法分割字符
        
        Args:
            binary_plate: 二值化后的车牌图像
            min_width: 最小字符宽度
            expected_chars: 期望字符数
            
        Returns:
            list: 分割结果 [(start, end), ...]
        """
        # 计算垂直投影
        projection = np.sum(binary_plate, axis=0)
        
        # 尝试多个阈值
        thresholds = [projection.mean() * 0.5, projection.mean() * 0.3, projection.mean() * 0.2]
        
        best_segments = []
        best_score = float('inf')  # 最小化与期望字符数的差异
        
        for threshold in thresholds:
            # 找出波峰
            peaks = self.find_waves(threshold, projection)
            
            # 过滤掉太窄的波峰
            filtered_peaks = [peak for peak in peaks if peak[1] - peak[0] >= min_width]
            
            # 计算与期望字符数的差异
            score = abs(len(filtered_peaks) - expected_chars)
            
            # 如果产生了更好的分割结果
            if score < best_score:
                best_score = score
                best_segments = filtered_peaks
        
        # 如果分割结果不满足期望，尝试优化
        if len(best_segments) != expected_chars:
            best_segments = self._optimize_segments(best_segments, binary_plate.shape[1], expected_chars)
        
        # 保存投影图用于调试
        if self.debug_mode:
            plt.figure(figsize=(10, 4))
            plt.plot(projection)
            
            # 绘制波峰区域
            for peak in best_segments:
                plt.axvspan(peak[0], peak[1], color='green', alpha=0.3)
            
            plt.savefig("debug_chars/04_projection.png")
            plt.close()
            
            # 在原图上标记分割线
            debug_img = cv2.cvtColor(binary_plate.copy(), cv2.COLOR_GRAY2BGR)
            h = binary_plate.shape[0]
            
            for start, end in best_segments:
                cv2.line(debug_img, (start, 0), (start, h), (0, 255, 0), 1)
                cv2.line(debug_img, (end, 0), (end, h), (0, 0, 255), 1)
            
            cv2.imwrite("debug_chars/04_projection_segments.jpg", debug_img)
        
        return best_segments
    
    def _segment_with_contours(self, binary_plate, expected_chars=7):
        """使用轮廓法分割字符
        
        Args:
            binary_plate: 二值化车牌图像
            expected_chars: 期望字符数
            
        Returns:
            list: 分割结果 [(start, end), ...]
        """
        # 寻找轮廓
        contours, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓
        valid_contours = []
        min_area = 100  # 最小轮廓面积
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # 排除太短或太扁的轮廓
                if h > binary_plate.shape[0] * 0.3 and w < binary_plate.shape[1] * 0.5:
                    valid_contours.append((x, y, w, h))
        
        # 按照x坐标排序
        valid_contours.sort(key=lambda c: c[0])
        
        # 轮廓合并：处理断开的字符
        merged_contours = []
        i = 0
        while i < len(valid_contours):
            x, y, w, h = valid_contours[i]
            
            # 检查是否需要与下一个轮廓合并
            if i + 1 < len(valid_contours):
                next_x = valid_contours[i+1][0]
                # 如果两个轮廓距离很近，可能是同一个字符
                if next_x - (x + w) < 10:
                    # 合并轮廓
                    next_w = valid_contours[i+1][2]
                    new_w = (next_x + next_w) - x
                    merged_contours.append((x, y, new_w, h))
                    i += 2
                    continue
            
            merged_contours.append((x, y, w, h))
            i += 1
        
        # 转换为字符分割格式
        segments = [(c[0], c[0] + c[2]) for c in merged_contours]
        
        # 如果轮廓数量与期望不符，进行优化
        if len(segments) != expected_chars:
            segments = self._optimize_segments(segments, binary_plate.shape[1], expected_chars)
        
        # 保存轮廓分割结果用于调试
        if self.debug_mode:
            debug_img = cv2.cvtColor(binary_plate.copy(), cv2.COLOR_GRAY2BGR)
            
            # 绘制原始轮廓
            for contour in contours:
                cv2.drawContours(debug_img, [contour], 0, (0, 255, 0), 1)
            
            # 绘制分割线
            h = binary_plate.shape[0]
            for start, end in segments:
                cv2.line(debug_img, (start, 0), (start, h), (255, 0, 0), 2)
                cv2.line(debug_img, (end, 0), (end, h), (0, 0, 255), 2)
            
            cv2.imwrite("debug_chars/06_contour_segments.jpg", debug_img)
        
        return segments
    
    def _compute_segment_quality(self, segments, binary_plate):
        """计算分割质量
        
        Args:
            segments: 分割结果
            binary_plate: 二值化图像
            
        Returns:
            float: 分割质量分数
        """
        h, w = binary_plate.shape
        
        # 统计所有分割块的字符像素密度
        densities = []
        for start, end in segments:
            if start < 0 or end > w:
                continue
                
            # 截取字符区域
            char_region = binary_plate[:, start:end]
            
            # 计算字符像素密度
            pixel_count = np.count_nonzero(char_region)
            region_area = char_region.shape[0] * char_region.shape[1]
            
            if region_area > 0:
                density = pixel_count / region_area
                densities.append(density)
        
        # 没有有效分割
        if not densities:
            return 0
        
        # 计算分数：平均密度和密度标准差的组合（标准差越小说明字符越均匀）
        avg_density = np.mean(densities)
        density_std = np.std(densities)
        
        # 字符密度均衡性得分
        balance_score = 1 / (1 + density_std)
        
        # 覆盖率（字符区域占总宽度的比例）
        covered_width = sum(end - start for start, end in segments)
        coverage_score = covered_width / w
        
        # 综合分数
        return avg_density * 0.6 + balance_score * 0.2 + coverage_score * 0.2
    
    def _optimize_segments(self, segments, plate_width, expected_chars=7):
        """优化分割结果以匹配期望字符数
        
        Args:
            segments: 原始分割 [(start, end), ...]
            plate_width: 车牌宽度
            expected_chars: 期望字符数
            
        Returns:
            list: 优化后的分割 [(start, end), ...]
        """
        # 没有分割或分割数为0时，使用固定分割方法
        if not segments:
            return self._get_fixed_segments(np.zeros((140, plate_width)), expected_chars)
        
        # 如果分割数正好符合预期，直接返回
        if len(segments) == expected_chars:
            return segments
        
        # 分割数过少
        if len(segments) < expected_chars:
            # 计算分割间的间隙
            gaps = []
            segments_sorted = sorted(segments)
            
            # 增加起始和结束间隙
            if segments_sorted[0][0] > 10:
                gaps.append((0, segments_sorted[0][0], 0))
            
            for i in range(len(segments_sorted)-1):
                gap_start = segments_sorted[i][1]
                gap_end = segments_sorted[i+1][0]
                gap_width = gap_end - gap_start
                
                # 只有大于一定宽度的间隙才考虑分割
                if gap_width > 10:
                    gaps.append((gap_start, gap_end, i+1))
            
            # 增加结束间隙
            if plate_width - segments_sorted[-1][1] > 10:
                gaps.append((segments_sorted[-1][1], plate_width, len(segments_sorted)))
            
            # 根据宽度排序
            gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
            
            # 添加新的分割，直到达到期望数量
            new_segments = segments[:]
            
            for gap_start, gap_end, insert_pos in gaps:
                if len(new_segments) >= expected_chars:
                    break
                    
                gap_width = gap_end - gap_start
                
                # 插入新的分割，宽度与平均字符宽度相近
                avg_width = plate_width // expected_chars
                
                # 如果间隙够大，可以分成多个字符
                num_chars = min(expected_chars - len(new_segments), max(1, gap_width // avg_width))
                
                for i in range(num_chars):
                    char_start = gap_start + i * (gap_width // num_chars)
                    char_end = gap_start + (i + 1) * (gap_width // num_chars)
                    
                    new_segments.append((char_start, char_end))
            
            # 排序并返回
            return sorted(new_segments)
        
        # 分割数过多
        else:
            # 计算每个分割的宽度
            widths = [(end - start, i) for i, (start, end) in enumerate(segments)]
            
            # 按宽度排序
            widths.sort()
            
            # 从宽度小的开始合并
            to_merge = []
            for _, idx in widths[:len(segments) - expected_chars]:
                to_merge.append(idx)
            
            # 对合并索引排序
            to_merge.sort()
            
            # 合并相邻的分割
            merged_segments = []
            skip_next = False
            
            for i in range(len(segments)):
                if skip_next:
                    skip_next = False
                    continue
                    
                # 如果当前索引需要与下一个合并
                if i in to_merge and i + 1 < len(segments):
                    merged_segments.append((segments[i][0], segments[i+1][1]))
                    skip_next = True
                elif i not in to_merge:
                    merged_segments.append(segments[i])
            
            # 确保我们有正确的数量
            if len(merged_segments) > expected_chars:
                # 如果还是太多，从最小宽度的开始移除
                merged_segments.sort(key=lambda s: s[1] - s[0])
                merged_segments = merged_segments[len(merged_segments) - expected_chars:]
                
            # 按位置排序
            merged_segments.sort(key=lambda s: s[0])
            
            return merged_segments
    
    def _extract_characters(self, binary_plate, segments):
        """从分割的区域提取字符图像
        
        Args:
            binary_plate: 二值化车牌图像
            segments: 分割区域 [(start, end), ...]
            
        Returns:
            list: 字符图像列表
        """
        char_imgs = []
        
        for i, (start, end) in enumerate(segments):
            # 确保索引有效
            if start < 0:
                start = 0
            if end > binary_plate.shape[1]:
                end = binary_plate.shape[1]
                
            if start >= end:
                logger.warning(f"无效的字符区域索引: {start}:{end}")
                continue
                
            # 提取字符区域
            char_img = binary_plate[:, start:end]
            
            # 去除上下空白
            char_rows = np.sum(char_img, axis=1)
            
            # 找到字符的上下边界
            non_zero_rows = np.where(char_rows > 0)[0]
            if len(non_zero_rows) > 0:
                top = non_zero_rows[0]
                bottom = non_zero_rows[-1]
                
                # 确保区域有效
                if bottom > top:
                    char_img = char_img[top:bottom+1, :]
                    
                    # 保存原始字符图像用于调试
                    if self.debug_mode:
                        cv2.imwrite(f"debug_chars/07_char_{i+1}_orig.jpg", char_img)
                    
                    # 调整大小为标准尺寸
                    char_h, char_w = char_img.shape
                    
                    if char_h > 0 and char_w > 0:
                        # 调整大小为标准尺寸，保持宽高比
                        target_h = self.char_h
                        target_w = min(int(char_w * (target_h / char_h)), self.char_w * 2)
                        
                        # 对于第一个字符（省份缩写），使用更宽的尺寸
                        if i == 0:
                            target_w = max(target_w, int(self.char_w * 1.3))
                        
                        # 调整大小
                        try:
                            char_img = cv2.resize(char_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                            
                            # 添加边缘填充，使所有字符具有相同的尺寸
                            if target_w < self.char_w:
                                # 水平居中填充
                                h_pad = (self.char_w - target_w) // 2
                                char_img = cv2.copyMakeBorder(
                                    char_img, 0, 0, h_pad, self.char_w - target_w - h_pad,
                                    cv2.BORDER_CONSTANT, value=0
                                )
                            
                            # 保存处理后的字符图像用于调试
                            if self.debug_mode:
                                cv2.imwrite(f"debug_chars/08_char_{i+1}_final.jpg", char_img)
                            
                            char_imgs.append(char_img)
                        except Exception as e:
                            logger.error(f"调整字符大小失败: {str(e)}")
        
        return char_imgs
    
    def segment_chars(self, plate_img, plate_color=None):
        """将车牌图像分割为单个字符
        
        Args:
            plate_img: 车牌图像
            plate_color: 车牌颜色，用于判断是否为新能源车牌
            
        Returns:
            list: 分割后的字符图像列表
            
        Raises:
            CharacterSegmentationError: 字符分割失败时抛出
        """
        try:
            if plate_img is None:
                raise CharacterSegmentationError("输入车牌图像为空")
                
            logger.info("开始分割字符")
            
            # 确定预期字符数量
            expected_chars = self.green_plate_chars if plate_color == "green" else self.std_plate_chars
            logger.info(f"预期字符数量: {expected_chars}")
            
            # 预处理车牌图像，获取多种二值化结果
            gray_plate, binary_results = self._preprocess_plate(plate_img)
            
            # 对每种二值化结果尝试分割，选择最佳结果
            best_segments = []
            best_quality = -1
            best_binary = None
            
            # 对每种二值化方法尝试多种分割方法
            for name, binary in binary_results:
                # 方法1：投影法
                segments1 = self._segment_with_projection(binary, expected_chars=expected_chars)
                quality1 = self._compute_segment_quality(segments1, binary)
                
                # 方法2：轮廓法
                segments2 = self._segment_with_contours(binary, expected_chars=expected_chars)
                quality2 = self._compute_segment_quality(segments2, binary)
                
                # 方法3：固定比例法
                segments3 = self._segment_with_fixed_ratio(binary, expected_chars=expected_chars)
                quality3 = self._compute_segment_quality(segments3, binary)
                
                # 选择当前二值化方法下的最佳分割
                methods = [(segments1, quality1), (segments2, quality2), (segments3, quality3)]
                methods.sort(key=lambda x: x[1], reverse=True)
                
                current_best_segments, current_best_quality = methods[0]
                
                if self.debug_mode:
                    logger.debug(f"二值化方法 {name} 的分割质量: 投影法={quality1:.4f}, 轮廓法={quality2:.4f}, 固定比例法={quality3:.4f}")
                
                # 更新全局最佳分割
                if current_best_quality > best_quality:
                    best_quality = current_best_quality
                    best_segments = current_best_segments
                    best_binary = binary
            
            if not best_segments or len(best_segments) < 5:
                logger.warning("分割失败，使用固定比例法")
                best_segments = self._get_fixed_segments(gray_plate, expected_chars)
            
            # 从二值化图像中提取字符
            char_imgs = self._extract_characters(best_binary, best_segments)
            
            # 确保分割出的字符数量符合预期
            if len(char_imgs) != expected_chars:
                logger.warning(f"分割出的字符数量 {len(char_imgs)} 不符合预期 {expected_chars}")
                
                # 如果字符数量太少，添加空白图像
                if len(char_imgs) < expected_chars:
                    for _ in range(expected_chars - len(char_imgs)):
                        blank_char = np.zeros((self.char_h, self.char_w), dtype=np.uint8)
                        char_imgs.append(blank_char)
                        logger.warning("添加空白字符以达到预期数量")
                
                # 如果字符数量太多，保留前几个
                if len(char_imgs) > expected_chars:
                    char_imgs = char_imgs[:expected_chars]
                    logger.warning(f"裁剪字符数量至预期的 {expected_chars}")
            
            logger.info(f"成功分割出 {len(char_imgs)} 个字符")
            return char_imgs
            
        except Exception as e:
            if isinstance(e, CharacterSegmentationError):
                raise e
            else:
                logger.error(f"字符分割失败: {str(e)}")
                raise CharacterSegmentationError(f"字符分割失败: {str(e)}")
        
        return [] 