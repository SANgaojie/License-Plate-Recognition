"""
字符识别模块
负责识别分割后的字符
"""

import os
import cv2
import numpy as np
from numpy.linalg import norm
from modules.utils.exceptions import ModelLoadError, CharacterRecognitionError
from modules.utils.logger import logger
from modules.image_processing.image_utils import deskew

class StatModel(object):
    """统计模型基类，具有加载保存功能"""
    
    def load(self, fn):
        """加载模型
        
        Args:
            fn: 模型文件路径
            
        Returns:
            模型对象
        """
        self.model = self.model.load(fn)
        return self

    def save(self, fn):
        """保存模型
        
        Args:
            fn: 保存路径
        """
        self.model.save(fn)

class SVM(StatModel):
    """支持向量机类，可以训练模型和测试模型"""
    
    def __init__(self, C=1, gamma=0.5):
        """初始化SVM模型
        
        Args:
            C: 正则化参数
            gamma: 核函数参数
        """
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        """训练SVM模型
        
        Args:
            samples: 训练样本
            responses: 标签
        """
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        """字符识别
        
        Args:
            samples: 待识别的样本
            
        Returns:
            预测结果
        """
        r = self.model.predict(samples)
        return r[1].ravel()

class CharRecognizer:
    """字符识别类，负责识别分割后的字符"""
    
    def __init__(self):
        """初始化字符识别器"""
        # 汉字在ASCII中的起始偏移，用于区分汉字和普通字符
        self.PROVINCE_START = 1000
        
        # 中国各省份的编码与汉字对应关系
        self.provinces = [
            "zh_cuan", "川",
            "zh_e", "鄂",
            "zh_gan", "赣",
            "zh_gan1", "甘",
            "zh_gui", "贵",
            "zh_gui1", "桂",
            "zh_hei", "黑",
            "zh_hu", "沪",
            "zh_ji", "冀",
            "zh_jin", "津",
            "zh_jing", "京",
            "zh_jl", "吉",
            "zh_liao", "辽",
            "zh_lu", "鲁",
            "zh_meng", "蒙",
            "zh_min", "闽",
            "zh_ning", "宁",
            "zh_qing", "靑",
            "zh_qiong", "琼",
            "zh_shan", "陕",
            "zh_su", "苏",
            "zh_sx", "晋",
            "zh_wan", "皖",
            "zh_xiang", "湘",
            "zh_xin", "新",
            "zh_yu", "豫",
            "zh_yu1", "渝",
            "zh_yue", "粤",
            "zh_yun", "云",
            "zh_zang", "藏",
            "zh_zhe", "浙"
        ]
        
        # 初始化模型
        self.model = SVM(C=1, gamma=0.5)       # 识别英文字母和数字的模型
        self.modelchinese = SVM(C=1, gamma=0.5)  # 识别中文的模型
        
        # 加载预训练模型
        self.load_models()
    
    def load_models(self):
        """加载预训练的模型"""
        try:
            logger.info("正在加载字符识别模型")
            
            # 获取根目录路径
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            # 加载英文数字识别模型
            model_path = os.path.join(root_dir, "svm.dat")
            if os.path.exists(model_path):
                self.model.load(model_path)
                logger.info("成功加载英文数字识别模型")
            else:
                logger.warning("未找到英文数字识别模型，需要先训练")
            
            # 加载中文识别模型
            model_chinese_path = os.path.join(root_dir, "svmchinese.dat")
            if os.path.exists(model_chinese_path):
                self.modelchinese.load(model_chinese_path)
                logger.info("成功加载中文识别模型")
            else:
                logger.warning("未找到中文识别模型，需要先训练")
        
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise ModelLoadError(f"加载模型失败: {str(e)}")
    
    def preprocess_hog(self, digits):
        """提取图像的方向梯度直方图HOG特征
        
        Args:
            digits: 字符图像列表
            
        Returns:
            numpy.ndarray: HOG特征
        """
        samples = []
        for img in digits:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)  # 计算梯度幅值和角度
            bin_n = 16
            bin = np.int32(bin_n * ang / (2 * np.pi))
            bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
            mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
            hist = np.hstack(hists)  # hist包含4个元素，每个元素是16维的
            
            # 特征归一化
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps
            samples.append(hist)
        return np.float32(samples)
    
    def train_svm(self):
        """训练SVM模型"""
        try:
            logger.info("开始训练字符识别模型")
            
            # 获取根目录路径
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            # 训练识别英文数字模型
            chars_train = []
            chars_label = []
            
            for root, dirs, files in os.walk(os.path.join(root_dir, "train", "chars2")):
                if len(os.path.basename(root)) > 1:  # 跳过非单字符目录
                    continue
                root_int = ord(os.path.basename(root))  # 返回字符对应的ASCII码
                
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    if digit_img is None:
                        logger.warning(f"无法读取训练图像: {filepath}")
                        continue
                    
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(root_int)
            
            chars_train = list(map(deskew, chars_train))  # 对所有训练图片进行校正
            chars_train = self.preprocess_hog(chars_train)  # 提取HOG特征
            chars_label = np.array(chars_label)
            
            if len(chars_train) > 0:
                logger.info(f"英文数字训练样本: {len(chars_train)}")
                self.model.train(chars_train, chars_label)
                
                # 保存模型
                model_path = os.path.join(root_dir, "svm.dat")
                self.model.save(model_path)
                logger.info(f"英文数字识别模型已保存: {model_path}")
            else:
                logger.warning("没有找到英文数字训练样本")
            
            # 训练识别汉字模型
            chars_train = []
            chars_label = []
            
            for root, dirs, files in os.walk(os.path.join(root_dir, "train", "charsChinese")):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                    
                pinyin = os.path.basename(root)
                index = self.provinces.index(pinyin) + self.PROVINCE_START + 1  # 1是拼音对应的汉字索引
                
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    if digit_img is None:
                        logger.warning(f"无法读取训练图像: {filepath}")
                        continue
                    
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(index)
            
            chars_train = list(map(deskew, chars_train))
            chars_train = self.preprocess_hog(chars_train)
            chars_label = np.array(chars_label)
            
            if len(chars_train) > 0:
                logger.info(f"中文训练样本: {len(chars_train)}")
                self.modelchinese.train(chars_train, chars_label)
                
                # 保存模型
                model_chinese_path = os.path.join(root_dir, "svmchinese.dat")
                self.modelchinese.save(model_chinese_path)
                logger.info(f"中文识别模型已保存: {model_chinese_path}")
            else:
                logger.warning("没有找到中文训练样本")
                
        except Exception as e:
            logger.error(f"训练模型失败: {str(e)}")
            raise CharacterRecognitionError(f"训练模型失败: {str(e)}")
    
    def recognize(self, char_imgs):
        """识别字符
        
        Args:
            char_imgs: 字符图像列表
            
        Returns:
            str: 识别结果
            
        Raises:
            CharacterRecognitionError: 字符识别失败时抛出
        """
        try:
            if not char_imgs or len(char_imgs) == 0:
                raise CharacterRecognitionError("输入字符图像为空")
                
            logger.info("开始识别字符")
            
            # 确保图像是单通道的
            processed_chars = []
            for img in char_imgs:
                if len(img.shape) > 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                processed_chars.append(img)
            
            # 特征提取前的图像处理
            processed_chars = list(map(deskew, processed_chars))
            
            # 提取HOG特征
            char_features = self.preprocess_hog(processed_chars)
            
            # 匹配结果
            result = []
            
            # 识别第一个字符（汉字）
            if len(char_features) > 0:
                resp = self.modelchinese.predict(char_features[0].reshape(1, -1))
                province_index = int(resp[0]) - self.PROVINCE_START
                if 0 <= province_index < len(self.provinces) and province_index % 2 == 1:
                    result.append(self.provinces[province_index])
                else:
                    result.append("未")  # 无法识别的汉字
            
            # 识别剩余字符
            for i in range(1, len(char_features)):
                resp = self.model.predict(char_features[i].reshape(1, -1))
                result.append(chr(int(resp[0])))
            
            # 拼接结果
            plate_str = "".join(result)
            logger.info(f"字符识别结果: {plate_str}")
            
            return plate_str
            
        except Exception as e:
            if isinstance(e, CharacterRecognitionError):
                raise e
            else:
                logger.error(f"字符识别失败: {str(e)}")
                raise CharacterRecognitionError(f"字符识别失败: {str(e)}")
        
        return "" 