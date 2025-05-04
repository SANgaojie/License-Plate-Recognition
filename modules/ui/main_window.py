"""
用户界面模块
提供车牌识别系统的图形用户界面
"""

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import time
import os
import threading

from modules.utils.exceptions import *
from modules.utils.logger import logger
from modules.license_plate_recognizer import LicensePlateRecognizer

class MainWindow(ttk.Frame):
    """主窗口类，提供用户界面"""
    
    def __init__(self, win):
        """初始化主窗口
        
        Args:
            win: 根窗口
        """
        ttk.Frame.__init__(self, win)
        self.win = win
        
        # 配置窗口
        win.title("车牌识别系统")
        win.state("normal")
        
        # 图像显示尺寸
        self.view_width = 600
        self.view_height = 600
        self.pic_path = ""
        self.update_time = 0
        
        # 车牌颜色转换
        self.color_transform = {
            "blue": ("蓝", "#6666ff"),
            "yello": ("黄", "#ffff00"),
            "green": ("绿", "#55ff55")
        }
        
        # 初始化车牌识别器
        try:
            self.recognizer = LicensePlateRecognizer()
            self.recognizer.set_debug_mode(True, self.show_debug_image)
            
            # 创建UI组件
            self._create_widgets()
            
            # 加载模型
            self.recognizer.train_models()
            
        except Exception as e:
            logger.critical(f"初始化主窗口失败: {str(e)}")
            messagebox.showerror("错误", f"初始化失败: {str(e)}")
    
    def _create_widgets(self):
        """创建UI组件"""
        # 布局设置
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        
        # 左侧框架 - 用于显示原图
        frame_left = ttk.Frame(self)
        frame_left.pack(side=tk.LEFT, expand=1, fill=tk.BOTH)
        
        # 右侧框架 - 用于显示车牌区域和识别结果
        frame_right1 = ttk.Frame(self)
        frame_right1.pack(side=tk.TOP, expand=1, fill=tk.Y)
        
        # 右下角框架 - 用于放置控制按钮
        frame_right2 = ttk.Frame(self)
        frame_right2.pack(side=tk.RIGHT, expand=0)
        
        # 原图标签
        ttk.Label(frame_left, text='原图：').pack(anchor="nw")
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")
        
        # 车牌区域标签
        ttk.Label(frame_right1, text='车牌区域：').grid(column=0, row=0, sticky=tk.W)
        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        
        # 识别结果标签
        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.result_ctl = ttk.Label(frame_right1, text="")
        self.result_ctl.grid(column=0, row=3, sticky=tk.W)
        
        # 车牌颜色标签
        self.color_ctl = ttk.Label(frame_right1, text="", width="20")
        self.color_ctl.grid(column=0, row=4, sticky=tk.W)
        
        # 打开图片按钮
        open_pic_btn = ttk.Button(frame_right2, text="打开图片", width=20, command=self._on_open_image)
        open_pic_btn.pack(anchor="se", pady="5")
        
        # 调试信息区域
        ttk.Label(frame_right1, text='调试信息：').grid(column=0, row=5, sticky=tk.W)
        self.debug_text = tk.Text(frame_right1, height=10, width=30)
        self.debug_text.grid(column=0, row=6, sticky=tk.W)
        
        # 调试图像区域
        self.debug_frame = ttk.Frame(frame_right1)
        self.debug_frame.grid(column=0, row=7, sticky=tk.W)
        self.debug_labels = []
    
    def _on_open_image(self):
        """打开图片按钮点击事件"""
        try:
            # 选择图片文件
            self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("图片文件", "*.jpg;*.png;*.jpeg;*.bmp")])
            if not self.pic_path:
                return
                
            # 显示原图
            img_bgr = cv2.imread(self.pic_path)
            if img_bgr is None:
                raise ImageLoadError(f"无法读取图像: {self.pic_path}")
                
            # 调整图像大小并显示
            self.imgtk = self._get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            
            # 清空之前的调试图像
            self._clear_debug_images()
            
            # 识别车牌
            try:
                result, roi, color = self.recognizer.recognize(img_bgr)
                self._show_recognition_result(result, roi, color)
                
            except Exception as e:
                logger.error(f"识别失败: {str(e)}")
                messagebox.showerror("识别失败", str(e))
                self._clear_result()
                
        except Exception as e:
            logger.error(f"打开图片失败: {str(e)}")
            messagebox.showerror("错误", f"打开图片失败: {str(e)}")
    
    def _show_recognition_result(self, result, roi, color):
        """显示识别结果
        
        Args:
            result: 识别结果文字
            roi: 车牌区域图像
            color: 车牌颜色
        """
        if result and roi is not None:
            # 显示车牌区域
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_img = Image.fromarray(roi_rgb)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi_img)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            
            # 显示识别结果
            self.result_ctl.configure(text=str(result))
            
            # 记录更新时间
            self.update_time = time.time()
            
            # 显示车牌颜色
            try:
                c = self.color_transform.get(color, None)
                if c:
                    self.color_ctl.configure(text=c[0], background=c[1], state='enable')
                else:
                    self.color_ctl.configure(text="未知", background="#FFFFFF", state='enable')
            except Exception as e:
                logger.error(f"显示车牌颜色失败: {str(e)}")
                self.color_ctl.configure(state='disabled')
                
        elif self.update_time + 8 < time.time():
            self._clear_result()
    
    def _clear_result(self):
        """清空识别结果显示"""
        self.roi_ctl.configure(state='disabled')
        self.result_ctl.configure(text="")
        self.color_ctl.configure(state='disabled')
    
    def _get_imgtk(self, img_bgr):
        """将OpenCV图像转换为Tkinter可用的格式
        
        Args:
            img_bgr: BGR格式的OpenCV图像
            
        Returns:
            ImageTk.PhotoImage: Tkinter可用的图像对象
        """
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # OpenCV是BGR格式，PIL是RGB
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        
        # 调整图像大小
        width = imgtk.width()
        height = imgtk.height()
        
        if width > self.view_width or height > self.view_height:
            # 按比例缩放
            width_factor = self.view_width / width
            height_factor = self.view_height / height
            factor = min(width_factor, height_factor)
            
            width = int(width * factor)
            height = int(height * factor)
            
            # 确保宽高至少为1
            width = max(1, width)
            height = max(1, height)
            
            im = im.resize((width, height), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=im)
            
        return imgtk
    
    def show_debug_image(self, img, title):
        """显示调试图像
        
        Args:
            img: 要显示的图像
            title: 图像标题
        """
        if img is None:
            return
            
        # 添加调试日志
        logger.debug(f"显示调试图像: {title}")
        self.debug_text.insert(tk.END, f"{title}\n")
        self.debug_text.see(tk.END)
        
        # 调整大小
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 统一调整大小为100x100
        img_resized = cv2.resize(img_rgb, (100, 100), interpolation=cv2.INTER_AREA)
        im = Image.fromarray(img_resized)
        imgtk = ImageTk.PhotoImage(image=im)
        
        # 创建新的标签显示图像
        label_frame = ttk.Frame(self.debug_frame)
        label_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        img_label = ttk.Label(label_frame, image=imgtk)
        img_label.pack()
        img_label.image = imgtk  # 保持引用，防止被垃圾回收
        
        text_label = ttk.Label(label_frame, text=title)
        text_label.pack()
        
        self.debug_labels.append((label_frame, img_label, text_label))
    
    def _clear_debug_images(self):
        """清空调试图像"""
        for frame, _, _ in self.debug_labels:
            frame.destroy()
        self.debug_labels = []
        self.debug_text.delete(1.0, tk.END)

def close_window(main_window, root):
    """关闭窗口"""
    logger.info("关闭应用程序")
    root.destroy()

def main():
    """主函数"""
    try:
        # 创建窗口
        root = tk.Tk()
        
        # 创建主窗口
        main_window = MainWindow(root)
        
        # 设置窗口关闭事件
        root.protocol('WM_DELETE_WINDOW', lambda: close_window(main_window, root))
        
        # 运行事件循环
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"应用程序运行失败: {str(e)}")
        messagebox.showerror("错误", f"应用程序运行失败: {str(e)}")

if __name__ == '__main__':
    main() 