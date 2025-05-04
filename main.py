"""
车牌识别系统主程序入口
"""

import tkinter as tk
from modules.ui.main_window import MainWindow, close_window
from modules.utils.logger import logger

def main():
    """主函数"""
    try:
        logger.info("启动车牌识别系统")
        
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
        print(f"应用程序运行失败: {str(e)}")

if __name__ == '__main__':
    main() 