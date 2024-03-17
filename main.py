from tkinter import Tk
from gui import App

def main():
    root = Tk()  # 创建一个Tk根窗口实例
    app = App(root)  # 将Tk实例传递给App类
    root.mainloop()  # 直接使用Tk的mainloop方法启动GUI主循环

if __name__ == "__main__":
    main()
