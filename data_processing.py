import pandas as pd
import pyperclip
import numpy as np

def clipboard_to_df():
    """
    从剪贴板读取数据并转换为pandas DataFrame。
    假设剪贴板中的数据是以制表符分隔的（例如，从Excel复制的数据）。
    """
    try:
        # 尝试从剪贴板读取数据
        df = pd.read_clipboard(sep='\t')
        print("数据已成功从剪贴板读取。")
        return df
    except Exception as e:
        print(f"读取剪贴板数据时出错: {e}")
        return None

def df_to_numpy(df):
    """
    将pandas DataFrame转换为NumPy数组。
    """
    try:
        array = df.to_numpy()
        print("DataFrame已成功转换为NumPy数组。")
        return array
    except Exception as e:
        print(f"将DataFrame转换为NumPy数组时出错: {e}")
        return None

def get_data_from_clipboard():
    """
    从剪贴板获取数据，并将其转换为NumPy数组。
    """
    df = clipboard_to_df()
    if df is not None:
        array = df_to_numpy(df)
        return array
    else:
        return None
