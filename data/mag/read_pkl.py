import pickle
import numpy as np

def read_pkl_file(file_path):
    """
    读取.pkl文件并返回其内容。

    参数:
        file_path (str): .pkl文件的路径。

    返回:
        obj: 从.pkl文件中读取的对象。
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='utf-8')
            return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except pickle.UnpicklingError:
        print(f"文件 {file_path} 不是有效的.pkl文件。")
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")

def print_summary(data):
    """
    打印数据的概要信息。

    参数:
        data (any): 从.pkl文件中读取的数据。
    """
    print(f"data type: {type(data)}")
    print(data)

# 示例用法
if __name__ == "__main__":
    file_path = './remap.pkl'
    data = read_pkl_file(file_path)
    if data is not None:
        print_summary(data)
    
    print('=======================')

    file_path = '../ml-10m.pkl'
    data = read_pkl_file(file_path)
    if data is not None:
        print_summary(data)
