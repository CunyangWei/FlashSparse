#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 2:
        print("用法: python test.py <matrix_file.mtx>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as f:
            # 跳过所有以 '%' 开头的注释或头部行
            for line in f:
                if line.startswith('%'):
                    continue
                # 第一个非注释行应包含矩阵的尺寸信息
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        # 假设第一项为行数（高），第二项为列数（宽）
                        height = int(parts[0])
                        width = int(parts[1])
                        # 如果存在第三个数字，则它就是非零元个数（nnz）
                        if len(parts) >= 3:
                            nnz = int(parts[2])
                        else:
                            # 若没有给出nnz，则假设为满矩阵（即所有元素非零）
                            nnz = height * width
                        print("宽: {}, 高: {}, nnz: {}".format(width, height, nnz))
                        return
                    except ValueError:
                        print("尺寸行格式不正确，请检查文件格式。")
                        sys.exit(1)
            # 如果文件中没有找到尺寸行
            print("未在文件中找到矩阵尺寸信息。")
    except FileNotFoundError:
        print("文件未找到: {}".format(filename))
        sys.exit(1)

if __name__ == '__main__':
    main()

