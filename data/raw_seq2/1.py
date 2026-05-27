def process_cv_file_simple(input_file, output_file):
    """
    极简版处理：按每4行一组，保留第1、2、4行，删除第3行
    输入：每组4行（名称、序列、高质量标签、混合标签）
    输出：每组3行（名称、序列、混合标签）
    """
    # 读取所有非空行（过滤空行，避免干扰分组）
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = [line.rstrip('\n') for line in fin if line.strip()]  # 保留换行符外的内容，便于还原格式
    
    # 按每4行分组处理
    processed_lines = []
    # 遍历所有行，步长为4（每组4行）
    for i in range(0, len(lines), 4):
        # 取当前组的4行（防止最后一组行数不足）
        group = lines[i:i+4]
        if len(group) >= 4:
            # 保留第1、2、4行（索引0、1、3），跳过第3行（索引2）
            processed_lines.append(group[0])  # 名称行
            processed_lines.append(group[1])  # 序列行
            processed_lines.append(group[3])  # 高低质量混合标签行
        else:
            # 容错：行数不足4行时打印警告，跳过该不完整组
            print(f"警告：行{i+1}开始的分组行数不足4行，跳过该组")
            continue
    
    # 写入处理后的文件
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(processed_lines) + '\n')  # 每行换行，最后补一个换行符
    
    print(f"处理完成！原始文件共{len(lines)}行，处理后共{len(processed_lines)}行")
    print(f"结果已保存至：{output_file}")

# ======================== 执行处理 ========================
if __name__ == "__main__":
    # 替换为你的实际文件路径
    input_path = "CV4.af"          # 原始文件
    output_path = "CV4_processed.txt"  # 处理后的文件
    
    process_cv_file_simple(input_path, output_path)