# 图像数学题求解系统

## 项目简介
基于 Qwen2.5-VL-3B 模型，接收包含数学题图片的 JSONL 数据，自动提取题目、生成解题思路与答案，并将结果以 JSONL 格式输出。

## 环境搭建

### 1. 克隆仓库
```shell
git clone https://github.com/thomaswei-cn/math_solver_qwen.git
cd math_solver_qwen
```

### 2. 创建并激活 Conda 虚拟环境
```shell
conda create -n math python=3.10 -y
conda activate math
```

### 3. 安装依赖
```shell
pip install -r requirements.txt
```

## 数据准备
- 输入文件：包含字段 `"image"` 的 JSONL 格式，存放所有待处理图片文件名。
- 图片目录：存放对应的图片文件，文件名需与 JSONL 中一致。

## 脚本说明

- `build_env.sh`：构建环境示例脚本。
- `run.py`：核心推理脚本，加载模型、处理图片与提示模板，生成思路与答案，并输出 JSONL。
- `run.sh`：调用 `run.py` 的示例运行脚本。

## 使用方法

### 执行构建环境脚本
```shell
bash build_env.sh
```

### 运行推理脚本
```shell
bash run.sh
```

#### 参数说明
- `IMAGE_INPUT_DIR`：图片所在目录路径
- `Query_PATH`：输入 JSONL 文件路径
- `OUPUT_PATH`：输出结果 JSONL 文件路径

## 输出格式
每行 JSON 对象包含：
- `image`：原图片文件名  
- 'tag': 题目类型
- `step`：解题思路（LaTeX 格式）
- `answer`：最终答案  

## 文件结构
- `build_env.sh`  
- `run.py`  
- `run.sh`  
- `requirements.txt`  
