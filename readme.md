# 通过参数传入
python main.py example-s6 2026ai-result --api-key sk-xxxxxxxx

# 通过环境变量传入
export SILICONFLOW_API_KEY=sk-xxxxxxxx
python main.py example-s6 2026ai-result

# 只指定输入文件夹（输出默认到 2026ai-result）
python main.py example-s6 --api-key sk-xxxxxxxx
