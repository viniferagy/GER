#!/bin/bash
# 检查是否至少有一个参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 VAR1=VALUE1 VAR2=VALUE2 ..."
    exit 1
fi

# 遍历所有的参数
for var in "$@"; do
    # 提取变量名和值
    name=$(echo $var | cut -d '=' -f 1)
    value=$(echo $var | cut -d '=' -f 2)
    # 输出变量名和值
    echo "$name=$value"
done