import subprocess
import sys

# 1. 运行 stitching.py
print("=== 阶段一：全局拼接（new_stitching.py） ===")
ret1 = subprocess.run([sys.executable, "new_stitching.py"], check=True)
if ret1.returncode != 0:
    print("new_stitching.py 运行失败！")
    sys.exit(1)

# 2. 运行 bigmapstiching.py
print("=== 阶段二：大图拼接（bigmapstiching.py） ===")
ret2 = subprocess.run([sys.executable, "bigmapstiching.py"], check=True)
if ret2.returncode != 0:
    print("bigmapstiching.py 运行失败！")
    sys.exit(1)

print("=== 全部拼接流程完成！===")