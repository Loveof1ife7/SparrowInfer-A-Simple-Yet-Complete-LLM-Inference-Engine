# 1. 清理旧编译文件
python setup.py clean

# 2. 删除编译产物
rm -rf build/ dist/ *.egg-info *.so

# 3. 重新编译安装
python setup.py build develop
