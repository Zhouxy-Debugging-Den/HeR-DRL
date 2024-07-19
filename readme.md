## 一、环境配置
```bash
conda create -n HeR-DRL-torch2 python=3.8
conda activate HeR-DRL-torch2
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
cd HeR-DRL/Python-RVO2
pip install Cython
python setup.py build
python setup.py install
cd ../socialforce
pip install -e '.[test,plot]'
pylint socialforce
cd ..
pip install --upgrade pip setuptools==57.5.0
pip install --upgrade pip wheel==0.38.4
pip install -e .
pip install protobuf==3.20.2
cd crowd_nav
pip install gitpython
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch_geometric==2.2.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard

```
## 二、测试
### 1. single-robot 测试
```bash
cd crowd_nav
. scripts/test/simple_test_1000.sh
```
### 2. mulit-robot 测试
```bash
cd crowd_nav
. scripts/test/complex_test_1000.sh
```

### 3. multi-robot ablation 测试
将`\crowd_nav\policy\multi\cadrl.py`中的`self.multi_carto`改为`False`
```bash
cd crowd_nav
. scripts/test/complex_ablation_test_1000.sh
```
### Note
The training code will be released later

