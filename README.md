## 环境准备
    将torchquantum 数据集放入根目录，运行 'run_all_test.py'

## 模型介绍
1. heils_qrc_end_qubit  
    该模型最后将所有qubit的测量值传入下一层
2. heils_qrc_end_feature  
    该模型最后一层rho，提取特征传入下一层，参考函数 ‘get_quantum_features’
3. heils_qrc_end_feature  
    该模型将每轮rho提取特征，最后将所有特征传入下一层。