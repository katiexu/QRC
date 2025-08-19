## 环境准备
  将torchquantum 数据集放入根目录，运行 'run_all_test.py'

## 模型介绍

  模型经过qrnn层后，提取特征到单层神经网络输出。核心代码： 
```
    def process(params, input_seq):
        # qrnn 循环
        features = jnp.array(circuit(params, input_seq))
        
        # 单层神经网络
        logits = jnp.dot(features, params['linear_weights']) + params['linear_bias']
        return logits
```
1. heils_qrc_end_all_expval
    数据集被分箱8份，以二进制(x,x,x)表示
``` 
    X_train = transform(X_train)
    X_test = transform(X_test)
``` 

   该模型将每一轮qubit的测量值传入下一层  
``` 
    def circuit(params, input_seq):
        all_expval=[]
        ...
        for i, x in enumerate(input_seq):
            ...
            all_expval.append(self.get_first_3qubit_expvals(rho))
        return jnp.array(all_expval)
```
2. heils_qrc_end_qubit  
    该模型最后将所有qubit的测量值传入下一层  
```
    @qml.qnode(dev, **self.qnode_kwargs)
        def single_qubit_circuit_end(hidden_state, params, n_qubits, n_layers):
            ......
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits_)]
            
    def circuit(params, input_seq):
        ......
        return single_qubit_circuit_end(hidden_state, params["weights"],
                                            self.n_qubits_, self.n_layers_)
```
3. heils_qrc_end_feature  
    该模型最后一层rho，提取特征传入下一层，参考函数 ‘get_quantum_features’
```
    def get_quantum_features(self, rho):
        """从密度矩阵中提取量子特征"""
        ......
        return jnp.array(features)

    def circuit(params, input_seq):
        ......
        return self.get_quantum_features(rho)
```
4. heils_qrc_end_feature  
    该模型将每轮rho提取特征，最后将所有特征传入下一层。
```
    def circuit(params, input_seq):
        all_features=[]
        ...
        for i, x in enumerate(input_seq):
            ...
            all_features.extend(self.get_quantum_features(rho))
        return jnp.array(all_features)

```