from sklearn.base import BaseEstimator, ClassifierMixin
import pennylane as qml
import numpy as np
from model_utils import *
jax.config.update("jax_enable_x64", True)
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
class SeparableVariationalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        encoding_layers=1,
        learning_rate=0.001,
        batch_size=32,
        max_vmap=None,
        jit=True,
        max_steps=10000,
        random_state=42,
        scaling=1.0,
        convergence_interval=200,
        dev_type="default.mixed",
        qnode_kwargs={"interface": "jax"},
        n_qubits=4,
        n_layers=4,
        n_classes=4
    ):
        r"""
        Variational model that uses only separable operations (i.e. there is no entanglement in the model). The circuit
        consists of layers of encoding gates and parameterised unitaries followed by measurement of an observable.

        Each encoding layer consists of a trainiable arbitrary qubit rotation on each qubit followed by
        a product angle embedding of the input data, using RY gates. A final layer of trainable qubit rotations is
        applied at the end of the circuit.

        The obserable O is the mean value of Pauli Z observables on each of the output qubits. The value of this
        observable is used to predict the probability for class 1 as :math:`P(+1)=\sigma(6\langle O \rangle)`
        where :math`\sigma` is the logistic funciton. The model is then fit using the cross entropy loss.

        Args:
            encoding_layers (int): number of layers in the data encoding circuit.
            learning_rate (float): learning rate for gradient descent.
            batch_size (int): Size of batches used for computing parameter updates.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            jit (bool): Whether to use just in time compilation.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): the keyword arguments passed to the circuit qnode.
            random_state (int): Seed used for pseudorandom number generation
        """
        # attributes that do not depend on data
        self.encoding_layers = encoding_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.convergence_interval = convergence_interval
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.n_qubits_=n_qubits
        self.n_layers_=n_layers
        self.n_classes_=n_classes
        self.g = 0.5
        self.M_matrix = self.get_M_matrix()
        if self.jit:
            self.merge_x = jax.jit(self.merge_x)
            # self.partial_trace=jax.jit(self.partial_trace)
            self.collapse_density_matrix = jax.jit(self.collapse_density_matrix)

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))
    def get_quantum_features(self, rho):
        """从密度矩阵中提取量子特征"""
        features = []

        # 提取单量子比特期望值
        for w in range(self.n_qubits_):
            Z_op = qml.PauliZ(w).matrix()
            # 构建完整算子
            full_op = jnp.array([[1.0]])
            for wire in range(self.n_qubits_):
                if wire == w:
                    full_op = jnp.kron(full_op, Z_op)
                else:
                    full_op = jnp.kron(full_op, jnp.eye(2))
            expval = jnp.trace(rho @ full_op).real
            features.append(expval)

        # 提取ZZ关联期望值
        for w1 in range(self.n_qubits_):
            for w2 in range(w1 + 1, self.n_qubits_):
                op1 = qml.PauliZ(w1).matrix()
                op2 = qml.PauliZ(w2).matrix()

                # 构建完整算子
                full_op = jnp.array([[1.0]])
                for wire in range(self.n_qubits_):
                    if wire == w1 or wire == w2:
                        if wire == w1:
                            full_op = jnp.kron(full_op, op1)
                        else:
                            full_op = jnp.kron(full_op, op2)
                    else:
                        full_op = jnp.kron(full_op, jnp.eye(2))

                expval = jnp.trace(rho @ full_op).real
                features.append(expval)

        return features
    def get_M_matrix(self):
        g = self.g
        Mi = jnp.array([[1.0, jnp.exp(-g ** 2 / 2)],
                        [jnp.exp(-g ** 2 / 2), 1]], dtype=jnp.complex128)

        # 对第一个量子位应用测量，其他量子位用单位矩阵
        M = Mi
        for _ in range(self.n_qubits_ - 1):
            M = jnp.kron(M, Mi)
        return M
    def merge_x(self,x,rho_reduced):
        rho_A = jnp.array([
            [1.0 - x, jnp.sqrt((1.0 - x) * x)],
            [jnp.sqrt((1.0 - x) * x), x]
        ], dtype=jnp.complex128)
        # return jnp.kron(rho_A, rho_reduced)   #合并到第一个qubit
        return jnp.kron(rho_reduced,rho_A)   #合并到最后一个qubit

    def partial_trace(self,rho, keep_wires):
        # rho: 当前密度矩阵（shape: 16x16）
        # keep_wires: 要保留的量子比特（如 [1,2,3]）

        # 计算要 trace out 的量子比特（这里是 wire=0）
        dim_keep = 2 ** len(keep_wires)
        dim_trace = 2 ** (self.n_qubits_ - len(keep_wires))  # 4-qubit 系统

        # 重塑为 (2, 8, 2, 8) 并求和
        rho_reshaped = rho.reshape((dim_trace, dim_keep, dim_trace, dim_keep))
        rho_reduced = jnp.einsum('ijik->jk', rho_reshaped)
        return rho_reduced

    def collapse_density_matrix(self,rho, outcome):
        # 定义 |0⟩⟨0| 和 |1⟩⟨1| 投影算符
        proj0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
        proj1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)

        # 根据测量结果选择投影算符
        proj = jnp.where(outcome == 0, proj0, proj1)

        # 构造全局投影算符（P ⊗ I_{1,2,3}）
        proj_full = jnp.kron(proj, jnp.eye(2**(self.n_qubits_-1)))

        # 坍缩后的密度矩阵: ρ_collapsed = (P ρ P†) / Tr(P ρ P†)
        rho_collapsed = proj_full @ rho @ proj_full.conj().T
        normalization = jnp.trace(rho_collapsed).real
        rho_collapsed /= normalization
        return rho_collapsed
    def construct_model(self):

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def single_qubit_circuit(hidden_state, params, n_qubits, n_layers):
            """修改后的量子循环神经网络单元，返回量子状态作为隐藏状态"""
            # 准备初始状态（使用前一个时间步的隐藏状态）
            if hidden_state is not None:
                qml.QubitDensityMatrix(hidden_state, wires=range(n_qubits))
            else:
                # 初始化状态（|0>态）
                pass

            # 参数化量子电路
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(*params[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # 获取新的隐藏状态（状态向量）
            density_matrix = qml.density_matrix(wires=range(self.n_qubits_))
            return density_matrix

        def circuit(params, input_seq):
            input_seq = jnp.squeeze(input_seq)
            # 初始记忆状态
            hidden_state = None  # 初始状态
            all_features = []  # 存储每个时间步的特征

            # 1. 执行量子电路，获取新 密度矩阵
            rho = single_qubit_circuit(hidden_state, params["weights"],
                                       self.n_qubits_, self.n_layers_)

            # 0. 应用 M 矩阵
            M = self.M_matrix
            rho = M @ rho @ M.conj().T
            rho /= jnp.trace(rho).real  # 归一化

            # 2. 测量
            pauli_z = qml.PauliZ(0).matrix()
            pauli_z_full = jnp.kron(pauli_z, jnp.eye(2 ** (self.n_qubits_ - 1)))
            expval = jnp.trace(rho @ pauli_z_full).real
            for i, x in enumerate(input_seq):
                # 1. 根据概率采样测量结果坍缩
                p0 = (1 + expval) / 2  # 计算测量到0的概率
                key, subkey = jax.random.split(self.generate_key())
                outcome = jax.random.bernoulli(subkey, p0).astype(int)

                # 2. 根据采样结果坍缩状态
                rho_collapsed = self.collapse_density_matrix(rho, outcome)

                # 3. 计算坍缩后 wires=[1,2,3] 的偏迹
                rho_reduced = self.partial_trace(rho_collapsed, keep_wires=list(range(1,self.n_qubits_)))

                # 4. 合并x 生成新的 密度矩阵
                hidden_state = self.merge_x(x, rho_reduced)

                # 5. 执行量子电路，获取新 密度矩阵
                rho = single_qubit_circuit(hidden_state, params["weights"],
                                           self.n_qubits_, self.n_layers_)

                # 0. 应用 M 矩阵
                M = self.M_matrix
                rho = M @ rho @ M.conj().T
                rho /= jnp.trace(rho).real  # 归一化

                all_features.extend(self.get_quantum_features(rho))
                # 6. 测量
                pauli_z = qml.PauliZ(0).matrix()
                pauli_z_full = jnp.kron(pauli_z, jnp.eye(2 ** (self.n_qubits_ - 1)))
                expval = jnp.trace(rho @ pauli_z_full).real
            return jnp.array(all_features)

        def process(params, input_seq):
            features = jnp.array(circuit(params, input_seq))
            logits = jnp.dot(features, params['linear_weights']) + params['linear_bias']
            return logits

        if self.jit:
            process = jax.jit(process)
        self.forward = jax.vmap(process, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize(self):
        """Initialize attributes that depend on the number of features and the class labels.

        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels that the classifier expects
        """
        self.initialize_params()
        self.construct_model()

    def initialize_params(self):
        self.params_ = {
            "weights": jax.random.uniform(self.generate_key(), (self.n_layers_, self.n_qubits_, 3), minval=0, maxval=2 * jnp.pi),
            'scale_factor': jnp.array(0.5),  # 可训练的比例因子
            'linear_weights': jax.random.uniform(self.generate_key(), (16*21, self.n_classes_)) * 0.01,
            'linear_bias': jnp.zeros(self.n_classes_)
        }

    def fit(self, X, y):
        self.initialize()

        optimizer = optax.adam

        def loss_fn(params, X, y):
            # we multiply by 6 because a relevant domain of the sigmoid function is [-6,6]
            vals = self.forward(params, X) * 6
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(vals, y))

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

        return self

    def predict(self, X):
        predictions = self.predict_proba(X)
        return jnp.argmax(predictions, axis=1)

    def predict_proba(self, X):
        predictions = self.chunked_forward(self.params_, X)
        return jax.nn.softmax(predictions, axis=-1)

if __name__ == "__main__":
    dataset_name = 'mnist4'
    from dataset import get_mnist_numpy

    train_datasets, val_datasets, test_datasets = get_mnist_numpy(dataset_name, 6)
    X_train, y_train = train_datasets
    X_test, y_test = test_datasets
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    model = SeparableVariationalClassifier(jit=True, max_vmap=32, n_layers=4, n_qubits=4, n_classes=4)
    model.fit(X_train, y_train)
    train_predictions = np.array(model.predict(X_train))
    test_predictions = np.array(model.predict(X_test))
    train_acc = np.mean(train_predictions == y_train)
    print(f"Train Accuracy: {train_acc:.4f}")
    test_acc = np.mean(test_predictions == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    with open('mnist_end_features.txt', 'a', encoding='utf-8') as file:
        print('test_acc', dataset_name, test_acc, file=file)