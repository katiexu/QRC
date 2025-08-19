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
            self.xtorho=jax.jit(self.xtorho)
            self.get_first_3qubit_expvals=jax.jit(self.get_first_3qubit_expvals)
            # self.partial_trace=jax.jit(self.partial_trace)
            # self.collapse_density_matrix = jax.jit(self.collapse_density_matrix)

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None
        self.pauli_z = qml.PauliZ(0).matrix()  # 单qubit Z矩阵

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))
    def get_M_matrix(self):
        g = self.g
        Mi = jnp.array([[1.0, jnp.exp(-g ** 2 / 2)],
                        [jnp.exp(-g ** 2 / 2), 1]], dtype=jnp.complex128)

        # 对第一个量子位应用测量，其他量子位用单位矩阵
        M = Mi
        for _ in range(self.n_qubits_ - 1):
            M = jnp.kron(M, Mi)
        return M

    def xtorho(self, x_values):
        """
        将3个x值（每个x值编码为1个qubit）与3-qubit的rho_reduced合并为6-qubit系统

        参数:
            x_values: 形状为(3,)的数组，每个元素∈[0,1]
            rho_reduced: 3-qubit的密度矩阵 (8x8)

        返回:
            6-qubit的密度矩阵 (64x64)
        """
        # 为每个x值构造单qubit密度矩阵
        rho_list = []
        for x in x_values:
            rho_A = jnp.array([
                [1.0 - x, jnp.sqrt((1.0 - x) * x)],
                [jnp.sqrt((1.0 - x) * x), x]
            ], dtype=jnp.complex128)
            rho_list.append(rho_A)

        # 将3个单qubit态合并为3-qubit态
        rho_x = jnp.kron(rho_list[0], jnp.kron(rho_list[1], rho_list[2]))
        return rho_x

    def partial_trace(self,rho, keep_wires):
        dim_keep = 2 ** keep_wires
        dim_trace = 2 ** (self.n_qubits_ - keep_wires)


        rho_reshaped = rho.reshape((dim_trace, dim_keep, dim_trace, dim_keep))
        rho_reduced = jnp.einsum('ijik->jk', rho_reshaped)

        return rho_reduced

    def get_first_3qubit_expvals(self,rho):
        """分别计算前3个qubit的Pauli-Z期望值"""
        expvals = []

        # 为每个qubit构造对应的观测算符
        for qubit_idx in [0, 1, 2]:  # 前3个qubit
            # 构造观测算符：目标qubit是Z，其他是I
            obs = None
            for i in range(self.n_qubits_):
                if i == qubit_idx:
                    component = qml.PauliZ(0).matrix()
                else:
                    component = jnp.eye(2)

                obs = component if obs is None else jnp.kron(obs, component)

            expvals.append(jnp.trace(rho @ obs).real)

        return jnp.array(expvals)
    # def collapse_density_matrix(self,rho, outcome):
    #     # 定义 |0⟩⟨0| 和 |1⟩⟨1| 投影算符
    #     proj0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
    #     proj1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)
    #
    #     # 根据测量结果选择投影算符
    #     proj = jnp.where(outcome == 0, proj0, proj1)
    #
    #     # 构造全局投影算符（P ⊗ I_{1,2,3}）
    #     proj_full = jnp.kron(proj, jnp.eye(2**(self.n_qubits_-1)))
    #
    #     # 坍缩后的密度矩阵: ρ_collapsed = (P ρ P†) / Tr(P ρ P†)
    #     rho_collapsed = proj_full @ rho @ proj_full.conj().T
    #     normalization = jnp.trace(rho_collapsed).real
    #     rho_collapsed /= normalization
    #     return rho_collapsed
    def construct_model(self):

        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def single_qubit_circuit(hidden_state, params, n_qubits, n_layers):
            qml.QubitDensityMatrix(hidden_state, wires=range(n_qubits))

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

            rho_1=self.xtorho(input_seq[0])
            rho_2=self.xtorho(input_seq[1])
            rho=jnp.kron(rho_1,rho_2)

            # 执行量子电路，获取新 密度矩阵
            rho = single_qubit_circuit(rho, params["weights"],
                                       self.n_qubits_, self.n_layers_)

            # 应用 M 矩阵
            M = self.M_matrix
            rho = M @ rho @ M.conj().T
            rho /= jnp.trace(rho).real  # 归一化

            # 测量
            expval=self.get_first_3qubit_expvals(rho)
            all_value = [expval]

            for x in input_seq[2:]:

                # 计算偏迹
                rho_reduced = self.partial_trace(rho, 3)

                # 合并x 生成新的 密度矩阵
                rho_x=self.xtorho(x)
                rho=jnp.kron(rho_x, rho_reduced)

                # 执行量子电路，获取新 密度矩阵
                rho = single_qubit_circuit(rho, params["weights"],
                                           self.n_qubits_, self.n_layers_)

                # 应用 M 矩阵
                M = self.M_matrix
                rho = M @ rho @ M.conj().T
                rho /= jnp.trace(rho).real  # 归一化


                # 测量
                expval=self.get_first_3qubit_expvals(rho)
                all_value.append(expval)

            return jnp.array(all_value).flatten()

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
            'linear_weights': jax.random.uniform(self.generate_key(), (15*3, self.n_classes_)) * 0.01,
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

def transform(data):
    n=data.shape[0]
    m=data.shape[1]
    # Step 1: 分箱为 8 类 (0-7)
    bins = np.linspace(0, 1, 9)  # 8 个分箱边界 (0 到 1 的等间隔划分)
    data_binned = np.digitize(data, bins=bins) - 1  # 分箱结果 [0-7]

    # Step 2: 将每个整数转换为 3 维二进制向量
    # 创建预定义的二进制编码映射 (0-7 → 3-bit 向量)
    binary_map = {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, 0],
        3: [0, 1, 1],
        4: [1, 0, 0],
        5: [1, 0, 1],
        6: [1, 1, 0],
        7: [1, 1, 1]
    }

    # Step 3: 将分箱数据转换为 (n, 16, 3)
    # 初始化结果数组
    binary_data = np.zeros((n, 16, 3), dtype=np.uint8)

    # 遍历并编码每个值
    for i in range(n):
        for j in range(m):
            value = data_binned[i, j]
            if value==8:
                value=7
            binary_data[i, j] = binary_map[value]
    return binary_data
if __name__ == "__main__":
    dataset_name = 'mnist4'
    from dataset import get_mnist_numpy

    train_datasets, val_datasets, test_datasets = get_mnist_numpy(dataset_name, 6)
    X_train, y_train = train_datasets
    X_test, y_test = test_datasets
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    X_train = transform(X_train)
    X_test = transform(X_test)


    model = SeparableVariationalClassifier(jit=False, max_vmap=1, n_layers=4, n_qubits=6, n_classes=4, batch_size=1)
    model.fit(X_train, y_train)
    train_predictions = np.array(model.predict(X_train))
    test_predictions = np.array(model.predict(X_test))
    train_acc = np.mean(train_predictions == y_train)
    print(f"Train Accuracy: {train_acc:.4f}")
    test_acc = np.mean(test_predictions == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    with open('mnist_end_all_expvals.txt', 'a', encoding='utf-8') as file:
        print('test_acc', dataset_name, test_acc, file=file)