from sklearn.base import BaseEstimator, ClassifierMixin
import pennylane as qml

from get_data import get_senta_data
from model_utils import *
from sklearn.preprocessing import MinMaxScaler

jax.config.update("jax_enable_x64", True)

seq_length = 10
n_layers = 3
n_qubits = 4
max_steps = 10000

def ptrace(rho,N):
    reshaped_rho = rho.reshape([2, 2**(N-1), 2, 2**(N-1)])
    reduced_rho = jnp.einsum('ijik->jk', reshaped_rho,optimize=True)
    return reduced_rho
class SeparableVariationalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            encoding_layers=1,
            learning_rate=0.001,
            batch_size=32,
            max_vmap=None,
            jit=True,
            max_steps=max_steps,
            random_state=42,
            scaling=1.0,
            convergence_interval=200,
            qnode_kwargs={"interface": "jax"},
            n_qubits_=4,
            layers=5
    ):
        # attributes that do not depend on data
        self.encoding_layers = encoding_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.convergence_interval = convergence_interval
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits_ = n_qubits_
        self.n_layers_ = layers
        self.num_hidden = self.n_qubits_-1
        self.scaler = None  # data scaler will be fitted on training data
        self.circuit = None
        self.dev = qml.device("default.mixed", wires=self.n_qubits_)
        self.g = 0.5

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))
    def M_matrix(self):
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
        return jnp.kron(rho_reduced, rho_A)   #合并到最后一个qubit

    def partial_trace(self,rho, keep_wires):
        # rho: 当前密度矩阵（shape: 16x16）
        # keep_wires: 要保留的量子比特（如 [1,2,3]）

        # 计算要 trace out 的量子比特（这里是 wire=0）
        dim_keep = 2 ** len(keep_wires)
        dim_trace = 2 ** (4 - len(keep_wires))  # 4-qubit 系统

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
        proj_full = jnp.kron(proj, jnp.eye(8))

        # 坍缩后的密度矩阵: ρ_collapsed = (P ρ P†) / Tr(P ρ P†)
        rho_collapsed = proj_full @ rho @ proj_full.conj().T
        normalization = jnp.trace(rho_collapsed).real
        rho_collapsed /= normalization
        return rho_collapsed

    def construct_model(self):
        @qml.qnode(self.dev, **self.qnode_kwargs)
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
            # 初始记忆状态
            hidden_state = None  # 初始状态

            # 1. 执行量子电路，获取新 密度矩阵
            rho = single_qubit_circuit(hidden_state, params["weights"],
                                       self.n_qubits_, self.n_layers_)

            # 0. 应用 M 矩阵
            M = self.M_matrix()
            rho = M @ rho @ M.conj().T
            rho /= jnp.trace(rho).real  # 归一化

            # 2. 测量
            pauli_z = qml.PauliZ(0).matrix()
            pauli_z_full = jnp.kron(pauli_z, jnp.eye(2 ** (self.n_qubits_ - 1)))
            expval = jnp.trace(rho @ pauli_z_full).real

            input_seq = np.squeeze(input_seq)
            for i, x in enumerate(input_seq):

                # 1. 根据概率采样测量结果坍缩
                p0 = (1 + expval) / 2  # 计算测量到0的概率
                key, subkey = jax.random.split(self.generate_key())
                outcome = jax.random.bernoulli(subkey, p0).astype(int)

                # 2. 根据采样结果坍缩状态
                rho_collapsed = self.collapse_density_matrix(rho, outcome)

                # 3. 计算坍缩后 wires=[1,2,3] 的偏迹
                rho_reduced = self.partial_trace(rho_collapsed, keep_wires=[1, 2, 3])

                # 4. 合并x 生成新的 密度矩阵
                hidden_state = self.merge_x(x, rho_reduced)

                # 5. 执行量子电路，获取新 密度矩阵
                rho = single_qubit_circuit(hidden_state, params["weights"],
                                           self.n_qubits_, self.n_layers_)

                # 0. 应用 M 矩阵
                M = self.M_matrix()
                rho = M @ rho @ M.conj().T
                rho /= jnp.trace(rho).real  # 归一化

                # 6. 测量
                pauli_z = qml.PauliZ(0).matrix()
                pauli_z_full = jnp.kron(pauli_z, jnp.eye(2 ** (self.n_qubits_ - 1)))
                expval = jnp.trace(rho @ pauli_z_full).real
            return expval

        if self.jit:
            circuit = jax.jit(circuit)
        # self.forward = jax.vmap(circuit, in_axes=(None, 0))
        self.forward = circuit
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize(self, n_features):
        """Initialize attributes that depend on the number of features and the class labels.

        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels that the classifier expects
        """

        self.initialize_params()
        self.construct_model()

    def output_transform(self, params, outputs):
        # 只取最后一个时间步的输出
        last_output = outputs[-1]
        return last_output

    def initialize_params(self):
        self.params_ = {
            "weights": jax.random.uniform(self.generate_key(), (n_layers, n_qubits, 3), minval=0, maxval=2*jnp.pi),
            'scale_factor': jnp.array(0.5)  # 可训练的比例因子
        }


    def fit(self, X, y):
        self.initialize(X.shape[1])

        optimizer = optax.adam

        def loss_fn(params, X, y):
            # we multiply by 6 because a relevant domain of the sigmoid function is [-6,6]
            vals = self.forward(params, X)
            return jnp.mean(jnp.mean((vals - y) ** 2))

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
        return jnp.squeeze(predictions)

    def predict_proba(self, X):
        predictions = self.chunked_forward(self.params_, X)
        return predictions


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')

    # 生成时序数据
    data = get_senta_data()
    time_steps = np.linspace(0, 10, len(data))

    # 准备数据
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            targets.append(data[i + seq_length])
        return np.array(sequences), np.array(targets)


    X, y = create_sequences(data, seq_length)

    # 分割数据集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = SeparableVariationalClassifier(jit=False, max_vmap=1, layers=n_layers, n_qubits_=n_qubits)
    model.fit(X_train, y_train)
    train_predictions = np.array(model.predict(X_train))
    test_predictions = np.array(model.predict(X_test))

    # 可视化预测结果
    plt.figure(figsize=(14, 7))

    # 训练集预测
    plt.subplot(1, 2, 1)
    plt.plot(time_steps[seq_length:train_size + seq_length], y_train, label='Actual')
    plt.plot(time_steps[seq_length:train_size + seq_length], train_predictions, label='Predicted')
    plt.title('Training Set Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    # 测试集预测
    plt.subplot(1, 2, 2)
    plt.plot(time_steps[train_size + seq_length:], y_test, label='Actual')
    plt.plot(time_steps[train_size + seq_length:], test_predictions, label='Predicted')
    plt.title('Test Set Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()
