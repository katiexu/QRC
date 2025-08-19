import heils_qrc_end_qubit
import heils_qrc_end_feature
import heils_qrc_end_all_features
from dataset import get_mnist_numpy
import numpy as np


for dataset_name in ['mnist01','mnist36','mnist4''mnist6','mnist8','mnist10']:
    train_datasets, val_datasets, test_datasets = get_mnist_numpy(dataset_name, 6)
    X_train, y_train = train_datasets
    X_test, y_test = test_datasets
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    for name,Model in {'qubit':heils_qrc_end_qubit.SeparableVariationalClassifier,
                  'feature': heils_qrc_end_feature.SeparableVariationalClassifier,
                  'features': heils_qrc_end_all_features.SeparableVariationalClassifier }.items():
        model = Model(jit=True, max_vmap=32, n_layers=4, n_qubits=6, n_classes=4)
        model.fit(X_train, y_train)
        train_predictions = np.array(model.predict(X_train))
        test_predictions = np.array(model.predict(X_test))
        train_acc = np.mean(train_predictions == y_train)
        test_acc = np.mean(test_predictions == y_test)
        print(dataset_name,name,train_acc,test_acc)