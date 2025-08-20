import numpy as np
def translator(single_code, enta_code, trainable, arch_code, fold=1):
    def gen_arch(change_code, base_code):  # start from 1, not 0
        # arch_code = base_code[1:] * base_code[0]
        n_qubits = base_code[0]
        arch_code = ([i for i in range(2, n_qubits + 1, 1)] + [1]) * base_code[1]
        if change_code != None:
            if type(change_code[0]) != type([]):
                change_code = [change_code]

            for i in range(len(change_code)):
                q = change_code[i][0]  # the qubit changed
                for id, t in enumerate(change_code[i][1:]):
                    arch_code[q - 1 + id * n_qubits] = t
        return arch_code

    def prune_single(change_code):
        single_dict = {}
        single_dict['current_qubit'] = []
        if change_code != None:
            if type(change_code[0]) != type([]):
                change_code = [change_code]
            length = len(change_code[0])
            change_code = np.array(change_code)
            change_qbit = change_code[:, 0] - 1
            change_code = change_code.reshape(-1, length)
            single_dict['current_qubit'] = change_qbit
            j = 0
            for i in change_qbit:
                single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1, 0)
                j += 1
        return single_dict

    def qubit_fold(jobs, phase, fold=1):
        if fold > 1:
            job_list = []
            for job in jobs:
                q = job[0]
                if phase == 0:
                    job_list.append([2 * q] + job[1:])
                    job_list.append([2 * q - 1] + job[1:])
                else:
                    job_1 = [2 * q]
                    job_2 = [2 * q - 1]
                    for k in job[1:]:
                        if q < k:
                            job_1.append(2 * k)
                            job_2.append(2 * k - 1)
                        elif q > k:
                            job_1.append(2 * k - 1)
                            job_2.append(2 * k)
                        else:
                            job_1.append(2 * q)
                            job_2.append(2 * q - 1)
                    job_list.append(job_1)
                    job_list.append(job_2)
        else:
            job_list = jobs
        return job_list
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code)

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # num of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits] - 1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits]) - 1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design