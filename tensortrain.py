import numpy as np
import scipy.linalg


# TT-SVD

class tt(object):

    def __init__(self, data=np.array([])):

        shape_list = list(data.shape)

        rank_list = []
        core_list = []

        totallen = 1
        for i in range(0, len(shape_list)):
            totallen *= shape_list[i]

        r = 1
        rank_list.append(r)

        for i in range(0, len(shape_list) - 1):
            t = r
            m = shape_list[i] * r
            n = totallen // m
            r = min(m, n)
            rank_list.append(r)
            totallen = n * r

            data_matrix = np.reshape(data, [m, n])

            [U, S, V] = np.linalg.svd(data_matrix, full_matrices=False)

            # U, S, V = truncateFuncion(U, S, V, r)

            S = np.diag(S)

            data = np.dot(S, V)

            core_list.append(np.reshape(U, [t, shape_list[i], r]))

        core_list.append(np.reshape(data, [r, shape_list[-1], 1]))
        rank_list.append(1)

        self.core_list = core_list
        self.rank_list = rank_list

    def setCoreList(self, core_list):
        self.core_list = core_list

    def ttconTraction(self):
        core_list = self.core_list
        core_sum = np.reshape(core_list[0], [core_list[0].shape[1], core_list[0].shape[2]])

        for i in range(0, len(core_list) - 2):
            core_sum = np.einsum('i...k, kmj->i...mj', core_sum, core_list[i + 1])

        core_lastone = np.reshape(core_list[-1], [core_list[-1].shape[0], core_list[-1].shape[1]])

        core_sum = np.einsum('i...k, km->i...m', core_sum, core_lastone)

        self.core_sum = core_sum

    def truncateFuncion(self, U, S, V, k):
        U = U[:, 0:k]

        S = S[:k]

        V = V[:k]

        return U, S, V

    def addTensorTrain(self, other):
        if len(other.core_list) != len(self.core_list):
            print("Add operation must have same size")

        new_core_list = []

        first_core = np.array([np.concatenate((self.core_list[0][0], other.core_list[0][0]), axis=1)])
        new_core_list.append(first_core)

        # 第一个core和最后一个core要特殊处理
        for i in range(1, len(self.core_list) - 1):

            rleft = self.core_list[i].shape[0] + other.core_list[i].shape[0]
            nmid = self.core_list[i].shape[1]
            rright = self.core_list[i].shape[2] + other.core_list[i].shape[2]

            new_core = np.zeros((rleft, nmid, rright))

            for j in range(0, self.core_list[i].shape[1]):
                a = self.core_list[i][:, j, :]
                b = other.core_list[i][:, j, :]
                c = scipy.linalg.block_diag(a, b)
                new_core[:, j, :] = c

            new_core_list.append(new_core)

        last_core = np.concatenate((self.core_list[-1], other.core_list[-1]), axis=0)
        new_core_list.append(last_core)
        return new_core_list

    def ttRounding(self):
        for i in range(len(self.core_list) - 1, 0, -1):
            r1, n, r2 = self.core_list[i].shape
            self.core_list[i], R = np.linalg.qr(np.reshape(self.core_list[i], [r1, n * r2]).T)
            r1 = self.core_list[i].shape[1]

            self.core_list[i] = np.reshape(self.core_list[i].T, [r1, n, r2])
            self.core_list[i - 1] = np.tensordot(self.core_list[i - 1], R.T, axes=1)

        r = 1
        for i in range(len(self.core_list) - 2):
            r1, n, r2 = self.core_list[i].shape
            self.core_list[i], sigma, vt = np.linalg.svd(np.reshape(self.core_list[i], [r * n, r2 * r1]),
                                                         full_matrices=False)

            s = np.diag(sigma)
            self.core_list[i + 1] = np.tensordot((s @ vt).T, self.core_list[i + 1], axes=([0], [0]))
            self.core_list[i] = np.reshape(self.core_list[i], [r1, n, self.core_list[i].shape[1]])
