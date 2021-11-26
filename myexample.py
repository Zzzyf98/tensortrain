from tensortrain import tt
import scipy.linalg
import numpy as np

# 需要用np.array初始化对象
data1 = np.array([[[0, 0], [1, 1]], [[2, 2], [3, 3]]])
data2 = np.array([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]])
data3 = np.array([[[4, 4, 4], [5, 5, 5]], [[6, 6, 6], [7, 7, 7]]])

tt_test2 = tt(data2)
tt_test3 = tt(data3)

# 张量链加法
res = tt_test2.addTensorTrain(tt_test3)


print("第一条张量链：",tt_test2.core_list[0].shape, tt_test2.core_list[1].shape, tt_test2.core_list[2].shape)

print("第二条张量链：",tt_test3.core_list[0].shape, tt_test3.core_list[1].shape, tt_test3.core_list[2].shape)

print("加完后的张量链：",res[0].shape, res[1].shape, res[2].shape)


tx = tt()
#用加完后的core_list初始化对象,
tx.setCoreList(res)

tx.ttRounding()

print("Rounding后的张量链：",tx.core_list[0].shape, tx.core_list[1].shape, tx.core_list[2].shape)
#复原原张量
tx.ttconTraction()

print("从张量链复原的张量：",tx.core_sum)
