import sys
import math
import os
from stl_to_obj import *

path1 = "C:/Users/win10/Desktop/test/obj_semi_G_new_stl" # STL文件所在的文件夹
path = "C:/Users/win10/Desktop/test/obj_semi_G_new" #当前文件夹
file = os.listdir(path1) # 列出当前文件夹的所有文件
print(file)
num = len(file)
# print(file)
# 循环遍历每个文件
for f in file:
    obj_name = f.split(".")[0] + '.obj'
    print(path1 + '/' + f, path + 'OBJ/' + obj_name)
    stl_to_obj(path1 + '/' + f, path + '/OBJ/' + obj_name) #STL2OBJ
    print(obj_name)

print('Total {} STL2OBJ has been Finished !'.format(num))
# print(f'Total {num} STEP2OBJ has benn Finished !')
# print('Total %s STEP2OBJ has benn Finished !' % num)
# print('The num of STEP2OBJ is :', num )