#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import paddle 
from paddle.nn import Conv1D, MaxPool1D, Linear
import paddle.nn.functional as F
from paddle.vision.transforms import ToTensor


use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu') #打开GPU进行加速运算

"""
    s.t.
    W2-W1+100 <= 0
    100+W3-W2 <= 0
    100+W4-W3 <= 0
    W5-W4 <= 0
"""


# In[ ]:


#神经网络的相关参数
parameters_num=11
zuhe_numbers=15
Absorptance_num=3000
Absorptance_num_select=1000

max_file='D:\\自己的paper\\一作\\约稿-宽光谱吸收器优化设计\\data\\code\\results_2D\\model\\model_12\\max.txt'
min_file='D:\\自己的paper\\一作\\约稿-宽光谱吸收器优化设计\\data\\code\\results_2D\\model\\model_12\\min.txt'
avg_file='D:\\自己的paper\\一作\\约稿-宽光谱吸收器优化设计\\data\\code\\results_2D\\model\\model_12\\avg.txt'

max_values=np.loadtxt(max_file).reshape(parameters_num+zuhe_numbers+Absorptance_num,1)
min_values=np.loadtxt(min_file).reshape(parameters_num+zuhe_numbers+Absorptance_num,1)
avg_values=np.loadtxt(avg_file).reshape(parameters_num+zuhe_numbers+Absorptance_num,1)


#导入波长
wavelength_file='D:\\自己的paper\\一作\\约稿-宽光谱吸收器优化设计\\data\\code\\results_2D\\Results_2D\\wavelength.txt'
Wavelength=np.loadtxt(wavelength_file)
Wavelength_select=np.array(Wavelength[0:Absorptance_num:int(Absorptance_num/Absorptance_num_select)]).reshape(1,Absorptance_num_select)



#神经网络模型
class Regressor(paddle.nn.Layer):
    def __init__(self):
        #初始化父类中的参数
        super(Regressor,self).__init__()
        
        # 定义两层全连接隐含层,第一个维度输出的dim为parameters_num+15个组合
        self.linear_1 = paddle.nn.Linear(in_features=parameters_num+zuhe_numbers, out_features=3000) 
        self.linear_2 = paddle.nn.Linear(in_features=3000, out_features=10000) 
        self.linear_3 = paddle.nn.Linear(in_features=10000, out_features=20000)
        self.linear_4 = paddle.nn.Linear(in_features=20000, out_features=5000)
        # 定义一层全连接输出层，输出维度是Absorptance_num，其中只选Absorptance_num_select个点进行学习,不使用激活函数
        self.linear_5 = paddle.nn.Linear(in_features=5000, out_features=Absorptance_num_select)
        
    def forward(self, inputs):
        outputs_final=self.linear_1(inputs)
        outputs_final=F.relu(outputs_final)
        #outputs_final=self.dropout(outputs_final)
        outputs_final=self.linear_2(outputs_final)
        outputs_final=F.relu(outputs_final)
        #outputs_final=self.dropout(outputs_final)
        outputs_final=self.linear_3(outputs_final)
        outputs_final=F.relu(outputs_final)
        #outputs_final=F.dropout(outputs_final)
        outputs_final=self.linear_4(outputs_final)
        outputs_final=F.relu(outputs_final)
        outputs_final=self.linear_5(outputs_final)
        return outputs_final

# In[2]:

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 11  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [300, 300,300, 300, 300, 300, 1300, 1000, 700, 300, 200]  # 决策变量下界
        ub = [1000, 1000, 1000, 1000, 1000,1000, 1600, 1500, 1400, 1300, 1300]  # 决策变量上界
        lbin = [1, 1, 1,1,1,1,1,1,1,1,1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1,1,1,1,1,1,1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        model=Regressor()
        model_dict = paddle.load('D:\\自己的paper\\一作\\约稿-宽光谱吸收器优化设计\\data\\code\\results_2D\\model\\model_12\\inference_model_12')
        #model_dict = paddle.load('D:\\自己的paper\\一作\\约稿-宽光谱吸收器优化设计\\data\\code\\results_2D\\model\\model_21\\inference_model_21')

        model.load_dict(model_dict)
        model.eval()
        Vars = pop.Phen  # 得到决策变量矩阵
        size=Vars.shape[0]
        h_GaAs_1 = Vars[:, 0].reshape(size,1)
        h_GaAs_2 = Vars[:, 1].reshape(size,1)
        h_GaAs_3 = Vars[:, 2].reshape(size,1)
        h_GaAs_4 = Vars[:, 3].reshape(size,1)
        h_GaAs_5 = Vars[:, 4].reshape(size,1)
        h_GaAs_6 = Vars[:, 5].reshape(size,1)
        W1 = Vars[:,6].reshape(size,1)
        W2 = Vars[:,7].reshape(size,1)
        W3 = Vars[:,8].reshape(size,1)
        W4 = Vars[:,9].reshape(size,1)
        W5 = Vars[:,10].reshape(size,1)
        #print(h_GaAs_1)
        jifen_final=[]
        # for i in range(size):
        #     jifen=0
        #     # parameters_sitting = np.array([h_GaAs_1[i],h_GaAs_2[i],h_GaAs_3[i],h_GaAs_4[i],h_GaAs_5[i],h_GaAs_6[i],W1[i],W2[i],W3[i],W4[i],W5[i],h_GaAs_1[i]/W1[i],h_GaAs_2[i]/W2[i],h_GaAs_3[i]/W3[i],h_GaAs_4[i]/W4[i],h_GaAs_5[i]/W5[i],h_GaAs_6[i]/W5[i],h_GaAs_1[i]/h_GaAs_2[i],h_GaAs_2[i]/h_GaAs_3[i],h_GaAs_3[i]/h_GaAs_4[i],h_GaAs_4[i]/h_GaAs_5[i],h_GaAs_5[i]/h_GaAs_6[i],W1[i]/W2[i],W2[i]/W3[i],W3[i]/W4[i],W4[i]/W5[i]]).reshape(parameters_num+zuhe_numbers,1)
        #     parameters_sitting = np.array([h_GaAs_1[i],h_GaAs_2[i],h_GaAs_3[i],h_GaAs_4[i],h_GaAs_5[i],h_GaAs_6[i],W1[i],W2[i],W3[i],W4[i],W5[i]]).reshape(parameters_num,1)
        #     #parameters_sitting_select=np.array((parameters_sitting-avg_values[0:parameters_num+zuhe_numbers])/(max_values[0:parameters_num+zuhe_numbers]-min_values[0:parameters_num+zuhe_numbers])).astype('float32')
        #     parameters_sitting_select=np.array((parameters_sitting-avg_values[0:parameters_num])/(max_values[0:parameters_num]-min_values[0:parameters_num])).astype('float32')
        #     eval_data_select = paddle.to_tensor(parameters_sitting_select.T).astype('float32')
        #     characteristic_eval = model(eval_data_select)
        #     characteristic_eval = characteristic_eval[:].numpy()
        #     #characteristic_eval =((max_values[parameters_num+zuhe_numbers:parameters_num+zuhe_numbers+Absorptance_num:int(Absorptance_num/Absorptance_num_select)]-min_values[parameters_num+zuhe_numbers:parameters_num+zuhe_numbers+Absorptance_num:int(Absorptance_num/Absorptance_num_select)])*characteristic_eval[:].T+avg_values[parameters_num+zuhe_numbers:parameters_num+zuhe_numbers+Absorptance_num:int(Absorptance_num/Absorptance_num_select)])
        #     characteristic_eval =((max_values[parameters_num:parameters_num+Absorptance_num:int(Absorptance_num/Absorptance_num_select)]-min_values[parameters_num:parameters_num+Absorptance_num:int(Absorptance_num/Absorptance_num_select)])*characteristic_eval[:].T+avg_values[parameters_num:parameters_num+Absorptance_num:int(Absorptance_num/Absorptance_num_select)])
        #     Absorptance_eval_select=characteristic_eval.T
        #     #print(Absorptance_eval_select)
        #     for j in range(Wavelength_select.shape[1]-1):
        #         jifen += (Absorptance_eval_select[0,j]+Absorptance_eval_select[0,j+1])*(Wavelength_select[0,j+1]-Wavelength_select[0,j])/2
        #     jifen=jifen/(max(Wavelength)-min(Wavelength))
        #     jifen_final.append(jifen)
        # jifen_final=np.array(jifen_final)
        # jifen_final=jifen_final.reshape(size,1)
        # pop.ObjV = jifen_final  # 计算目标函数值，赋值给pop种群对象的ObjV属性
        
        
        #parameters_sitting = np.hstack((h_GaAs_1,h_GaAs_2,h_GaAs_3,h_GaAs_4,h_GaAs_5,h_GaAs_6,W1,W2,W3,W4,W5,h_GaAs_1/W1,h_GaAs_2/W2,h_GaAs_3/W3,h_GaAs_4/W4,h_GaAs_5/W5,h_GaAs_6/W1,h_GaAs_1/h_GaAs_2,h_GaAs_2/h_GaAs_3,h_GaAs_3/h_GaAs_4,h_GaAs_4/h_GaAs_5,h_GaAs_5/h_GaAs_6,W1/W2,W2/W3,W3/W4,W4/W5)).reshape(size,parameters_num+zuhe_numbers)
        parameters_sitting = np.hstack((h_GaAs_1*1e-9,h_GaAs_2*1e-9,h_GaAs_3*1e-9,h_GaAs_4*1e-9,h_GaAs_5*1e-9,h_GaAs_6*1e-9,W1*1e-9,W2*1e-9,W3*1e-9,W4*1e-9,W5*1e-9,h_GaAs_1/W1,h_GaAs_2/W2,h_GaAs_3/W3,h_GaAs_4/W4,h_GaAs_5/W5,h_GaAs_6/W1,h_GaAs_1/h_GaAs_2,h_GaAs_2/h_GaAs_3,h_GaAs_3/h_GaAs_4,h_GaAs_4/h_GaAs_5,h_GaAs_5/h_GaAs_6,W1/W2,W2/W3,W3/W4,W4/W5)).reshape(size,parameters_num+zuhe_numbers)
        parameters_sitting = parameters_sitting.T
        parameters_sitting_select=np.array((parameters_sitting-avg_values[0:parameters_num+zuhe_numbers])/(max_values[0:parameters_num+zuhe_numbers]-min_values[0:parameters_num+zuhe_numbers])).astype('float32')
        eval_data_select = paddle.to_tensor(parameters_sitting_select.T)   
        characteristic_eval = model(eval_data_select)
        characteristic_eval = characteristic_eval[:].numpy()
        characteristic_eval =((max_values[parameters_num+zuhe_numbers:parameters_num+zuhe_numbers+Absorptance_num:int(Absorptance_num/Absorptance_num_select)]-min_values[parameters_num+zuhe_numbers:parameters_num+zuhe_numbers+Absorptance_num:int(Absorptance_num/Absorptance_num_select)])*characteristic_eval[:].T+avg_values[parameters_num+zuhe_numbers:parameters_num+zuhe_numbers+Absorptance_num:int(Absorptance_num/Absorptance_num_select)])
        Absorptance_eval_select=np.array(characteristic_eval.T)
        #print(Absorptance_eval_select)
        for i in range(size):
            jifen=0
            #print(Absorptance_eval_select[i])
            for j in range(Wavelength_select.shape[1]-1):
                jifen += (Absorptance_eval_select[i,j]+Absorptance_eval_select[i,j+1])*(Wavelength_select[0,j+1]-Wavelength_select[0,j])/2
            jifen=jifen/(max(Wavelength)-min(Wavelength))
            jifen_final.append(jifen)
        jifen_final=np.array(jifen_final)
        jifen_final=jifen_final.reshape(size,1)
        pop.ObjV = jifen_final  # 计算目标函数值，赋值给pop种群对象的ObjV属性
        
        # 采用可行性法则处理约束
        pop.CV = np.hstack([W2-W1+100,100+W3-W2,100+W4-W3,W5-W4])
