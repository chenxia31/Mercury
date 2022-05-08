import os
import pandas as pd

import pyomo.environ as pyo



def read_data(path):
    '''
    解析Excel表格中的原始数据
    path:时刻表的路径


    :return:
    shift_time_upstream:上行班次的时间，[出发时间，到达时间]
    shift_time_downstream:下行班次的时间,[出发时间，到达时间]
    shift_index_upstream：上行班次的到达时间晚于下行班次的发车时间的班次对[i,j]
    shift_index_downstream:下行班次的到达时间晚于上行班次的发车时间的班次对[j,i]
    '''
    data_d = pd.read_excel(path+'/104路时刻表.xlsx', sheet_name='Sheet2')
    data_u = pd.read_excel(path+'/104路时刻表.xlsx', sheet_name='Sheet1')

    shift_time_upstream = [[0, 0]]
    for time_start in data_u['depart_time']:
        time_start = int(time_start)
        if 340 < time_start <= 6 * 60 + 10:
            shift_time_upstream.append([time_start, time_start + 50])
        elif time_start <= 6 * 60 + 10:
            shift_time_upstream.append([time_start, time_start + 55])
        elif time_start <= 60 * 7.5:
            shift_time_upstream.append([time_start, time_start + 65])
        elif time_start <= 60 * 7 + 50:
            shift_time_upstream.append([time_start, time_start + 60])
        elif time_start <= 60 * 8 + 24:
            shift_time_upstream.append([time_start, time_start + 55])
        elif time_start <= 60 * 14 + 45:
            shift_time_upstream.append([time_start, time_start + 50])
        elif time_start <= 60 * 15 + 45:
            shift_time_upstream.append([time_start, time_start + 55])
        elif time_start <= 60 * 17 + 24:
            shift_time_upstream.append([time_start, time_start + 60])
        else:
            shift_time_upstream.append([time_start, time_start + 50])
    shift_time_downstream = [[0, 0]]
    for time_start in data_d['depart_time']:
        time_start = int(time_start)
        if time_start > 380 and time_start <= 6 * 60 + 50:
            shift_time_downstream.append([time_start, time_start + 55])
        elif time_start <= 8 * 60 + 6:
            shift_time_downstream.append([time_start, time_start + 65])
        elif time_start <= 60 * 8.5:
            shift_time_downstream.append([time_start, time_start + 60])
        elif time_start <= 60 * 9:
            shift_time_downstream.append([time_start, time_start + 50])
        elif time_start <= 60 * 16:
            shift_time_downstream.append([time_start, time_start + 55])
        elif time_start <= 60 * 18 + 2:
            shift_time_downstream.append([time_start, time_start + 60])
        else:
            shift_time_downstream.append([time_start, time_start + 50])

    shift_index_upstream = []
    for i in range(len(shift_time_upstream)):
        # shift_index_upstream.append([0,i])
        shift_index_upstream.append([i, 999])
    for i, num in enumerate(shift_time_upstream):
        for j, num_j in enumerate(shift_time_downstream):
            if num_j[0] >= num[1] + 5:
                shift_index_upstream.append([i, j])

    shift_index_downstream = []
    for i in range(len(shift_time_downstream)):
        # shift_index_downstream.append([0,i])
        shift_index_downstream.append([i, 999])
    for i, num in enumerate(shift_time_downstream):
        for j, num_j in enumerate(shift_time_upstream):
            if num_j[0] >= num[1] + 5:
                shift_index_downstream.append([i, j])
    return shift_index_upstream, shift_index_downstream, shift_time_upstream, shift_time_downstream




def createParameter(fleet_size):
    '''
    promo建立模型时，所必须的参数

    input:
    fleet_size:车队规模上限，认为设置的超参数
    :return:
    :parameter[0]:station类型
    :parameter[1]:车辆上限
    :parameter[2]:可行上行班次对序号
    :parameter[3]:可行下行班次序号
    :parameter[4]:上行班次对时间
    :parameter[5[:下行班次对时间
    :parameter[6]:上行班次对序号
    :parameter[7]:下行班次对序号
    ;:parameter[8]:可以吃饭班次的序号

    '''
    # 车站的类型
    parameter={}
    parameter['station']= [0, 1]
    # 司机的序号，根据车辆数量生成
    parameter['driver'] = list(range(fleet_size))
    # 通过read_data()得到班次数据，上下行可能的班次对序号、上行班次的时间对、下行班次的时间对
    parameter['shift_index_upstream'] ,parameter['shift_index_downstream'] ,parameter['shift_time_upstream']\
        ,parameter['shift_time_downstream']  = read_data()
    # 上下行的班次序号
    parameter['driver']  = list(range(1, len(parameter['shift_time_upstream'])))
    parameter['driver'] = list(range(1, len(parameter['shift_time_downstream'])))
    # 在用餐时间之前的班次序号
    parameter['driver']  = [i for i in parameter['index_downstream'] if parameter['shift_time_downstream'][i][1] <= 13.5 * 60]

    return parameter


def createModel(parameter):
    '''
    在createParameter的基础上，建立模型

    :return:
    '''
    model=pyo.ConcreteModel()

    # 设置模型的变量
    model.alpha=pyo.Var(parameter['shift_index_upstream'],parameter['driver'],within=pyo.Binary,intialize=0)
    model.beta=pyo.Var(parameter['shift_index_downstream'],parameter['driver'],within=pyo.Binary,intialize=0)
    model.gamma=pyo.Var(parameter['index_downstream'],parameter['driver'],within=pyo.Binary,intialize=0)
    model.delta=pyo.Var(parameter['driver'],within=pyo.Binary,intialize=0)
    model.WK=pyo.Var(parameter['driver'],intialize=0)
    model.WU=pyo.Var([0],intialize=0)
    model.WL=pyo.Var([0],intialize=0)

    # 设置模型的约束
    def delta_def(model,driver):
        '''
        每个司机都会有一个上班的约束

        :param model: 默认第一个参数是model
        :param driver: 对于每个司机来说
        :return:
        '''
        M_temp=[i for i in parameter['shift_time_upstream'] if i[0]==0]
        N_temp=[j for j in parameter['shift_time_downstream'] if j[0]==0]
        expr=sum(model.alpha[i[0],i[1],driver] for i in M_temp)+sum(model.beta[j[0],j[1],driver] for j in N_temp)
        return expr==model.delta[driver]
    model.delta_def=pyo.Constraint(parameter['driver'],rule=delta_def)

    def flow_banlance_up(model,index_upstream,driver):
        '''
        针对每一个上行的班次驶入，都会有一个下行班次出来

        :param model:
        :param index_upstream:
        :param driver:
        :return:
        '''
        M_temp=[i for i in parameter['shift_index_downstram'] if i[1]==index_upstream]
        N_temp=[i for i in parameter['shift_index_upsteam'] if i[0]==index_upstream]
        expr=sum(model.beta[i[0],i[1],driver] for i in M_temp)-sum(model.alpha[i[0],i[1],driver] for i in N_temp)
        return expr==0
    model.flow_banlance_up=pyo.Constraint(parameter['index_upstream'],parameter['driver'],rule=flow_banlance_up)

    def flow_banlance_down(model,index_downstream,driver):
        '''
        针对每一个下行的班次始入，都会有一个下行班次出来

        :param model:
        :param index_downstream:
        :param driver:
        :return:
        '''
        M_temp = [i for i in parameter['shift_index_downstram'] if i[0] == index_downstream]
        N_temp = [i for i in parameter['shift_index_upsteam'] if i[1] == index_downstream]
        expr = sum(model.beta[i[0], i[1], driver] for i in M_temp) - sum(
            model.alpha[i[0], i[1], driver] for i in N_temp)
        return expr == 0
    model.flow_banlance_down=pyo.Constraint(parameter['index_downstram'],parameter['driver'],rule=flow_banlance_down)

    def task_down(model,index_downstream):
        M_temp=[i for i in parameter['shift_index_downstram'] if i[0]==index_downstream]
        expr=sum(model.beta[i[0],i[1],d] for i in M_temp for d in parameter['driver'])

    def task_downstream():
        pass

    def task_upstream():
        pass

# 未完待续～


    flow_banlance_up()

def analyse():
    '''
    对createModel()的模型的求解结果，进行分析
    :return:
    '''
    pass
