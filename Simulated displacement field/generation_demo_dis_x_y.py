#从abaqus的odb文件中提取表面节点的位移场
# -*- coding: utf-8 -*-
from odbAccess import openOdb
import csv
my_odb = openOdb(r"Z:\YYL\abaqus\ABAQUS_compose\V2\Job-lashen1.odb")
step = my_odb.steps['Step-1'] #选择步数的名称
frame = step.frames[-1] #-1代表最后一帧
 #    PART1-1，SET-1
setname = 'SET-1'
NodeSet = my_odb.rootAssembly.instances['PART-1-1'].nodeSets[setname]
with open('dislashen1_x_y.csv','w') as csvfile:
    fieldnames = ['NodeLabel', 'UX', 'UY']#定义列表名,标签，X位移，Y位移
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)#用于将字典数据写入CSV文件，fieldnames参数指定了列名
    writer.writerow(dict((fn, fn) for fn in fieldnames))
    for node in NodeSet.nodes:
        u = frame.fieldOutputs['U'].getSubset(region=node).values[0]  # 获取节点位移
        writer.writerow({'NodeLabel': node.label, 'UX': u.data[0], 'UY': u.data[1]})  # 写入X和Y方向位移
