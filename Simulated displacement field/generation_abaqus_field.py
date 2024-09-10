import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

#获取csv数据，返回列表
def dis_from_csv(csv_file_path):
    # CSV文件路径
    # 初始化坐标列表
    coordinates = []
    dis_u = []
    dis_v = []
    i = 0
    # 使用with语句打开文件，这样可以确保文件在操作完成后被正确关闭
    #'utf-8'是一种广泛使用的字符编码格式，能够支持几乎所有的Unicode字符。
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        # 创建csv阅读器,每一行数据转换成一个字典
        reader = csv.DictReader(file)
        #遍历每一行数据
        for row in reader:
            # 提取X坐标和Y坐标
            x = float(row['UX'])
            y = float(row['UY'])
            coordinates.append((x, y))
    # 打印坐标列表  744个坐标 31行 24列
    for coord in coordinates:
        dis_u.append(coord[0])
        dis_v.append(coord[1])
        i+=1
    return dis_u,dis_v

def abaques_dis(lst,columns=24,row=31):
    assert columns*row==len(lst)
    #使得列表反向
    lst.reverse()
    #创建了columns个空列表
    col=[[] for _ in range(columns)]
    for i in range(columns):
        start_index=i*row
        end_index=start_index+row
        lst1=lst[start_index:end_index]
        lst1.reverse()
        col[i]=lst1
    out=np.array(col)
    out_dis=out.T
    #左右翻转，axis=1表示按列的方向进行左右翻转
    # out_dis=np.flip(out_dis,axis=1)
    return  out_dis

#线性插值，将24*31大小的图片插值到与图像大小相同的尺寸512*512
def gen_dis_field(displacement):
    height,width=displacement.shape
    old_points= np.dstack(np.mgrid[0:height, 0:width]).reshape(-1, 2)
    target_shape=(690,512)
    # 创建目标网格坐标
    grid_x, grid_y = np.mgrid[0:width:complex(0, target_shape[1]),
                     0:height:complex(0, target_shape[0])]
    interpolated_displacement = griddata(old_points, displacement.flatten(), (grid_x, grid_y), method='cubic')
    return interpolated_displacement


#画图
def plot_image(image_r,image_rname,image_d,image_dname):
    image_r=np.array(image_r)
    image_d=np.array(image_d)
    plt.subplot(1,2,1)
    plt.imshow(image_r,cmap='jet')
    plt.title(image_rname)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(image_d,cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.title(image_dname)
    plt.show()

#裁剪->480*480
def crop_image(dis):
    width,height=dis.shape
    #裁剪大小crop_size
    crop_size=512
    left=0
    top=0
    right=crop_size
    bottom=crop_size
    out_dis=dis[left:right,top:bottom]
    return out_dis


dis_u,dis_v=dis_from_csv(csv_file_path= 'Z:\YYL\code\dis_x_y.csv')
dis_u_out=abaques_dis(dis_u)
dis_v_out=abaques_dis(dis_v)
fdis_u=gen_dis_field(dis_u_out)
fdis_v=gen_dis_field(dis_v_out)
dis_u_c=crop_image(fdis_u)
dis_v_c=crop_image(fdis_v)

dis_v_c=dis_v_c[...,np.newaxis]
dis_u_c=dis_u_c[...,np.newaxis]
dis_u_v = np.concatenate((dis_u_c,dis_v_c), axis=2)
print(dis_u_v.shape)
dis_save_path = f'./abaqus_field/dis_00001.npy'
np.save(dis_save_path,dis_u_v)

plot_image(dis_u_out,'u',dis_v_out,'v')
plot_image(fdis_u,'u1',fdis_v,'v')
plot_image(dis_u_c,'u',dis_v_c,'v')
