#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 23:15:18 2018

@author: labadmin
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:19:27 2018

@author: labadmin
"""

import numpy as np
from matplotlib import pyplot as plt
import copy

def Get_edge(img):
    new=copy.deepcopy(img)
    row,col=img.shape
    for i in range(row-1):
        for j in range(col-1):
            if i-1>0 & i+1<row-1 & j-1>0 & j+1<col-1:
                if img[i][j-1]&img[i][j+1]&img[i-1][j]&img[i+1][j]:
                    new[i][j]=0
#                    new[i][j-1]=0
#                    new[i][j+1]=0
#                    new[i-1][j]=0
#                    new[i+1][j]=0


    return new
#####获得轮廓的点的位置
def Get_Edge_position(img):
    row,col=img.shape
    row-=1
    col-=1
    new=copy.deepcopy(img)
    x=[]
    y=[]
    while (new==0).all()==False:   
        for i in range(row):
            for j in range(col):
                    if new[i][j]:
                        start_x=i
                        start_y=j
                        x.append(start_x)
                        y.append(start_y)
                        new[start_x][start_y]=0
                        break
    
         #根据4联通领域依次进行寻找
        flage=1
        while(flage):
    
            if new[start_x][start_y-1]:
                start_x=start_x
                start_y=start_y-1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y]=0
    #            flage=1
            elif new[start_x+1][start_y-1]:
                start_x=start_x+1
                start_y=start_y-1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y]=0
            elif new[start_x+1][start_y]:
                start_x=start_x+1
                start_y=start_y
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y]=0
            elif new[start_x+1][start_y+1]:
                start_x=start_x+1
                start_y=start_y+1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y]=0
            elif new[start_x-1][start_y-1]:
                start_x=start_x-1
                start_y=start_y-1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y]=0
            elif new[start_x-1][start_y]:
                start_x=start_x-1
                start_y=start_y
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y]=0
            elif new[start_x-1][start_y+1]:
                start_x=start_x-1
                start_y=start_y+1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y]=0
            elif new[start_x][start_y+1]:
                start_y=start_y+1
                start_x=start_x
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y]=0
    #            flage=1
            else:
                    flage=0
    x=np.array(x)
    y=np.array(y)
    return x,y



def Distance_Map(array):
    '''
    for non-prostate_image_huge value be set
    '''
    chan, col, row = np.shape(array)
    DisMap = np.zeros_like(array)
    TempMap = np.zeros_like(array[0])
    TempMap[TempMap==0]=1000
    for i in range(chan):
        img =  array[i]
        if np.sum(img):
            img=Get_edge(img.astype('uint8'))
            edgex,edgey = Get_Edge_position(img)
            for c in range(col):
                for r in range(row):
                    DisMap[i,c,r]=np.min(np.sqrt(np.square(edgex-c) + np.square(edgey-r)))
         
        else:
            DisMap[i] = TempMap
    return DisMap



if __name__ == '__main__':
    
    data_path='TrainAndTest/Resample_0.625_0.625_1.5/'
    for i in range(50):
        GT = np.load(data_path+'GT%d.npy'%i)
        print(GT.shape)
        DM = Distance_Map(GT)
        print(DM.shape)
        path='TrainAndTest/Resample_0.625_0.625_1.5/'
        np.save(path+'DSM%d.npy'%i,DM)        
        
#        xx = DM
#        plt.figure(str(10)+'Y')
#        plt.imshow(xx[10])     
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
