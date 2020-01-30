# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:55:21 2020

@author: Prava
"""


"""
Created on Wed Jan 22 15:37:43 2020
@author: Weberknecht2
"""


import numpy as np
import matplotlib.pyplot as plt    
from mpl_toolkits.mplot3d import Axes3D
"""
Function to read the data
"""

def Read(file_path,header,step_number):

    with open(file_path, "r") as f:
        a = f.read()
    step = a.split(header)[step_number + 1].strip()
    v_liste =[]
    for line in step.split("\n"):
        _, x, y, z = line.split(" ")
        v_liste.append(np.array([float(x),float(y),float(z)]))
       
    return np.array(v_liste)


if __name__ == '__main__':
    Z =Y= 5
    X = 50
    Data = Read('positions/positions.xyz','1000\n\n',1000)#To read the data
    #Data = init_pos(30, 125, scale=2/15)
    print(Data)
    b= (np.histogramdd(Data, bins=5)[0]).flat
    
    mask_zero = b != 0
    b = b[mask_zero]
    #len
    fig = plt.figure()
    m =np.histogram(b,bins=300)[0]
    plt.hist(b, bins=300)
    plt.xlabel(r'$\rho')
    plt.ylabel('Count (N)')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(Data[:, 0], Data[:, 1], Data[:, 2], marker=",")
    ax.set_xlim(-X, X)
    ax.set_ylim(-Y, Y)
    ax.set_zlim(-Z, Z)



