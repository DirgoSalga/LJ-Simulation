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
        _, x, y, z = line.split("\t")
        v_liste.append(np.array([float(x),float(y),float(z)]))
       
    return np.array(v_liste)

"""
d_boxSize : the size of a small box with in the box
Tota_size : the size of the whole box.
Data: informantion about the position
"""

def init_pos(box_size, num_particles, scale=1.):
    """
    Initialisation of the position of N particles in a box with a given side length in 3 dimensions. The particles are
    positioned on a 3D-lattice with equidistant points.
    :param box_size:  <int> box size length
    :param num_particles: <int> number of particles (must be a perfect cube)
    :param scale: <float> scale of how to shrink the distance between the partciles. This always remain centred. The
    default is set to 1, which means the particles spread from one end of the box to the other.
    :return: r: <array> of 3D positions of each particle.
    """
    n = round(num_particles ** (1 / 3))
    if n ** 3 != num_particles:
        raise NotCubicNumber(
            "Number of particles N is not a perfect cube and cannot be used in this implementation.")
    one_direction = np.linspace(box_size / 2 * -1, box_size / 2, n)
    one_direction *= scale
    positions = []
    for i in one_direction:
        for j in one_direction:
            for k in one_direction:
                positions.append(np.array([i, j, k]))

    return np.array(positions)


def check_range(pos, i, j, k,d_boxSize):
    a = 0
    if i <= pos[0] < i+d_boxSize:
            if j <= pos[1] < j+d_boxSize:
                    if k <= pos[2] < k+d_boxSize:
                            a = 1
    return a


def count(data,Total_size,d_boxSize):

    N = []

    to_do = data

    print ('Counting number of particles per cell...')

    for k in range(-Total_size,Total_size+d_boxSize,d_boxSize):
            for j in range(-Total_size,Total_size+d_boxSize,d_boxSize):
                    for i in range(-Total_size,Total_size+d_boxSize,d_boxSize):
                            #print(k,j,i)
                            n = []
                            for count in range(len(to_do)):
                                    n.append(check_range(to_do[count],i,j,k,d_boxSize))                            
                            N.append(sum(n))
    
    return N



def main():
    Total_Size = 30
    d_boxSize =5
    #Data = Read('position.xyz','125\n\n',0)#To read the data
    Data = init_pos(30, 125, scale=1.)
    Count = count(Data,Total_Size,d_boxSize)
    print('Toatl number of particles:',sum(Count))
    #plt.hist(Count, bins=4)
    fig = plt.figure()
    x = np.linspace(-Total_Size,Total_Size,len(Count))
    plt.plot(x, Count)
    plt.xlabel("Area??")
    plt.ylabel("Count")
    #plt.legend(loc=0, fancybox=True)
    #fig.savefig(".pdf")
    #only to check
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(Data[:, 0], Data[:, 1], Data[:, 2], marker=",")
    ax.set_xlim(-Total_Size, Total_Size)
    ax.set_ylim(-Total_Size, Total_Size)
    ax.set_zlim(-Total_Size, Total_Size)

if __name__ == '__main__':
    main()
#for m in range(-10,10+5,5):
    #print(m)


