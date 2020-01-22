# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:37:43 2020

@author: Weberknecht2
"""

from __future__ import division
import numpy as np


"""
d_boxSize : the size of a small box with in the box
Tota_size : the size of the whole box.
"""

def check_range(pos, i, j, k,d_boxSize):
    a = 0
    if i <= pos[2] < i+d_boxSize:
            if j <= pos[3] < j+d_boxSize:
                    if k <= pos[4] < k+d_boxSize:
                            a = 1
    return a


def Count(data,Total_size,d_boxSize):

    N = []

    to_do = data

    print ('Counting number of particles per cell...')

    for k in range(0,Total_size,d_boxSize):
            for j in range(0,Total_size,d_boxSize):
                    for i in range(0,Total_size,d_boxSize):
                            temp = []
                            n = []
                            for count in range(len(to_do)):
                                    n.append(check_range(to_do[count],i,j,k,d_boxSize))
                                    to_do[count][1] = n[count]
                                    if to_do[count][1] == 0:
                                           temp.append(to_do[count])
                                    #Only particles that have not been found are
                                    # searched for again

                            to_do = temp
                            N.append(sum(n))
                    print ('Next row')
            print ('Next slice, %i still to find') % len(to_do)

   
    return N
   