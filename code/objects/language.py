import numpy as np
import random as r
from scipy.ndimage import binary_dilation

from simulation.sim_main import *


class Language:
    def __init__(self, starting_point, color, start_time, start_map = None):
        self.starting_point = starting_point #think about this
        self.color = color
        self.start_time = start_time

        if start_map is None:
            self.map = np.zeros([100,100])
            self.map[starting_point[0]][starting_point[1]] = 1
        else:
            self.map = start_map

        self.directionalization = np.random.random(4)

        self.momentum = np.random.random()


    def step(self, other_ling_area):
        self.map = (\
            (self.map.astype(bool) | \
            binary_dilation(
                self.map.astype(bool) & (np.random.random(self.map.shape) < 0.2)
            ).astype(bool))
        ).astype(int)

        self.map = (self.map.astype(bool) & np.invert(other_ling_area)).astype(int)

        #self.direct()
    
    def direct(self):
        #down
        if 1 not in np.where(self.map > 0)[0]:
            if self.directionalization[0] > 0:
                self.directionalization[0] = self.directionalization[0]-0.01

                self.map = np.roll(self.map, -1, 0)
                        

        #right
        if 99 not in np.where(self.map > 0)[1]:
            if self.directionalization[1] > 0:
                self.directionalization[1] = self.directionalization[1]-0.01

                self.map = np.roll(self.map, 1, 1)  
            
            
        #up
        if 99 not in np.where(self.map > 0)[0]:
            if self.directionalization[2] > 0:
                self.directionalization[2] = self.directionalization[2]-0.01

                self.map = np.roll(self.map, 1, 0)  

        #left    
        if 0 not in np.where(self.map > 0)[1]:
            if self.directionalization[3] > 0:
                self.directionalization[3] = self.directionalization[3]-0.01

                self.map = np.roll(self.map, -1, 1)  

        if (self.directionalization < 0).all():
            self.directionalization = np.random.random(4)

    def reset_momentum(self):
        self.momentum = np.random.random()

    def death(self):
        threshold = 0.4
        self.map[np.random.random((100, 100)) < threshold] = 0


        
        


