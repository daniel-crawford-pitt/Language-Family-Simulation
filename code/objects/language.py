import numpy as np
import random as r
import os
from scipy.ndimage import binary_dilation
from simulation.sim_main import *


class Language:
    """ 
    Language object - carries information about the language 
    """
    def __init__(self, starting_point, color, start_time, start_map = None,
                 SPLIT_THRESHOLD_FUNC_CLASS = None, MOMENTUM_FUNC_CLASS = None,
                 SPLIT_THRESHOLD_CONST_VALUE = None,  MOMENTUM_FUNC_STATIC_BOOL = None,
                prev_history = ""):
        """ 
        starting_point: tuple(int, int) - the point that the language originated - NOT CURRENTLY USED
        color: string - the color that the language should appear in the animation
        start_time: int - the time in simulation that the language began
        start_map: np.array((FIELD_SIZE_TUPLE,FIELD_SIZE_TUPLE)) - the initial map
        prev_history: string - the string the contains info on previous history. See parse_history.py for more details
        
        NOTES
        self.map is the map of the language. It is a np.array of size (FIELD_SIZE_TUPLE,FIELD_SIZE_TUPLE) eg (100,100).
        Currently it is a binary array: 0 for where the language is not present, 1 for where it is. An extension is to 
            have these values be more continuous.
        
        """

        self.starting_point = starting_point #think about this
        self.color = color
        self.start_time = start_time

        self.SPLIT_THRESHOLD_FUNC_CLASS = SPLIT_THRESHOLD_FUNC_CLASS
        self.MOMENTUM_FUNC_CLASS = MOMENTUM_FUNC_CLASS
        self.SPLIT_THRESHOLD_CONST_VALUE = SPLIT_THRESHOLD_CONST_VALUE
        self.MOMENTUM_FUNC_STATIC_BOOL = MOMENTUM_FUNC_STATIC_BOOL
        

        self.alive = True

        if start_map is None:
            self.map = np.zeros([100,100]) #Need to generalize with config
            self.map[starting_point[0]][starting_point[1]] = 1 #not used when initiating at a point
        else:
            self.map = start_map

        #concatenation of history string, see parse_history.py for more info
        self.history = prev_history + "START_" + str(start_time) + '_L' + str(os.environ.get('LANG_ID_CTR')).zfill(3) + '+'
        
        #increment the environmental variable for language counters by 1
        os.environ['LANG_ID_CTR'] = str(int(os.environ.get('LANG_ID_CTR'))+1)
        

        #Initialize a momentum value
        match MOMENTUM_FUNC_CLASS: #function class is set in config, different values depending on func class
            case 'SIN': #sine function
                self.A = 20*np.random.random()-10 #amplitude
                self.p = 20*np.random.random()-10 #period
                self.momentum = self.A * np.sin(self.p*self.start_time) #calculate - need t 
            case 'CONSTANT': #constant
                self.m = np.random.random() #select randomly
                self.momentum = self.m
            
        #Initialize Split Threshold
        match SPLIT_THRESHOLD_FUNC_CLASS: #function class is set in config, different values depending on func class
            case 'SIZE_INVERSE': #sine function
                #self.split_threshold = np.sum(self.map)/10000.0*0.01
                self.split_threshold = (np.exp(np.log(2)/10000.0*np.sum(self.map))-1)*0.1
            case 'CONSTANT': #constant
                self.split_threshold = self.SPLIT_THRESHOLD_CONST_VALUE
            
    def append_split_history(self, start_time):
        #History tag added to language when another splits form it
        self.history = self.history + "START_" + str(start_time) + '_L' + str(os.environ.get('LANG_ID_CTR')).zfill(3) + '+'
        os.environ['LANG_ID_CTR'] = str(int(os.environ.get('LANG_ID_CTR'))+1)
        
    
    def update_momentum(self, t): 
        """ 
        Upadate the momentum
        
        t: int - timestep
        """
        match self.MOMENTUM_FUNC_CLASS: #determine momentum function class from environment
            case 'SIN':
                self.momentum = self.A * np.sin(self.p*t)
            case 'CONSTANT':
                self.momentum = self.m

    def update_split_threshold(self): 
        match self.SPLIT_THRESHOLD_FUNC_CLASS: #function class is set in config, different values depending on func class
            case 'SIZE_INVERSE': #sine function
                #self.split_threshold = np.sum(self.map)/10000.0*0.01
                self.split_threshold = (np.exp(np.log(2)/10000.0*np.sum(self.map))-1)*0.1
            case 'CONSTANT': #constant
                self.split_threshold = self.SPLIT_THRESHOLD_CONST_VALUE
            

    
    def step(self, other_ling_area):
        """ The function that has a language take a step (spread) around the map
        
        other_ling_area: np.array((MAX_NUM_LANGUAGES-1, FIELD_SIZE_TUPLE, FIELD_SIZE_TUPLE)) -
            an array that contains the maps of there the other languages are. Currently we do not allow for 
            languages to overlap (more than one occupy an area) 
        
        """

        #map gets
        self.map = (\
            #current map (where the language is now)
            (self.map.astype(bool) | \
            #or, dialate (expand one step in all directions)
            binary_dilation(
                #from current location AND where random number (0,1) is greater than 0.2
                self.map.astype(bool) & (np.random.random(self.map.shape) < 0.2)
            ).astype(bool))
        ).astype(int)

        #rewrite the map to be only where other languages are not.
        self.map = (self.map.astype(bool) & np.invert(other_ling_area)).astype(int)

        #self.direct()
    
    #used to artificially control direction, not in use.
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

    #reset momentum if needed
    def reset_momentum(self):
        self.momentum = np.random.random()

    #death function - not used
    def death(self):
        threshold = 0.4
        #if random number under threshold, remove language from that point
        self.map[np.random.random((100, 100)) < threshold] = 0


        
        


