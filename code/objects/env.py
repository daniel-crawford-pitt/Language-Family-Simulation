import numpy as np
from numpy import random as r
import sys
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from matplotlib import colors
from concave_hull import concave_hull, concave_hull_indexes
from shapely import geometry
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import cdist
from copy import deepcopy


sys.path.insert(0, 'C:/Users\dcraw\OneDrive\Desktop\Language Family Simulation\code\objects')
from language import *

MAX_NUMBER_LANGUAGES = 25
NUM_INIT_LANGS = 5
FIELD_SIZE_TUPLE = (100,100)

class Env:
    def __init__(self):
        self.languages = [None]*MAX_NUMBER_LANGUAGES
        self.map = np.zeros([101,101])
        self.t = 0
        self.color_list = [
            'tab:blue','tab:orange','tab:green',
            'tab:red','tab:purple','tab:brown',
            'tab:pink','tab:gray','tab:olive','tab:cyan']*10
        self.hull_points = [None]*MAX_NUMBER_LANGUAGES

        self.curr_map = np.zeros((MAX_NUMBER_LANGUAGES,100,100))

        #self.competition_matrix = np.zeros((10,10))+0.5

        



    def sim(self):

        def sim_map(existance_map):

            existance_map_next = copy.deepcopy(existance_map)

            for i in range(1,existance_map.shape[0]):
                for j in range(1,existance_map.shape[1]):
                    existance_map_next[i][j] = next_exist_fxn(
                        existance_map[i][j],existance_map[i-1:i+2, j-1:j+2]
                        )

            return existance_map_next

        def onClick(event):
            global pause
            pause ^= True

        def update_sim(frame):

            if not pause:
                # Clear previous plot contents
                fig.clear()

                self.step()
                if self.t%100 == 0: print(self.t)

                ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan = 3)
                #ax2 = plt.subplot2grid((3, 4), (0,3), rowspan=1)
                #ax3 = plt.subplot2grid((3, 4), (1,3), rowspan=1)
                #ax4 = plt.subplot2grid((3, 4), (2,3), rowspan=1)

                p = np.zeros((MAX_NUMBER_LANGUAGES, 100,100)) #10 colors

                self.curr_map = p

                #print('len: ', len(self.languages))
                for i,l in enumerate(self.languages):
                    
                    #print(i)
                    #print('p-shape: ',p.shape)
                    if l is not None: 
                        

                
                #p[(p == [0, 0, 0]).all(axis=-1)] = [255,255,255]
                #p[(p == [1, 1, 1]).all(axis=-1)] = [0,0,0]
                        #_ = np.zeros((100,100))
                        #p[:,:,i] = _[np.where(l.map > 0)] = 1
                        p[i,:,:] = l.map
                        cmap = colors.ListedColormap(['white', self.color_list[i]])
                        bounds=[0,0.01,1]
                        norm = colors.BoundaryNorm(bounds, cmap.N)
                        ax1.imshow(p[i,:,:], alpha=0.75, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
                        ax1.set_xticks(np.arange(0, 100, 10))
                        ax1.set_yticks(np.arange(0, 100, 10))


                if True:
                    if len([l for l in self.languages if l is not None]) > 0:
                        for i in range(MAX_NUMBER_LANGUAGES):  # Assuming you have three color groups
                                indices = np.where((p[i,:, :] > 0)&(p[i, :, :] <= 1))  # Change the condition based on your data
                                x = indices[1]
                                y = indices[0]  # Invert y-axis if needed
                                if len(x) > 5:
                                    points = list(zip(x,y))
                                    idxes = concave_hull_indexes(points)
                                    hull_points = np.array([points[i] for i in idxes])
                                    hull_points = np.append(hull_points,hull_points[0]).reshape(-1,2)
                                    
                                    c = self.color_list[i]
                                    ax1.plot(hull_points[:,0], hull_points[:,1], color = c)
                                    #plt.scatter(hull_points[:,0], hull_points[:,1], marker = 'o',  edgecolors='r', color='none', lw=0.5)

                    """
                    for i, ax in enumerate([ax2,ax3,ax4]):
                        c = p[np.where((p[:, :, i] > 0)&(p[:, :, i] <= 1))][:,i]
                        ax.hist(c, 
                                bins = np.arange(0.05,1.06,0.1),
                                weights=np.ones(len(c)) / len(c),
                                color = ['r','g','b'][i]
                            )
                        ax.set_xlim(0,1.1)
                       
                         ax.set_ylim(0,1.1)
                    """

                plt.subplots_adjust(wspace = 0.4, hspace=0.4)




        

            return fig

        

        fig, ax = plt.subplots()
        #time_template = 'Time = %.1f s'
        #time_text = ax.text(0.05, 0.9, f'{self.t}', transform=ax.transAxes)


        #fig.canvas.mpl_connect('button_press_event', onClick)

    
        ani = animation.FuncAnimation(fig, update_sim,
            blit=False, interval=10, frames = 100, 
            cache_frame_data=False,
            repeat=True)

        return ani


    def step(self):
        if self.t == 0:
            self.do_start()
        self.do_moves()
        #self.do_births()
        #self.do_deaths()        
        self.do_splits()
        #self.do_competitions()


        self.t += 1

    def do_start(self):
        temp_map = np.zeros(FIELD_SIZE_TUPLE)
        points = np.argwhere(temp_map == 0)
        kmeans = MiniBatchKMeans(n_clusters=NUM_INIT_LANGS, init='random').fit(points)
        preds = kmeans.predict(points)      
        
        for i in np.unique(preds):

            new_map = np.zeros((100,100))
            for p in np.reshape(points[np.argwhere(preds == i)],(-1,2)):
                new_map[p[0],p[1]] = 1

            self.add_language(
                centroid = kmeans.cluster_centers_.astype(int),
                color = np.random.choice(self.color_list),
                start_time = 0,
                start_map = deepcopy(new_map)
            )
    
    def add_language(self, centroid, color, start_time, start_map):

        self.languages[self.languages.index(None)] = Language(
            centroid, color, start_time, start_map
        )

        
                        



    def do_moves(self):
        for i,l in enumerate(self.languages):
            if l is not None:
                other_ling_areas = np.zeros((100,100)).astype(bool)
                indices = np.where((l.map> 0)&(l.map <= 1))  # Change the condition based on your data
                x = indices[1]
                y = indices[0]  # Invert y-axis if needed
                if len(x) > 5:
                    points = list(zip(x,y))
                    idxes = concave_hull_indexes(points)
                    hull_points = np.array([points[i] for i in idxes])
                    hull_points = np.append(hull_points,hull_points[0]).reshape(-1,2)
                    
                    self.hull_points[i] = hull_points


                    #Find other patches:
                    for j in [k for k in list(np.arange(0,MAX_NUMBER_LANGUAGES,1)) if k != i]:
                        if self.hull_points[j] is not None and l.momentum <= self.languages[j].momentum:
                            
                            #print(l.momentum)
                            #print(self.languages[i].momentum)

                            other_ling_areas = \
                                other_ling_areas |\
                                np.array(Delaunay(self.hull_points[j]).find_simplex(
                                    [(ii,ji) for ii in np.arange(0,100,1) for ji in np.arange(0,100,1)]
                                )>=0).reshape(100,100).T
                    
                
                l.step(other_ling_areas)


    def do_births(self):
        if self.languages.count(None) > 5:
            for i,l in enumerate(self.languages):
                if l is None:
                    if np.random.rand() < 0.01:
                        
                        

                        self.languages[i] = Language(
                            (np.random.randint(0,100),np.random.randint(0,100)),
                            'Blues', self.t)


    def do_deaths(self):
        for i,l in enumerate(self.languages):
            if l is not None:
                l.death()
                if np.sum(l.map) < 1: self.languages[i] = None


    def do_splits(self):
        for i,l in enumerate(self.languages):
            if l is not None and None in self.languages:
                 if np.random.rand() < 0.01:

                    # Assuming you have a numpy array of points in the form [(x1, y1), (x2, y2), ...]
                    (As, Bs) = np.where(l.map > 0.001)
                    points = [[a,b] for a, b in zip(As, Bs)]     
                    if len(points) > 30:

                        kmeans = MiniBatchKMeans(n_clusters=2, init='random').fit(points)
                        preds = kmeans.predict(points)

                        #print(points)
                        #print(preds)

                        cluster1 = np.array([p for p, b in zip(points, preds) if b == 0])
                        cluster2 = np.array([p for p, b in zip(points, preds) if b == 1])
                        

                        #print(cluster1)
                        _ = np.zeros((100,100))
                        for p in cluster1: _[p[0],p[1]] = 1

                        l.map = deepcopy(_)

                        #At first blank space in language list
                        new_l_index = self.languages.index(None)
                        self.languages[new_l_index] = Language(
                            (np.random.randint(0,100),np.random.randint(0,100)),
                            'Blues', self.t)
                        _ = np.zeros((100,100))
                        for p in cluster2: _[p[0],p[1]] = 1
                        self.languages[new_l_index].map = deepcopy(_)

                        print("SPLIT HAPPENS")


                        
