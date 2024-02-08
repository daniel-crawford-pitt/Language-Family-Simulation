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
import os

sys.path.insert(0, 'C:/Users\dcraw\OneDrive\Desktop\Language Family Simulation\code\objects')
from language import *
from parse_history import *
from object_utils import *

MAX_NUMBER_LANGUAGES = int(os.environ.get("MAX_NUMBER_LANGUAGES"))
NUM_INIT_LANGS = int(os.environ.get("NUM_INIT_LANGS"))

SHOW_CONCAVE_HULL = bool(os.environ.get("SHOW_CONCAVE_HULL"))
SHOW_TREE_DIAGRAM = bool(os.environ.get("SHOW_TREE_DIAGRAM"))

print(SHOW_TREE_DIAGRAM)

FIELD_SIZE_TUPLE = (100,100)#tuple(os.environ["FIELD_SIZE_TUPLE"])

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
                if self.t%100 == 0:
                    print(self.t)
                    for l in self.languages:
                        if l is not None: print(l.history)
                    for l in self.languages:
                        if l is not None: print(l.split_threshold)
                    



                ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan = 1)             
                
                p = np.zeros((MAX_NUMBER_LANGUAGES, 100,100)) #10 colors

                self.curr_map = p
                for i,l in enumerate(self.languages):
                    if l is not None: 
                        
                        p[i,:,:] = l.map
                        #cmap = colors.ListedColormap(['white', self.color_list[i]])
                        bounds=[0,0.01,1]
                        #norm = colors.BoundaryNorm(bounds, cmap.N)
                        ax1.imshow(p[i,:,:], interpolation='nearest', cmap=colors.ListedColormap([(0,0,0,0),l.color]), alpha = 0.9)
                        #ax1.imshow(p[i,:,:], alpha=0.75, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
                        ax1.set_xticks(np.arange(0, 100, 10))
                        ax1.set_yticks(np.arange(0, 100, 10))


                if SHOW_CONCAVE_HULL:
                    if len([l for l in self.languages if l is not None]) > 0:
                        for i in range(MAX_NUMBER_LANGUAGES):  
                                indices = np.where((p[i,:, :] > 0)&(p[i, :, :] <= 1))  # Change the condition based on your data
                                x = indices[1]
                                y = indices[0]  # Invert y-axis if needed
                                if len(x) > 5:
                                    points = list(zip(x,y))
                                    idxes = concave_hull_indexes(points)
                                    hull_points = np.array([points[i] for i in idxes])
                                    hull_points = np.append(hull_points,hull_points[0]).reshape(-1,2)
                                    
                                    c = self.languages[i].color
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
                if SHOW_TREE_DIAGRAM:
                    ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan = 1) 
                    
                    #We are getting the first entry in array being INITIAL - will need to work on this
                    
                    hist_scatter = history_figure([l.history for l in self.languages if l is not None], test_mode = False)
                
                    ax2.scatter(
                        hist_scatter['x'],
                        #hist_scatter['y'],
                        #[self.t]*len(hist_scatter['y']),
                        hist_scatter['t_start'],
                        c = [l.color for l in self.languages if l is not None]
                                )
                    
                    connectors = []
                    for n in preorder_iter(hist_scatter['tree']):
                        if n.name != 'INITIAL':
                            connectors = connectors + [((n.get_attr('x'),n.get_attr('t_start')),(c.get_attr('x'),c.get_attr('t_start'))) for c in n.children]

                    #print(connectors)
                    #print(np.array(connectors))
                    
                    if len(connectors) > 0:
                        for x,y in zip(np.array(connectors)[:,:,0].astype(float),np.array(connectors)[:,:,1].astype(float)):
                            ax2.plot(x,y,markerfacecolor = "black")
                    
                    for i, txt in enumerate(hist_scatter['labels']):
                        ax2.annotate(txt, (hist_scatter['x'][i], hist_scatter['t_start'][i]))
                    
                    ax2.invert_yaxis()
                    ax2.axes.get_xaxis().set_visible(False)

                plt.subplots_adjust(wspace = 0.4, hspace=0.4)




        

            return fig

        

        fig, ax = plt.subplots()
        #time_template = 'Time = %.1f s'
        #time_text = ax.text(0.05, 0.9, f'{self.t}', transform=ax.transAxes)


        #fig.canvas.mpl_connect('button_press_event', onClick)

    
        ani = animation.FuncAnimation(fig, update_sim,
            blit=False, interval=100, frames = 100, 
            cache_frame_data=False,
            repeat=True)

        return ani


    def step(self):
        if self.t == 0:
            self.do_start()
        else:
            if not bool(os.environ.get("MOMENTUM_FUNC_STATIC_BOOL")): self.do_update_momentums()
        self.do_moves()
        #self.do_births()
        self.do_deaths()        
        self.do_splits()
        #self.do_competitions()


        self.t += 1

    def do_update_momentums(self):
        for l in self.languages:
            if l is not None:
                l.update_momentum(self.t)

    def do_update_momentums(self):
        for l in self.languages:
            if l is not None:
                l.update_split_threshold()


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
                color = np.random.choice(self.color_list, replace=False),
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
                        if self.hull_points[j] is not None and self.languages[j] is not None and l.momentum <= self.languages[j].momentum:
                            
                            #print(l.momentum)
                            #print(self.languages[i].momentum)

                            try:
                                other_ling_areas = \
                                    other_ling_areas |\
                                    np.array(Delaunay(self.hull_points[j]).find_simplex(
                                        [(ii,ji) for ii in np.arange(0,100,1) for ji in np.arange(0,100,1)]
                                    )>=0).reshape(100,100).T
                            except:
                                pass
                    
                
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
                #l.death()
                if (l.map == 0).all(): 
                    self.languages[i] = None
                    print('DEATH HAPPENS')


    def do_splits(self):
        for i,l in enumerate(self.languages):
            if l is not None and None in self.languages:
                 if np.random.rand() < l.split_threshold:

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
                        _ = np.zeros((100,100))
                        for p in cluster2: _[p[0],p[1]] = 1
                        #self.languages[new_l_index].map = 

                        self.languages[self.languages.index(None)] = Language(
                            #Pick random color - None, np.random.choice(self.color_list), self.t, deepcopy(_), prev_history=l.history
                            #Pick color based on parent
                            None, random_color_near(l.color), self.t, deepcopy(_), prev_history=l.history
                        )

                        print("SPLIT HAPPENS")
                        


                        
