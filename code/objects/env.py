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


class Env:
    def __init__(self, config):


        self.MAX_NUMBER_LANGUAGES = config["MAX_NUMBER_LANGUAGES"]

        self.FIELD_SIZE_1DIM = config["FIELD_SIZE_1DIM"]
        self.FIELD_SIZE_TUPLE = (config['FIELD_SIZE_1DIM'],config['FIELD_SIZE_1DIM'])
        self.PRINT_t_EVERY = float(config["PRINT_t_EVERY"])

        self.NUM_INIT_LANGS = config["NUM_INIT_LANGS"]

        self.SHOW_CONCAVE_HULL = config["SHOW_CONCAVE_HULL"]
        self.SHOW_TREE_DIAGRAM = config["SHOW_TREE_DIAGRAM"]
        self.SHOW_MAP = config["SHOW_MAP"]
        self.SHOW_NOTHING = config["SHOW_NOTHING"]

        self.MAX_TIME_STEPS = config["MAX_TIME_STEPS"]

        self.SPLIT_THRESHOLD_FUNC_CLASS = config["SPLIT_THRESHOLD_FUNC_CLASS"]
        self.MOMENTUM_FUNC_CLASS = config["MOMENTUM_FUNC_CLASS"]
        self.MOMENTUM_FUNC_STATIC_BOOL = config["MOMENTUM_FUNC_STATIC_BOOL"]
        self.SPLIT_THRESHOLD_CONST_VALUE = config["SPLIT_THRESHOLD_CONST_VALUE"]
        



        self.languages = [None]*self.MAX_NUMBER_LANGUAGES
        self.map = np.zeros([self.FIELD_SIZE_1DIM+1,self.FIELD_SIZE_1DIM+1])
        self.t = 0
        self.color_list = [
            'tab:blue','tab:orange','tab:green',
            'tab:red','tab:purple','tab:brown',
            'tab:pink','tab:gray','tab:olive','tab:cyan']*10
        self.hull_points = [None]*self.MAX_NUMBER_LANGUAGES

        self.curr_map = np.zeros((self.MAX_NUMBER_LANGUAGES,self.FIELD_SIZE_1DIM,self.FIELD_SIZE_1DIM))

        #self.competition_matrix = np.zeros((10,10))+0.5


    def sim(self, config):

        

        if not self.SHOW_NOTHING:

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
                    if self.t%self.PRINT_t_EVERY == 0:
                        print(f"Time: {self.t-1}")
                        for l in self.languages:
                            if l is not None: print(l.history)
                        for l in self.languages:
                            pass
                            #if l is not None: print(l.split_threshold)

                    if self.SHOW_MAP:
                        ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan = 1)             
                        
                        p = np.zeros((self.MAX_NUMBER_LANGUAGES, self.FIELD_SIZE_1DIM, self.FIELD_SIZE_1DIM)) #10 colors

                        self.curr_map = p
                        for i,l in enumerate(self.languages):
                            if l is not None and l.alive: 
                                
                                p[i,:,:] = l.map
                                #cmap = colors.ListedColormap(['white', self.color_list[i]])
                                bounds=[0,0.01,1]
                                #norm = colors.BoundaryNorm(bounds, cmap.N)
                                ax1.imshow(p[i,:,:], interpolation='nearest', cmap=colors.ListedColormap([(0,0,0,0),l.color]), alpha = 0.9)
                                #ax1.imshow(p[i,:,:], alpha=0.75, interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
                                ax1.set_xticks(np.arange(0, self.FIELD_SIZE_1DIM, self.FIELD_SIZE_1DIM/10))
                                ax1.set_yticks(np.arange(0, self.FIELD_SIZE_1DIM, self.FIELD_SIZE_1DIM/10))


                    if self.SHOW_CONCAVE_HULL:
                        if len([l for l in self.languages if l is not None and l.alive]) > 0:
                            for i in range(self.MAX_NUMBER_LANGUAGES):  
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

                    
                    if self.SHOW_TREE_DIAGRAM:
                        ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan = 1) 
                        
                        #We are getting the first entry in array being INITIAL - will need to work on this
                        
                        hist_scatter = history_figure([l.history for l in self.languages if l is not None and l.alive], test_mode = False)
                    
                        ax2.scatter(
                            hist_scatter['x'],
                            #hist_scatter['y'],
                            #[self.t]*len(hist_scatter['y']),
                            hist_scatter['t_start'],
                            c = [l.color for l in self.languages if l is not None and l.alive]
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
                interval=1, frames = self.MAX_TIME_STEPS, repeat=False)
            
            
            
            return ani

        else:
            for i in range(self.MAX_TIME_STEPS):
                self.step()

                if i%self.PRINT_t_EVERY == 0:
                    print(f"Time: {self.t-1}")
                    for l in self.languages:
                        pass
                        #if l is not None: print(l.history)
                    for l in self.languages:
                        pass

            

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
            if l is not None and l.alive:
                l.update_momentum(self.t)

    def do_update_momentums(self):
        for l in self.languages:
            if l is not None and l.alive:
                l.update_split_threshold()


    def do_start(self):
        temp_map = np.zeros(self.FIELD_SIZE_TUPLE)
        points = np.argwhere(temp_map == 0)
        kmeans = MiniBatchKMeans(n_clusters=self.NUM_INIT_LANGS, init='random', n_init=self.NUM_INIT_LANGS).fit(points)
        preds = kmeans.predict(points)      
        
        for i in np.unique(preds):

            new_map = np.zeros((self.FIELD_SIZE_1DIM,self.FIELD_SIZE_1DIM))
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
            centroid, color, start_time, start_map,
            SPLIT_THRESHOLD_FUNC_CLASS = self.SPLIT_THRESHOLD_FUNC_CLASS, 
            MOMENTUM_FUNC_CLASS = self.MOMENTUM_FUNC_CLASS,
            SPLIT_THRESHOLD_CONST_VALUE = self.SPLIT_THRESHOLD_CONST_VALUE,
            MOMENTUM_FUNC_STATIC_BOOL = self.MOMENTUM_FUNC_STATIC_BOOL
            
        )
      
                    

    def do_moves(self):
        for i,l in enumerate(self.languages):
            if l is not None and l.alive:
                other_ling_areas = np.zeros((self.FIELD_SIZE_1DIM,self.FIELD_SIZE_1DIM)).astype(bool)
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
                    for j in [k for k in list(np.arange(0,self.MAX_NUMBER_LANGUAGES,1)) if k != i]:
                        if self.hull_points[j] is not None and self.languages[j] is not None and l.momentum <= self.languages[j].momentum:
                            
                            #print(l.momentum)
                            #print(self.languages[i].momentum)

                            try:
                                other_ling_areas = \
                                    other_ling_areas |\
                                    np.array(Delaunay(self.hull_points[j]).find_simplex(
                                        [(ii,ji) for ii in np.arange(0,self.FIELD_SIZE_1DIM,1) for ji in np.arange(0,self.FIELD_SIZE_1DIM,1)]
                                    )>=0).reshape(self.FIELD_SIZE_1DIM,self.FIELD_SIZE_1DIM).T
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
            if l is not None and l.alive:
                #l.death()
                if (l.map == 0).all(): 
                    
                    self.languages[i].alive = False
                    self.languages[i].history = self.languages[i].history + f"DEATH_{self.t}+"
                    print('DEATH HAPPENS')


    def do_splits(self):
        for i,l in enumerate(self.languages):
            if l is not None and l.alive and None in self.languages:
                 if np.random.rand() < l.split_threshold:

                    # Assuming you have a numpy array of points in the form [(x1, y1), (x2, y2), ...]
                    (As, Bs) = np.where(l.map > 0.001)
                    points = [[a,b] for a, b in zip(As, Bs)]     
                    if len(points) > 30:

                        kmeans = MiniBatchKMeans(n_clusters=2, init='random', n_init=2).fit(points)
                        preds = kmeans.predict(points)

                        #print(points)
                        #print(preds)

                        cluster1 = np.array([p for p, b in zip(points, preds) if b == 0])
                        cluster2 = np.array([p for p, b in zip(points, preds) if b == 1])
                        

                        #print(cluster1)
                        _ = np.zeros((self.FIELD_SIZE_1DIM,self.FIELD_SIZE_1DIM))
                        for p in cluster1: _[p[0],p[1]] = 1

                        l.map = deepcopy(_)

                        

                        #At first blank space in language list
                        _ = np.zeros((self.FIELD_SIZE_1DIM,self.FIELD_SIZE_1DIM))
                        for p in cluster2: _[p[0],p[1]] = 1
                        #self.languages[new_l_index].map = 


                        

                        self.languages[self.languages.index(None)] = Language(
                            #Pick random color - None, np.random.choice(self.color_list), self.t, deepcopy(_), prev_history=l.history
                            #Pick color based on parent
                            None, random_color_near(l.color), self.t, deepcopy(_), 
                            
                            SPLIT_THRESHOLD_FUNC_CLASS = self.SPLIT_THRESHOLD_FUNC_CLASS, 
                            MOMENTUM_FUNC_CLASS = self.MOMENTUM_FUNC_CLASS,
                            SPLIT_THRESHOLD_CONST_VALUE = self.SPLIT_THRESHOLD_CONST_VALUE,
                            MOMENTUM_FUNC_STATIC_BOOL = self.MOMENTUM_FUNC_STATIC_BOOL,
                            
                            prev_history=l.history
                        )

                        #add this history to l
                        l.append_split_history(self.t)

                        #print("SPLIT HAPPENS")
                        


                        
