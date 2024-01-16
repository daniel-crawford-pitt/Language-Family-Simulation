import matplotlib.pyplot as plt
import matplotlib.animation as animation

from objects.language import *
from objects.env import *
from simulation.sim_main import *

#l1 = Language((1,1), 'Blues')
#l2 = Language((99,99), 'Reds')
#l3 = Language((50,50), 'Greens')


#TEST

'''

ls = [
    #Language((1,1), 'Blues'),
    Language((99,99), 'Reds'),
    Language((50,50), 'Greens')
]
sim_anim = simulate(ls)

plt.show()

plt.close()
'''

#Sim Env
env = Env()

sim_anim = env.sim()
plt.show()
#f = r"C:/Users/dcraw/OneDrive/Desktop/Language Family Simulation/anims" 
#writergif = animation.PillowWriter(fps=30) 
#sim_anim.save(f, writer=writergif)






