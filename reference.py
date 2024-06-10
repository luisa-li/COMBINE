import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import math as math
argfig, argax = plt.subplots(figsize=(8,8),subplot_kw=dict(projection='3d')) #help
## Adjust axes limits according to your problem. Here we don't need more than a couple of meters left or right, and 600 meters up
argax.set(xlim=(-7.1*10**(-9), 7.1*10**(-9)), ylim=(-7.1*10**(-9), 7.1*10**(-9)),zlim=(-7.1*10**(-9), 7.1*10**(-9)), xlabel='Position, meters', ylabel='Height, meters',zlabel='Width, meters', title='Argon Gas')

#Parameters:
time = 7*10**(-11) #s
argmass = 3*10**(-26) #kg
argsize = 7.1*10**(-11) #meters
temp = 100 #K
vi = 0.1 #m/s

#number of steps
dt = time/400
N = int(time/dt)
# print(N)
def rand1orneg():
    b = np.random.randint(1,3)
    if b == 1:
        return 1
    else:
        return -1

#force
def LDF_force(displacement):
    if displacement == 0:
        return 0
    else:
        return -((6.13817*10**(-77))*((displacement**(6))-(3.08961*10**(-57))))/((displacement)**(13))



#argon
def main(n_a):
    vt = np.zeros((N+1, n_a, 3))
    rt = np.zeros((N+1, n_a, 3))
    ft = np.zeros((N+1, n_a, 3))

    for n in range(n_a):
        rt[0,n] = np.array([np.random.uniform(-6.9*10**(-9), 6.9*10**(-9)),np.random.uniform(-6.9*10**(-9), 6.9*10**(-9)),np.random.uniform(-6.9*10**(-9), 6.9*10**(-9))])
        vt[0,n] = np.array([(rand1orneg())*vi,(rand1orneg())*vi,(rand1orneg())*vi])

    

    
    def allbutme_force(time,myindex):
        degree_of_flex = -10
        notme_x = []
        notme_y = []
        notme_z = []
        notme_f_x = []
        notme_f_y = []
        notme_f_z = []
        for n in range(len(rt[0,:,0])):
            if n == myindex:
                continue
            elif n!= myindex:
                notme_x.append(rt[time,n,0])
                notme_y.append(rt[time,n,1])
                notme_z.append(rt[time,n,2])
        for n in range(len(rt[0,:,0])-1):
            # print(print(math.sqrt((LDF_force(notme_x[n]-rt[time,myindex,0]))**2+(LDF_force(notme_y[n]-rt[time,myindex,1]))**2+(LDF_force(notme_z[n]-rt[time,myindex,2]))**2)))
            notme_f_x.append(LDF_force(notme_x[n]-rt[time,myindex,0]))
            if notme_f_x[n] > 1*10**(degree_of_flex) or notme_f_x[n] < -1*10**(degree_of_flex):
                notme_f_x[n] = 0
            notme_f_y.append(LDF_force(notme_y[n]-rt[time,myindex,1]))
            if notme_f_y[n] > 1*10**(degree_of_flex) or notme_f_y[n] < -1*10**(degree_of_flex):
                notme_f_y[n] = 0
            notme_f_z.append(LDF_force(notme_z[n]-rt[time,myindex,2]))
            if notme_f_z[n] > 1*10**(degree_of_flex) or notme_f_z[n] < -1*10**(degree_of_flex):
                notme_f_z[n] = 0
        ft[time+1, myindex, 0] = sum(notme_f_x)
        ft[time+1, myindex, 1] = sum(notme_f_y)
        ft[time+1, myindex, 2] = sum(notme_f_z)

    def num_in_box(time):
        num = 0
        for r in range(n_a):
            for x in range(len(rt[0,0,:])):
                if rt[time+1,r,x] < -7.1*10**(-9) or rt[time+1,r,x] > 7.1*10**(-9):
                    continue
                else:
                    num = num+1
        return num

    #force of collision with the axis
    def edgecollisioncheck(time,particle):
        if rt[time,particle,0] >= 7.1*10**(-9)  or rt[time,particle,0] <= -7.1*10**(-9):
            # rt[time,particle,0] = (-1)*rt[time,particle,0]
            ft[time+1,particle,0] = (0.2)*ft[time+1,particle,0]
        elif rt[time,particle,1] >= 7.1*10**(-9) or rt[time,particle,1] <= -7.1*10**(-9):
            # rt[time,particle,1] = (-1)*rt[time,particle,1]
            ft[time+1,particle,1] = (0.2)*ft[time+1,particle,1]
        elif rt[time,particle,2] >= 7.1*10**(-9) or rt[time,particle,2] <= -7.1*10**(-9):
            rt[time,particle,2] = (-1)*rt[time,particle,2]
            # ft[time+1,particle,2] = (0.2)*ft[time+1,particle,2]

    def part_collision(time,myindex):
        notme_x = []
        notme_y = []
        notme_z = []
        for n in range(len(rt[0,:,0])):
            if n == myindex:
                continue
            elif n!= myindex:
                notme_x.append(rt[time,n,0])
                notme_y.append(rt[time,n,1])
                notme_z.append(rt[time,n,2])
        for l in range(len(rt[0,:,0])-1):
            if abs(rt[time,l,0]-rt[time,myindex,0])<=2*argsize/1000:
                vt[time+1,myindex,0] = -0.5*vt[time+1,myindex,0]
            elif abs(rt[time,l,1]-rt[time,myindex,1])<=2*argsize/1000:
                vt[time+1,myindex,1] = -0.5*vt[time+1,myindex,1]
            elif abs(rt[time,l,2]-rt[time,myindex,2])<=2*argsize/1000:
                vt[time+1,myindex,2] = -0.5*vt[time+1,myindex,2]


    for n in range(N):
        for r in range(n_a):
            allbutme_force(n,r)
            vt[n+1,r] = vt[n,r] + (ft[n+1,r]/argmass)*dt + ft[n,r]
            part_collision(n,r)
            edgecollisioncheck(n,r)
            rt[n+1,r] = vt[n+1,r]*dt + rt[n,r]
            # print(ft[n+1,0])
            # print((ft[n+1,0]/argmass)*dt)
            # print(math.sqrt(((ft[n+1,0,0]/argmass)*dt)**2+((ft[n+1,0,1]/argmass)*dt)**2+((ft[n+1,0,2]/argmass)*dt)**2))
            # print(math.sqrt((vt[n+1,0,0])**2+(vt[n+1,0,1])**2+(vt[n+1,0,2])**2))
        print(num_in_box(n))


    scat = argax.scatter(rt[0,:,0],rt[0,:,1],rt[0,:,2],marker='o', s=5.041)

    def animate(i):
        scat._offsets3d = [rt[i,:,0], rt[i,:,1], rt[i,:,2]]
        # help

    ani = animation.FuncAnimation(argfig, animate, frames=N)
    plt.close()

    ani.save('Argon1.html', writer=animation.HTMLWriter(fps= 1//dt))
main(100)