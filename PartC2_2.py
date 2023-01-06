from matplotlib import pyplot as plt
import numpy as np

tt=np.linspace(0,1200,2401)

plt.plot(tt,pop_1_vector,'b',label='Prey')
plt.plot(tt,pop_2_vector,'r',label='Predator')
plt.plot(tt,pop_0_vector,'g',label='Mutalist')
plt.title('Population rates for Prey, Mutalist and Predator')
plt.xlabel('Time')
plt.ylabel('Population rate')
plt.grid()
plt.show()

#
#plt.plot3(pop_M0,pop_M1,pop_M2)
#mplot3d

ax = plt.axes(projection='3d')
ax.plot3D(pop_0_vector,pop_1_vector,pop_2_vector)
ax.set_title('Phase portrait')
ax.set_xlabel('Mutalist')
ax.set_ylabel('Prey')
ax.set_zlabel('Predator')
plt.show()