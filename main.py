import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
channels = ["s1d1", "s1d2", "s1d3", "s2d1", "s2d3", "s2d4", "s3d2", "s3d3", "s4d3", "s4d4"]
data = []
for channel in channels:
    ch = pd.read_csv(os.path.join("data", channel) + ".csv").transpose()
    array = np.array(ch)
    data.append(array)
data = np.array(data)

#DISPLAY ALL CHANNELS
for i in range(len(data)):
    ch = data[i]
    oxy = ch[0]
    glm_oxy = ch[1]
    deoxy = ch[2]
    glm_deoxy = ch[3]
    
    #plt.title()
    #plt.plot(oxy)
    #plt.plot(deoxy)
    #plt.plot(glm_oxy)
    #plt.plot(glm_deoxy)
    #plt.show()


mean_oxy =     np.mean([data[0][0], 
                        data[1][0],
                        data[2][0],
                        data[3][0],
                        data[4][0],
                        data[5][0],
                        data[6][0],
                        data[7][0],
                        data[8][0],
                        data[9][0]], axis=0)
mean_glm_oxy = np.mean([data[0][1], 
                        data[1][1],
                        data[2][1],
                        data[3][1],
                        data[4][1],
                        data[5][1],
                        data[6][1],
                        data[7][1],
                        data[8][1],
                        data[9][1]], axis=0)

mean_deoxy =   np.mean([data[0][2], 
                        data[1][2],
                        data[2][2],
                        data[3][2],
                        data[4][2],
                        data[5][2],
                        data[6][2],
                        data[7][2],
                        data[8][2],
                        data[9][2]], axis=0)

mean_glm_deoxy=np.mean([data[0][3], 
                        data[1][3],
                        data[2][3],
                        data[3][3],
                        data[4][3],
                        data[5][3],
                        data[6][3],
                        data[7][3],
                        data[8][3],
                        data[9][3]], axis=0)


fig, axs = plt.subplots(1, 2)
l1 = axs[0].plot(mean_oxy, color='tab:blue')
l2 = axs[0].plot(mean_deoxy, color='tab:orange')

l3 = axs[1].plot(mean_glm_oxy, color='tab:blue')
l4 = axs[1].plot(mean_glm_deoxy, color='tab:orange')


fig.legend((l1, l2), ('HbO', 'HbR'), loc='upper left')
fig.legend((l3, l4), ('HbO', 'HbR'), loc='upper right')

plt.tight_layout()
plt.show()