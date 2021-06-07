import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILENAME = "1623095208"
TIME_PERIOD = 5 # seconds
dataframe = pd.read_csv(f"./data/{FILENAME}.csv")


dataframe.MA_prediction = dataframe.MA_prediction.replace(to_replace={"NOT_PRESENT":0,"RED":1,"YELLOW":2,"GREEN":3})
dataframe.stationary_prediction = dataframe.stationary_prediction.replace(to_replace={"NOT_PRESENT":0,"RED":1,"YELLOW":2,"GREEN":3})

plt.subplot(2,1,1)
plt.yticks(ticks=[0,1,2,3])
plt.plot(dataframe.time_since_start, dataframe.MA_prediction)
plt.subplot(2,1,2)
plt.yticks(ticks=[0,1,2,3])
plt.plot(dataframe.time_since_start, dataframe.stationary_prediction)
plt.show()

'''
dataframe.yaw = dataframe.yaw.fillna(value=180)
dataframe.pitch = dataframe.pitch.fillna(value=90)
dataframe['time_period'] = dataframe.time_since_start // TIME_PERIOD


yaw_graph_data = []
pitch_graph_data = []
groups = dataframe.groupby(by='time_period')

for group in groups['yaw']:
    yaw_graph_data.append(np.abs(np.array(group[1])))

for group in groups['pitch']:
    pitch_graph_data.append(np.abs(np.array(group[1])))


plt.subplot(2,1,1)
plt.boxplot(yaw_graph_data)
plt.subplot(2,1,2)
plt.boxplot(pitch_graph_data)
plt.show()
'''
