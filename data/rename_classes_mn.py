import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load mn data using pandas
df = pd.read_pickle('./data/example_mn_data.pkl')

correct_name = ['A - Tonic spiking', 
                'B - Class 1',
                'C - Spike frequency adaptation',
                'D - Phasic spiking',
                'E - Accommodation',
                'F - Threshold variability',
                'G - Rebound spike',
                'H - Class 2',
                'I - Integrator',
                'J - Input bistability',
                'K - Hyperpolarizing spiking',
                'L - Hyperpolarizing bursting',
                'M - Tonic bursting',
                'N - Phasic bursting',
                'O - Rebound burst',
                'P - Mixed mode',
                'Q - Afterpotentials',
                'R - Basal bistability',
                'S - Preferred frequency',
                'T - Spike latency']

# we want to iterate over the class members of the loaded pkl data and rename them according to the correct name
for i, name in enumerate(correct_name):
    # find the according class in the pkl
    old_name = name.split(' - ')[-1]
    # df must have a column named 'class' with old_name as member
    df.loc[df['class'] == old_name, 'class'] = name

    # let us visualize the data
    # plt.plot(df.loc[df['class'] == name, 'data'].values[0])
    # plt.title(name)
    # plt.show()
    print(name)
    print(np.unique(df.loc[df['class'] == name, 'data'].values[0]))
    print()
    pass

# now we can save the data again
df.to_pickle('./data/example_mn_data_renamed.pkl')

