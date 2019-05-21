##### plots

import plotly.figure_factory as ff
import plotly
import numpy as np
import matplotlib.pyplot as plt

    

#def histogram(x1):
#    
#    hist_data = [x1]
#    
#    group_labels = ['propbensity score']#['No Treatment Assignment', 'Treatment Assigment']
#    
#    # Create distplot with custom bin_size
#    bin_s = list(np.arange(-50, 50)/10)
#    fig = ff.create_distplot(hist_data, group_labels, show_hist=False)#, bin_size = bin_s)
#    
#    # Plot!
#    # Adjust title, legend
#    plt.interactive(False)
#    plotly.offline.plot(fig, filename='Distplot with Multiple Bin Sizes')

def propensity_score_plt(x): 
    plt.hist(x)
    plt.xlabel(' Propensity score')
    plt.ylabel('Number of observations')
    plt.title('Propensity score distribution')
    plt.show()



plt.hist(x)

























