##### plots
#
#import plotly.figure_factory as ff
#import plotly
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
    

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

def propensity_score_plt(conditioned, random): 
    fig, axes = plt.subplots(1,2, figsize=(8,5), sharex = False)
    
    axes[0].set_title('Conditioned treatment assignment')
    axes[1].set_title('Random treatment assignment')
    
#    axes[0].set_ylabel('Number of Observations')
    axes[0].set_xlabel('Probabilities')
    axes[1].set_xlabel('Probabilities')
    
    axes[0].set_xlim([0,1])
    axes[1].set_xlim([0,1])
    
    sns.despine(left=True, right=True, top=True)
    
    sns.distplot(conditioned, ax=axes[0], hist=True, kde=False, 
             bins=10, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

    sns.distplot(random, ax=axes[1], hist=True, kde=False, 
             bins=9, color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    
    plt.savefig('propensity_score_plot.png')




def single_treatment_effect_plt(treatment, assignment, title):
    only_treat = treatment[assignment==1]
    fig, ax = plt.subplots()
    
    ax.set_title(title)
    ax.set_xlabel('Size of treatment effect')
#    ax.set_xticks(list(np.arange(-0.3,0.4,0.1)))
    #ax.set_xlim([0.1,0.3])
    
    sns.despine(left=True, right=True, top=True)
    
    sns.distplot(only_treat, ax = ax, hist=True, kde=False, 
         bins=21, color = 'darkblue', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})
    
    plt.setp(ax, yticks=[])    
    plt.tight_layout()


def all_treatment_effect_plt(treatment_list, assignment_list):
    
    only_treat_list = []
    for i in range(4):
        only_treat = treatment_list[i][assignment_list[i]==1]
        only_treat_list.append(only_treat)
    
    fig, ax = plt.subplots(2,2, figsize=(7.5,6), sharex=False)
    

    ax[0,0].set_title('Constant effect')
    ax[0,1].set_title('Heterogeneous effect')
    ax[1,0].set_title('Negative effect')
    ax[1,1].set_title('No effect')
    
    ax[0,0].set_xlabel('Size of treatment effect')
    ax[0,1].set_xlabel('Size of treatment effect')
    ax[1,0].set_xlabel('Size of treatment effect')
    ax[1,1].set_xlabel('Size of treatment effect')
  
    
    ax[0,0].set_xlim([-0.3,0.3])
    ax[0,1].set_xlim([-0.3,0.3])
    ax[1,0].set_xlim([-0.3,0.3])
    ax[1,1].set_xlim([-0.3,0.3])
      
#    fig.suptitle('Different kinds of treatment effects')

    sns.despine(left=True)
    
    sns.distplot(only_treat_list[0], ax = ax[0,0], hist=True, kde=False, 
         bins=15, color = 'darkblue', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})

    sns.distplot(only_treat_list[1], ax = ax[0,1], hist=True, kde=False, 
         bins=10, color = 'darkred', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})

    sns.distplot(only_treat_list[2], ax = ax[1,0], hist=True, kde=False, 
         bins=15, color = 'darkgreen', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})

    sns.distplot(only_treat_list[3], ax = ax[1,1], hist=True, kde=False, 
         bins=15, color = 'gold', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})
    
    plt.setp(ax, yticks=[])
    
    plt.subplots_adjust(top=2)
    plt.tight_layout()


def output_difference_plt(y_not_treated, y_treated, binary = False): 
    fig, axes = plt.subplots(1,1)
    
    if binary:
        axes.set_title('Binary output distributions')
    else:
        axes.set_title('Continuous Output distributions')
    

    
    axes.set_ylabel('Density')
    axes.set_xlabel('y')
        
    sns.distplot(y_not_treated, ax=axes, hist=False, kde=True, 
             bins=10, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4}, 
             label = 'y not treated')

    sns.distplot(y_treated, ax=axes, hist=False, kde=True, 
             bins=10, color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             label = 'y treated')

    axes.legend()    
    
    sns.despine(right=True, top=True)
    
    if binary:
        plt.setp(axes, xticks=[0,1])
    
    plt.tight_layout()

def avg_treatment_effect_plt(treatment_list, assignment_list, ate_list):
    
    only_treat_list = []
    for i in range(4):
        only_treat = treatment_list[i][assignment_list[i]==1]
        only_treat_list.append(only_treat)
    
    fig, ax = plt.subplots(2,2, figsize=(7.5,6), sharex=False)
    

    ax[0,0].set_title('Constant')
    ax[0,1].set_title('Positive/Negative')
    ax[1,0].set_title('Mixed')
    ax[1,1].set_title('Mostly none')
    
    ax[0,0].set_xlabel('Size of treatment effect')
    ax[0,1].set_xlabel('Size of treatment effect')
    ax[1,0].set_xlabel('Size of treatment effect')
    ax[1,1].set_xlabel('Size of treatment effect')
  
    
    ax[0,0].set_xlim([-1,1])
    ax[0,1].set_xlim([-1,1])
    ax[1,0].set_xlim([-1,1])
    ax[1,1].set_xlim([-1,1])
    
    ax[0,0].axvline(ate_list[0], 0, len(treatment_list), linewidth=4, color='r')
    ax[0,1].axvline(ate_list[1], 0, len(treatment_list), linewidth=4, color='r')
    ax[1,0].axvline(ate_list[2], 0, len(treatment_list), linewidth=4, color='r')
    ax[1,1].axvline(ate_list[3], 0, len(treatment_list), linewidth=4, color='r')
    
    
#    fig.suptitle('Different kinds of treatment effects')

    sns.despine(left=True)
    
    sns.distplot(only_treat_list[0], ax = ax[0,0], hist=True, kde=False, 
         bins=15, color = 'darkblue', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})

    sns.distplot(only_treat_list[1], ax = ax[0,1], hist=True, kde=False, 
         bins=10, color = 'darkred', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})

    sns.distplot(only_treat_list[2], ax = ax[1,0], hist=True, kde=False, 
         bins=15, color = 'darkgreen', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})

    sns.distplot(only_treat_list[3], ax = ax[1,1], hist=True, kde=False, 
         bins=9, color = 'gold', 
         hist_kws={'edgecolor':'black'},
         kde_kws={'linewidth': 4})
    
    plt.setp(ax, yticks=[])
    
    plt.subplots_adjust(top=2)
    plt.tight_layout()















