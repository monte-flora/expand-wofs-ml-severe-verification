import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import numpy as np 
from metrics import stat_testing
import seaborn as sns
sns.set_theme()


def make_twin_ax(ax):
    """
    Create a twin axis on an existing axis with a shared x-axis
    """
    # align the twinx axis
    twin_ax = ax.twinx()

    # Turn twin_ax grid off.
    twin_ax.grid(False)

    # Set ax's patch invisible
    ax.patch.set_visible(False)
    # Set axtwin's patch visible and colorize it in grey
    twin_ax.patch.set_visible(True)

    # move ax in front
    ax.set_zorder(twin_ax.get_zorder() + 1)

    return twin_ax


def pretty_plot(ax, x, data, histdata, xlabel, ylabel, right_ylabel, model_name=None, 
                scatter=False, y=None, metric=None, hist_xdata=None):
    """
    Plot the results. 
    """
    good_colors = ['xkcd:darkish red', '#2A3459', 'black', 'magenta']
    
    if y is None:
        twin_ax = make_twin_ax(ax)
        ax.set_zorder(twin_ax.get_zorder() + 1)
    
        if isinstance(x[0], str):
            twin_ax.bar(range(len(x)), height=histdata, align='center', alpha=0.8) #edgecolor='black', color = 'grey', alpha=0.8)
        else:
            #print(x)
            twin_ax.bar(x, height=histdata, align='center', alpha=0.8) #edgecolor='black', color = 'grey', alpha=0.8)
        
        #if len(histdata) == len(x):
        #    if hist_xdata is None:
        #        twin_ax.bar(x, histdata, alpha=0.4, color = 'grey', width=3.5)
        #    else:
        #        twin_ax.bar(x, height=histdata, align='center', edgecolor='black', color = 'grey', alpha=0.8)
        #else:
        #    twin_ax.hist(x=histdata, 
        #                 alpha=0.4, 
        #                 color = 'grey', 
        #                 bins=hist_xdata,
        #                 rwidth=1.0
        #                )

    for i, key in enumerate(data.keys()):
        lower_bound, upper_bound = np.percentile(data[key], [2.5, 97.5], axis=-1)
        if scatter and y is None:
            ymean = np.mean(data[key], axis=1)
            yerr = np.array([ymean-lower_bound, upper_bound-ymean])
            
            ax.errorbar(x, ymean, yerr = yerr,
                color=good_colors[i], zorder=2, marker='o', markersize=6, 
                        label=key, fmt='o', elinewidth=2.0, capsize=5.5)
        elif y is None:
            ax.plot(x, np.mean(data[key], axis=1), 
                color=good_colors[i], linewidth=2.5, zorder=2, marker='o', label=key)
            # Plot the confidence intervals. 
            ax.fill_between(x, lower_bound, upper_bound, color=good_colors[i], alpha=0.4)
        else:
            z = np.mean(data[key], axis=-1)
            z = np.ma.masked_where(z==0., z)
            cm = ax.pcolormesh(z, cmap='hot')
            ax.set_xticks(np.arange(len(x)+1))
            ax.set_xticklabels(x)
            ax.set_yticks(np.arange(len(y)+1))
            ax.set_yticklabels(y)
            fig.colorbar(cm, ax=ax, label=metric)
            for i,j in itertools.product(range(len(x)), range(len(y))):
                value = z[j, i]
                if value is np.nan:
                    txt = 'Insuff.\nSamp.'
                else:
                    txt = f'{value:.2f}' 
        
                ax.text(i + 0.5, j + 0.5, txt,
                     horizontalalignment='center',
                     verticalalignment='center',
                        color='lightgrey'
                     )
        
    
    if y is None:
        # HARDCORING. Need to generalize! 
        p_values = np.array([stat_testing(data['ML'][i,:], data['BL'][i,:]) for i in range(len(x))])
        # Using an alpha level of 0.05 or 95% confidence. 
        p_values = np.where(p_values < 0.05,0, np.nan) 
    
        #twin_ax.scatter(x, p_values, marker='x', label=r'$\alpha$<0.05')
        twin_ax.set_ylabel(right_ylabel)
    
        #ax.grid(color='#2A3459')
        ax.legend()
        ax.tick_params(axis='x', labelrotation = 90)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
    if hist_xdata is not None:
        # Set the ticks to the middle of the bars
        ax.set_xticks([0.5+i for i,_ in enumerate(x)])

        # Set the xticklabels to a string that tells us what the bin edges were
        ax.set_xticklabels([f'{hist_xdata[i]:.0f} - {hist_xdata[i+1]:.0f}' for i,_ in enumerate(x)])