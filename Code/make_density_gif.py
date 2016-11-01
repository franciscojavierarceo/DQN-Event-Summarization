import os
import sys
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def buildCDF(df, var):
    cdf = pd.DataFrame(df[var].value_counts())
    cdf = cdf.reset_index(drop=False)
    cdf.columns = [var, 'count']
    cdf['percent'] = cdf['count'] / cdf['count'].sum()
    cdf = cdf.sort_values(by=var)
    cdf['cumpercent'] = cdf['percent'].cumsum()
    return cdf

def main(nepochs, model):
    if type(nepochs) == str:
        nepochs = int(nepochs)
    
    # Pulling in the images that were exported
    afile_names = [('./plotdata/%s/%i_actual.txt' % (model, x) ) for x in range(nepochs) ] 
    pfile_names = [('./plotdata/%s/%i_preds.txt' % (model, x)) for x in range(nepochs) ] 

    llow, lhigh = 0., 0.
    for (afile, pfile, epoch) in zip(afile_names, pfile_names, range(nepochs)):
        # Loading data sets and concatenating them
        dfy = pd.read_csv(afile, header=None, names=['actual'])
        dfp = pd.read_csv(pfile, header=None, names=['preds'])
        llow  = min(llow, dfy['actual'].min(), dfp['preds'].min())
        lhigh = max(lhigh, dfy['actual'].max(), dfp['preds'].max())

    for (afile, pfile, epoch) in zip(afile_names, pfile_names, range(nepochs)):
        # Loading data sets and concatenating them
        dfy = pd.read_csv(afile, header=None, names=['actual'])
        dfp = pd.read_csv(pfile, header=None, names=['preds'])
        dft = pd.concat([dfy, dfp], axis=1)
        # Plotting density
        ax = dft.plot(kind='density', 
                        figsize=(16, 8), 
                        title=('Training Epoch %i' % epoch),
                        xlim=[llow, lhigh], 
                        ylim = [0, 2] )
        fig = ax.get_figure()
        fig.savefig('./plotdata/plotfiles/denplot_%i.png' % epoch )

        # Plotting cdf
        cdfy = buildCDF(dfy, 'actual')
        cdfp = buildCDF(dfp, 'preds')
        plt.figure(figsize=(16,8))
        plt.xlim([llow, lhigh])
        plt.plot(cdfp['preds'], cdfp['cumpercent'], label='Predicted', c='blue')
        plt.plot(cdfy['actual'], cdfy['cumpercent'], label='Actual', c='red')
        plt.title(("Training Epoch %i" % epoch))
        plt.legend(loc="lower right")
        plt.savefig('./plotdata/plotfiles/cdfplot_%i.png' % epoch )
        plt.close()

    # Exporting the images to a gif
    file_names = [ ('./plotdata/denplot_%i.png' %x) for x in range(nepochs)]
    images = []
    for filename in file_names:
        images.append(imageio.imread(filename))
        # Actual v Predicted gif
    imageio.mimsave('./avp_density.gif', images, duration=0.75)

    file_names = [ ('./plotdata/cdfplot_%i.png' %x) for x in range(nepochs)]
    images = []
    for filename in file_names:
        images.append(imageio.imread(filename))
        # Actual v Predicted gif
    imageio.mimsave('./avp_cdf.gif', images, duration=0.75)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])