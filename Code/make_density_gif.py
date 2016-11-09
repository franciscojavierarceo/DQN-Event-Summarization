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
    ofile_names = [('./plotdata/%s/%i_epoch.txt' % (model, x) ) for x in range(nepochs) ] 
    llow, lhigh = -2., 2.
    for (ofile, epoch) in zip(ofile_names, range(nepochs)):
        if epoch  == 0:
            llow, lhigh = 
        # Loading data sets and concatenating them
        odf = pd.read_csv(ofile, sep=';')
        llow  = min(llow, odf['actual'].min(), odf['predSelect'].min(), odf['predSkip'].min())
        lhigh  = max(lhigh, odf['actual'].max(), odf['predSelect'].max(), odf['predSkip'].max())

    for (ofile, epoch) in zip(ofile_names, range(nepochs)):
        # Loading data sets and concatenating them
        # Two subplots, the axes array is 1-d
        odf = pd.read_csv(ofile, sep=';')
        odf['predOptimal'] = odf[['predSelect','predSkip']].max(axis=1)
        cdfp = buildCDF(odf, 'predOptimal')
        cdfa = buildCDF(odf, 'actual')
        f, axarr = plt.subplots(1, 2, figsize=(16,8))
        axarr[0].imshow(odf[['Skip', 'Select']], cmap='autumn')
        axarr[0].set_title('Actions')
        axarr[0].set_xticks([])
        axarr[1].plot(cdfp['predOptimal'], cdfp['cumpercent'], label='Predicted', c='blue')
        axarr[1].plot(cdfa['actual'], cdfa['cumpercent'], label='Actual', c='red')
        axarr[1].set_ylim([0,1])
        axarr[1].set_xlim([llow, lhigh])
        axarr[1].legend(loc ='upper left')
        axarr[1].grid()
        f.suptitle('%s model performance at epoch %i' % (model, epoch))
        f.tight_layout()
        f.savefig('./plotdata/%s/plotfiles/perfplot_%i.png' % (model, epoch) )

    # Exporting the images to a gif
    file_names = [ ('./plotdata/%s/plotfiles/perfplot_%i.png' % (model, epoch)) for epoch in range(nepochs)]
    images = []
    for filename in file_names:
        images.append(imageio.imread(filename))
        # Actual v Predicted gif
    imageio.mimsave('./perf.gif', images, duration=0.75)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])