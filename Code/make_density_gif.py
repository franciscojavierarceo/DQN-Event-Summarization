import os
import sys
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def main(nepochs, model):
    if type(nepochs) == str:
        nepochs = int(nepochs)
    
    # Pulling in the images that were exported
    afile_names = [('./plotdata/%s/00%i_actual.txt' % (model, x) ) for x in range(nepochs) ] 
    pfile_names = [('./plotdata/%s/00%i_preds.txt' % (model, x)) for x in range(nepochs) ] 

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

        ax = dft.plot(kind='density', 
                        figsize=(16, 8), 
                        title=('Training Epoch %i' % epoch),
                        xlim=[-2, 2], 
                        ylim = [0, 2] )
        fig = ax.get_figure()
        fig.savefig('./plotdata/plot_%i.png' % epoch )
    # Exporting the images to a gif
    file_names = [ ('./plotdata/plot_%i.png' %x) for x in range(nepochs)]
    images = []
    for filename in file_names:
        images.append(imageio.imread(filename))
        # Actual v Predicted gif
    imageio.mimsave('./avp_density.gif', images, duration=0.75)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])