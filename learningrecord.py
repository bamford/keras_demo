# Heavily adapted from https://github.com/stared/livelossplot/

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, clear_output
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.models as models
import time


class LearningRecord(Callback):
    def __init__(self, name, model=None, load=None):
        self.name = name
        self.recordfile = f'record/{name}_record.h5'
        self.modelfile = f'record/{name}_model.h5'
        self.logfile = f'record/{name}_log.csv'
        self.plotfile = f'record/{name}_plot.pdf'
        self.resultsfile = f'record/{name}_results.h5'
        self.statsfile = f'record/{name}_stats.csv'
        self.model = model
        self.df = None
        self.count = 0
        if load is not None:
            try:
                resume_df = pd.read_hdf(load, 'df')
            except FileNotFoundError:
                pass
            else:
                self.df = resume_df
                self.count = len(self.df)
    
    def on_train_begin(self, logs={}):
        self.fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(16, 6))
        
        ax1.set_yscale('log')
        self.loss_plot, = ax1.plot([1, 1], [1, 2], label="training")
        self.val_loss_plot, = ax1.plot([1, 1], [1, 2], label="validation")
        self.loss_plot.set_visible(False)
        self.val_loss_plot.set_visible(False)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend()
        
        self.acc_plot, = ax2.plot([0, 1], [0, 1], label="training")
        self.val_acc_plot, = ax2.plot([0, 1], [0, 1], label="validation")
        self.acc_plot.set_visible(False)
        self.val_acc_plot.set_visible(False)
        ax2.set_ylabel('accuracy')
        ax2.legend()
        display(self.fig)
        
        self.lines = (self.loss_plot, self.val_loss_plot, self.acc_plot, self.val_acc_plot)
        self.update_plot()

    def on_epoch_end(self, epoch, logs={}):
        self.count += 1
        v = np.array([list(logs.values())])
        c = list(logs.keys())
        newdf = pd.DataFrame(v, columns=c, index=[self.count])
        if self.df is None:
            self.df = newdf
        else:
            self.df = pd.concat((self.df, newdf))

        self.save()
            
        self.update_plot()
        if self.plotfile is not None:
            plt.savefig(self.plotfile)
        time.sleep(1)


    def update_plot(self):
        if self.count > 1:
            for l in self.lines:
                l.set_visible(True)
                l.set_xdata(self.df.index)
            self.loss_plot.set_ydata(self.df.loss)
            self.val_loss_plot.set_ydata(self.df.val_loss)
            self.acc_plot.set_ydata(self.df.acc)
            self.val_acc_plot.set_ydata(self.df.val_acc)
        
            for ax in self.fig.axes:
                ax.legend()
                ax.relim()
                ax.autoscale_view()
            
            clear_output(wait=True)
            display(self.fig)
        
    def on_train_end(self, log):
        clear_output()

    def load_model(self):
        try:
            self.model = models.load_model(self.modelfile)
            self.df = pd.read_hdf(self.recordfile, 'df')
            self.count = len(self.df)
        except FileNotFoundError:
            pass
        return self.model

    def save(self):
        try:
            nold = len(pd.read_hdf(self.recordfile, 'df'))
        except FileNotFoundError:
            nold = 0
        nnew = len(self.df) if self.df is not None else 0
        if nnew < nold:
            print('Refusing to overwrite larger existing record and corresponding model')
        else:
            if self.df is not None:
                self.df.to_hdf(self.recordfile, 'df')
                self.df.to_csv(self.logfile)
            if self.model is not None:
                self.model.save(self.modelfile)
                if self.count % 10 == 0:
                    self.model.save(self.modelfile.replace('.h5', f'_epoch_{self.count}.h5'))