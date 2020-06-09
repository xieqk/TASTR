import visdom
import time
import numpy as np


class Visualizer(object):
    def __init__(self, env='default'):
        self.vis = visdom.Visdom(env=env)
        self.index = {}
    
    def reinit(self, env='default'):
        '''
        修改visdom配置
        '''
        self.vis = visdom.Visdom(env=env)
        return self

    def plot_line(self, win, name, epoch, y):
        '''
        self.plot_line('loss', 1.00)
        '''
        x = self.index.get(win, 0)
        self.win_result = self.vis.line(
            Y=np.array([y]), X=np.array([epoch]),
            win=win, name=name, opts=dict(title=win, showlegend=True, width=800, height=700, xlabel='Epoch'),
            update = None if x == 0 else 'append'
            )
        self.index[win] = x + 1

    def plot_res(self, epoch, res):
        for res_name, value in res.items():
            self.plot_line(win='Result', name=res_name, epoch=epoch, y=value)

    def plot_loss(self, epoch, loss):
        for loss_name, value in loss.items():
            self.plot_line(win='Loss', name=loss_name, epoch=epoch, y=value)

        