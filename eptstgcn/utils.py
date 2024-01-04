import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import animation
import json
import urllib

# # torch
import torch
# import torch.nn.functional as F
import torch_geometric_temporal
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
# from torch_geometric_temporal.nn.recurrent import GConvGRU


# utils
#import copy
#import time
import pickle
import itertools
#from tqdm import tqdm
#import warnings

import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'NanumGothic'


temporal_signal_split = torch_geometric_temporal.signal.temporal_signal_split

# def save_data(data_dict,fname):
#     with open(fname,'wb') as outfile:
#         pickle.dump(data_dict,outfile)
        
# def load_data(fname):
#     with open(fname, 'rb') as outfile:
#         data_dict = pickle.load(outfile)
#     return data_dict

def load_data(fname):
    with open(fname, 'r') as f:
        data_dict = json.load(f)
    return data_dict
        
def save_data(fname, data_dict):
    with open(fname, 'w') as f:
        json.dump(data_dict, f)

def minmaxscaler(arr):
    arr = arr - arr.min()
    arr = arr/arr.max()
    return arr 


class DatasetLoader(object):
    """Hourly solar radiation of observatories from South Korean  for 2 years. 
    Vertices represent 44 cities and the weighted edges represent the strength of the relationship. 
    The target variable allows regression operations. 
    (The weight is the correlation coefficient of solar radiation by region.)
    """

#     def __init__(self, url):
#         self.url = url
#         self._read_web_data()
        
#     def _read_web_data(self):
#         self._dataset = json.loads(urllib.request.urlopen(self.url).read().decode())

    def __init__(self, data_dict):
        self._dataset = data_dict
    
    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        # self._edge_weights = np.array(self._dataset["weights"]).T
        edge_weights = np.array(self._dataset["weights"]).T
        scaled_edge_weights = minmaxscaler(edge_weights)
        self._edge_weights = scaled_edge_weights
    """
    def _get_targets_and_features(self):
        stacked_target = np.stack(self._dataset["FX"])
        standardized_target = (stacked_target - np.mean(stacked_target, axis=0)) / (
            np.std(stacked_target, axis=0) + 10 ** -10
        )
        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(standardized_target.shape[0] - self.lags)
        ]
        """
    def _get_targets_and_features(self):
        stacked_target = np.array(self._dataset["FX"])
        self.features = [
            stacked_target[i : i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]
        self.targets = [
            stacked_target[i + self.lags, :].T
            for i in range(stacked_target.shape[0] - self.lags)
        ]


    def get_dataset(self, lags: int = 4) -> StaticGraphTemporalSignal:
        """Returning the Solar radiation Output data iterator.
        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Solar radiation Output dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
    
class Evaluator():
    def __init__(self,learner,train_dataset,test_dataset):
        self.learner = learner
        # self.learner.model.eval()
        try:self.learner.model.eval()
        except:pass
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lags = self.learner.lags
        rslt_tr = self.learner(self.train_dataset) 
        rslt_test = self.learner(self.test_dataset)
        self.X_tr = rslt_tr['X']
        self.y_tr = rslt_tr['y']
        self.f_tr = torch.concat([self.train_dataset[0].x.T,self.y_tr],axis=0).float()
        self.yhat_tr = rslt_tr['yhat']
        self.fhat_tr = torch.concat([self.train_dataset[0].x.T,self.yhat_tr],axis=0).float()
        self.X_test = rslt_test['X']
        self.y_test = rslt_test['y']
        self.f_test = self.y_test 
        self.yhat_test = rslt_test['yhat']
        self.fhat_test = self.yhat_test
        self.f = torch.concat([self.f_tr,self.f_test],axis=0)
        self.fhat = torch.concat([self.fhat_tr,self.fhat_test],axis=0)
    def calculate_mse(self):
        test_base_mse_eachnode = ((self.y_test - self.y_test.mean(axis=0).reshape(-1,self.y_test.shape[-1]))**2).mean(axis=0).tolist()
        test_base_mse_total = ((self.y_test - self.y_test.mean(axis=0).reshape(-1,self.y_test.shape[-1]))**2).mean().item()
        train_mse_eachnode = ((self.y_tr-self.yhat_tr)**2).mean(axis=0).tolist()
        train_mse_total = ((self.y_tr-self.yhat_tr)**2).mean().item()
        test_mse_eachnode = ((self.y_test-self.yhat_test)**2).mean(axis=0).tolist()
        test_mse_total = ((self.y_test-self.yhat_test)**2).mean().item()
        self.mse = {'train': {'each_node': train_mse_eachnode, 'total': train_mse_total},
                    'test': {'each_node': test_mse_eachnode, 'total': test_mse_total},
                    'test(base)': {'each_node': test_base_mse_eachnode, 'total': test_base_mse_total},
                   }
    def _plot(self,*args,t=None,h=2.5,max_node=44,**kwargs):
        T,N = self.f.shape
        if t is None: t = range(T)
        fig = plt.figure()
        nof_axs = max(min(N,max_node),2)
        if min(N,max_node)<2: 
            print('max_node should be >=2')
        ax = fig.subplots(nof_axs ,1)
        for n in range(nof_axs):
            ax[n].plot(t,self.f[:,n],color='gray',*args,**kwargs)
            # ax[n].set_title('node='+str(n))
            ax[n].set_title(str(n))
        fig.set_figheight(nof_axs*h)
        fig.tight_layout()
        plt.close()
        return fig
    def plot(self,*args,t=None,h=2.5,**kwargs):
        self.calculate_mse()
        fig = self._plot(*args,t=None,h=2.5,**kwargs)
        ax = fig.get_axes()
        node_ids = ['북춘천', '철원', '대관령', '춘천', '백령도', '북강릉', '강릉', '서울', '인천', '원주',
       '울릉도', '수원', '서산', '청주', '대전', '추풍령', '안동', '포항', '대구', '전주', '창원',
       '광주', '부산', '목포', '여수', '흑산도', '고창', '홍성', '제주', '고산', '진주', '고창군',
       '영광군', '김해시', '순창군', '북창원', '양산시', '보성군', '강진군', '의령군', '함양군',
       '광양시', '청송군', '경주시']
        for i,a in enumerate(ax):
            _mse1= self.mse['train']['each_node'][i]
            _mse2= self.mse['test']['each_node'][i]
            _mse3= self.mse['test(base)']['each_node'][i]
            """
            # _mrate = self.learner.mrate_eachnode if set(dir(self.learner.mrate_eachnode)) & {'__getitem__'} == set() else self.learner.mrate_eachnode[i]
            # _title = 'node{0}, mrate: {1:.2f}% \n mse(train) = {2:.2f}, mse(test) = {3:.2f}, mse(test_base) = {4:.2f}'.format(i,_mrate*100,_mse1,_mse2,_mse3)
            """
            # _title = 'node{0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}, mse(test_base) = {3:.2f}'.format(i, _mse1, _mse2, _mse3)
            _title = 'node: {0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}, mse(test_base) = {3:.2f}'.format(node_ids[i], _mse1, _mse2, _mse3)
            a.set_title(_title)
            _t1 = self.lags
            _t2 = self.yhat_tr.shape[0]+self.lags
            _t3 = len(self.f)
            a.plot(range(_t1,_t2),self.yhat_tr[:,i],label='fitted (train)',color='C0')
            a.plot(range(_t2,_t3),self.yhat_test[:,i],label='fitted (test)',color='C1')
            a.legend()
        _mse1= self.mse['train']['total']
        _mse2= self.mse['test']['total']
        _mse3= self.mse['test(base)']['total']
        _title =\
        'dataset: {0} \n method: {1} \n epochs={2} \n number of filters={3} \n lags = {4} \n mse(train) = {5:.2f}, mse(test) = {6:.2f}, mse(test_base) = {7:.2f} \n'.\
        format(self.learner.dataset_name,self.learner.method,self.learner.epochs,self.learner.nof_filters,self.learner.lags,_mse1,_mse2,_mse3)
        fig.suptitle(_title)
        fig.tight_layout()
        return fig
    
    ## 추가
    # def _plot2(self,*args,t=None,h=2.5,max_node=44,**kwargs):
    #     T,N = self.f_tr.shape
    #     # if t is None: t = range(T)
    #     if t is None: t = 30
    #     fig = plt.figure()
    #     nof_axs = max(min(N,max_node),2)
    #     if min(N,max_node)<2: 
    #         print('max_node should be >=2')
    #     ax = fig.subplots(nof_axs ,1)
    #     for n in range(nof_axs):
    #         # ax[n].plot(t,self.f_tr[:,n],color='gray',*args,**kwargs)
    #         ax[n].plot(np.array(self.train_dataset.targets)[:t,n], color='gray',*args,**kwargs)
    #         # ax[n].set_title('node='+str(n))
    #         ax[n].set_title(str(n))
    #     fig.set_figheight(nof_axs*h)
    #     fig.tight_layout()
    #     plt.close()
    #     return fig
    
    def _plot2(self,*args,t=None,h=2.5,max_node=44,**kwargs):
        T,N = self.f_tr.shape
        # if t is None: t = range(T)
        if t is None: t = 30
        fig = plt.figure()
        nof_axs = max(min(N,max_node),2)
        if min(N,max_node)<2: 
            print('max_node should be >=2')
        ax = fig.subplots(nof_axs ,1)
        for n in range(nof_axs):
            # ax[n].plot(t,self.f_tr[:,n],color='gray',*args,**kwargs)
            ax[n].plot(np.array(self.train_dataset.targets)[:t,n], color='gray',*args,**kwargs)
            # ax[n].set_title('node='+str(n))
            ax[n].set_title(str(n))
        fig.set_figheight(nof_axs*h)
        fig.tight_layout()
        plt.close()
        return fig
    
    
    def tr_plot(self,*args,t=None,h=2.5,**kwargs):
        self.calculate_mse()
        fig = self._plot2(*args,t=None,h=2.5,**kwargs)
        ax = fig.get_axes()
        node_ids = ['북춘천', '철원', '대관령', '춘천', '백령도', '북강릉', '강릉', '서울', '인천', '원주',
                   '울릉도', '수원', '서산', '청주', '대전', '추풍령', '안동', '포항', '대구', '전주', '창원',
                   '광주', '부산', '목포', '여수', '흑산도', '고창', '홍성', '제주', '고산', '진주', '고창군',
                   '영광군', '김해시', '순창군', '북창원', '양산시', '보성군', '강진군', '의령군', '함양군',
                   '광양시', '청송군', '경주시']
        for i,a in enumerate(ax):
            _mse1= self.mse['train']['each_node'][i]
            # _mse2= self.mse['test']['each_node'][i]
            # _mse3= self.mse['test(base)']['each_node'][i]
            """
            # _mrate = self.learner.mrate_eachnode if set(dir(self.learner.mrate_eachnode)) & {'__getitem__'} == set() else self.learner.mrate_eachnode[i]
            # _title = 'node{0}, mrate: {1:.2f}% \n mse(train) = {2:.2f}, mse(test) = {3:.2f}, mse(test_base) = {4:.2f}'.format(i,_mrate*100,_mse1,_mse2,_mse3)
            """
            # _title = 'node{0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}, mse(test_base) = {3:.2f}'.format(i, _mse1, _mse2, _mse3)
            # _title = 'node: {0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}, mse(test_base) = {3:.2f}'.format(node_ids[i], _mse1, _mse2, _mse3)
            _title = 'node: {0}, \n mse(train) = {1:.2f}'.format(node_ids[i], _mse1)
            a.set_title(_title)
            # _t1 = self.lags
            # _t2 = self.yhat_tr.shape[0]+self.lags
            # _t3 = len(self.f)
            a.plot(range(t),self.yhat_tr[:t,i],label='fitted (train)',color='C0')
            # a.plot(range(_t2,_t3),self.yhat_test[:,i],label='fitted (test)',color='C1')
            a.legend()
        _mse1= self.mse['train']['total']
        _mse2= self.mse['test']['total']
        _mse3= self.mse['test(base)']['total']
        _title =\
        'dataset: {0} \n method: {1} \n epochs={2} \n number of filters={3} \n lags = {4} \n mse(train) = {5:.2f}, mse(test) = {6:.2f}, mse(test_base) = {7:.2f} \n'.\
        format(self.learner.dataset_name,self.learner.method,self.learner.epochs,self.learner.nof_filters,self.learner.lags,_mse1,_mse2,_mse3)
        fig.suptitle(_title)
        fig.suptitle(_title, y=1.00)
        fig.tight_layout()
        return fig
    
    
        ## 추가
        
    def _plot3(self,*args,t=None,h=2.5,max_node=44,**kwargs):
        # T,N = self.f_tr.shape
        T,N = self.f_test.shape
        # if t is None: t = range(T)
        if t is None: t = 30
        fig = plt.figure()
        nof_axs = max(min(N,max_node),2)
        if min(N,max_node)<2: 
            print('max_node should be >=2')
        ax = fig.subplots(nof_axs ,1)
        for n in range(nof_axs):
            # ax[n].plot(t,self.f_tr[:,n],color='gray',*args,**kwargs)
            ax[n].plot(np.array(self.test_dataset.targets)[:t,n], color='gray',*args,**kwargs)
            # ax[n].set_title('node='+str(n))
            ax[n].set_title(str(n))
        fig.set_figheight(nof_axs*h)
        fig.tight_layout()
        plt.close()
        return fig
    
    def test_plot(self,*args,t=None,h=2.5,**kwargs):
        self.calculate_mse()
        fig = self._plot3(*args,t=None,h=2.5,**kwargs)
        ax = fig.get_axes()
        node_ids = ['북춘천', '철원', '대관령', '춘천', '백령도', '북강릉', '강릉', '서울', '인천', '원주',
                   '울릉도', '수원', '서산', '청주', '대전', '추풍령', '안동', '포항', '대구', '전주', '창원',
                   '광주', '부산', '목포', '여수', '흑산도', '고창', '홍성', '제주', '고산', '진주', '고창군',
                   '영광군', '김해시', '순창군', '북창원', '양산시', '보성군', '강진군', '의령군', '함양군',
                   '광양시', '청송군', '경주시']
        for i,a in enumerate(ax):
            _mse1= self.mse['train']['each_node'][i]
            _mse2= self.mse['test']['each_node'][i]
            # _mse3= self.mse['test(base)']['each_node'][i]
            """
            # _mrate = self.learner.mrate_eachnode if set(dir(self.learner.mrate_eachnode)) & {'__getitem__'} == set() else self.learner.mrate_eachnode[i]
            # _title = 'node{0}, mrate: {1:.2f}% \n mse(train) = {2:.2f}, mse(test) = {3:.2f}, mse(test_base) = {4:.2f}'.format(i,_mrate*100,_mse1,_mse2,_mse3)
            """
            # _title = 'node{0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}, mse(test_base) = {3:.2f}'.format(i, _mse1, _mse2, _mse3)
            # _title = 'node: {0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}, mse(test_base) = {3:.2f}'.format(node_ids[i], _mse1, _mse2, _mse3)
            _title = 'node: {0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}'.format(node_ids[i], _mse1, _mse2)
            a.set_title(_title)
            # _t1 = self.lags
            # _t2 = self.yhat_tr.shape[0]+self.lags
            # _t3 = len(self.f)
            # a.plot(range(t),self.yhat_tr[:t,i],label='fitted (train)',color='C0')
            a.plot(range(t),self.yhat_test[:t,i],label='fitted (test)',color='C1')
            a.legend()
        _mse1= self.mse['train']['total']
        _mse2= self.mse['test']['total']
        _mse3= self.mse['test(base)']['total']
        _title =\
        'dataset: {0} \n method: {1} \n epochs={2} \n number of filters={3} \n lags = {4} \n mse(train) = {5:.2f}, mse(test) = {6:.2f}, mse(test_base) = {7:.2f} \n'.\
        format(self.learner.dataset_name,self.learner.method,self.learner.epochs,self.learner.nof_filters,self.learner.lags,_mse1,_mse2,_mse3)
        fig.suptitle(_title)
        fig.suptitle(_title, y=0.999)
        fig.tight_layout()
        return fig
    
### Toy Example용
class ToyEvaluator():
    def __init__(self,learner,train_dataset,test_dataset):
        self.learner = learner
        # self.learner.model.eval()
        try:self.learner.model.eval()
        except:pass
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lags = self.learner.lags
        rslt_tr = self.learner(self.train_dataset) 
        rslt_test = self.learner(self.test_dataset)
        self.X_tr = rslt_tr['X']
        self.y_tr = rslt_tr['y']
        self.f_tr = torch.concat([self.train_dataset[0].x.T,self.y_tr],axis=0).float()
        self.yhat_tr = rslt_tr['yhat']
        self.fhat_tr = torch.concat([self.train_dataset[0].x.T,self.yhat_tr],axis=0).float()
        self.X_test = rslt_test['X']
        self.y_test = rslt_test['y']
        self.f_test = self.y_test 
        self.yhat_test = rslt_test['yhat']
        self.fhat_test = self.yhat_test
        self.f = torch.concat([self.f_tr,self.f_test],axis=0)
        self.fhat = torch.concat([self.fhat_tr,self.fhat_test],axis=0)
    def calculate_mse(self):
        test_base_mse_eachnode = ((self.y_test - self.y_test.mean(axis=0).reshape(-1,self.y_test.shape[-1]))**2).mean(axis=0).tolist()
        test_base_mse_total = ((self.y_test - self.y_test.mean(axis=0).reshape(-1,self.y_test.shape[-1]))**2).mean().item()
        train_mse_eachnode = ((self.y_tr-self.yhat_tr)**2).mean(axis=0).tolist()
        train_mse_total = ((self.y_tr-self.yhat_tr)**2).mean().item()
        test_mse_eachnode = ((self.y_test-self.yhat_test)**2).mean(axis=0).tolist()
        test_mse_total = ((self.y_test-self.yhat_test)**2).mean().item()
        self.mse = {'train': {'each_node': train_mse_eachnode, 'total': train_mse_total},
                    'test': {'each_node': test_mse_eachnode, 'total': test_mse_total},
                    'test(base)': {'each_node': test_base_mse_eachnode, 'total': test_base_mse_total},
                   }
        
    def _plot(self,*args,t=None,h=2.5,max_node=44,**kwargs):
        T,N = self.f.shape
        if t is None: t = range(T)
        fig = plt.figure()
        nof_axs = max(min(N,max_node),2)
        if min(N,max_node)<2: 
            print('max_node should be >=2')
        ax = fig.subplots(nof_axs ,1)
        for n in range(nof_axs):
            ax[n].plot(t,self.f[:,n],color='gray',*args,**kwargs)
            # ax[n].set_title('node='+str(n))
            ax[n].set_title(str(n))
        fig.set_figheight(nof_axs*h)
        fig.tight_layout()
        plt.close()
        return fig
    def plot(self,*args,t=None,h=2.5,**kwargs):
        self.calculate_mse()
        fig = self._plot(*args,t=None,h=2.5,**kwargs)
        ax = fig.get_axes()
        for i,a in enumerate(ax):
            _mse1= self.mse['train']['each_node'][i]
            _mse2= self.mse['test']['each_node'][i]
            _mse3= self.mse['test(base)']['each_node'][i]
            _title = 'node: {0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}, mse(test_base) = {3:.2f}'.format(i, _mse1, _mse2, _mse3)
            a.set_title(_title)
            _t1 = self.lags
            _t2 = self.yhat_tr.shape[0]+self.lags
            _t3 = len(self.f)
            a.plot(range(_t1,_t2),self.yhat_tr[:,i],label='fitted (train)',color='C0')
            a.plot(range(_t2,_t3),self.yhat_test[:,i],label='fitted (test)',color='C1')
            a.legend()
        _mse1= self.mse['train']['total']
        _mse2= self.mse['test']['total']
        _mse3= self.mse['test(base)']['total']
        _title =\
        'dataset: {0} \n method: {1} \n epochs={2} \n number of filters={3} \n lags = {4} \n mse(train) = {5:.2f}, mse(test) = {6:.2f}, mse(test_base) = {7:.2f} \n'.\
        format(self.learner.dataset_name,self.learner.method,self.learner.epochs,self.learner.nof_filters,self.learner.lags,_mse1,_mse2,_mse3)
        fig.suptitle(_title)
        fig.tight_layout()
        return fig

    def _plot2(self,*args,t=None,h=2.5,max_node=44,**kwargs):
        T,N = self.f_tr.shape
        # if t is None: t = range(T)
        if t is None: t = 30
        fig = plt.figure()
        nof_axs = max(min(N,max_node),2)
        if min(N,max_node)<2: 
            print('max_node should be >=2')
        ax = fig.subplots(nof_axs ,1)
        for n in range(nof_axs):
            # ax[n].plot(t,self.f_tr[:,n],color='gray',*args,**kwargs)
            ax[n].plot(np.array(self.train_dataset.targets)[:t,n], color='gray',*args,**kwargs)
            # ax[n].set_title('node='+str(n))
            ax[n].set_title(str(n))
        fig.set_figheight(nof_axs*h)
        fig.tight_layout()
        plt.close()
        return fig
    
    def tr_plot(self,*args,t=None,h=2.5,**kwargs):
        self.calculate_mse()
        fig = self._plot2(*args,t=None,h=2.5,**kwargs)
        ax = fig.get_axes()
        for i,a in enumerate(ax):
            _mse1= self.mse['train']['each_node'][i]
            _title = 'node: {0}, \n mse(train) = {1:.2f}'.format(i+1, _mse1)
            a.set_title(_title)
            # _t1 = self.lags
            # _t2 = self.yhat_tr.shape[0]+self.lags
            # _t3 = len(self.f)
            a.plot(range(t),self.yhat_tr[:t,i],label='fitted (train)',color='C0')
            # a.plot(range(_t2,_t3),self.yhat_test[:,i],label='fitted (test)',color='C1')
            a.legend()
        _mse1= self.mse['train']['total']
        _mse2= self.mse['test']['total']
        _mse3= self.mse['test(base)']['total']
        _title =\
        'dataset: {0} \n method: {1} \n epochs={2} \n number of filters={3} \n lags = {4} \n mse(train) = {5:.2f}, mse(test) = {6:.2f}, mse(test_base) = {7:.2f} \n'.\
        format(self.learner.dataset_name,self.learner.method,self.learner.epochs,self.learner.nof_filters,self.learner.lags,_mse1,_mse2,_mse3)
        fig.suptitle(_title)
        fig.suptitle(_title, y=1.00)
        fig.tight_layout()
        return fig
    
    def _plot3(self,*args,t=None,h=2.5,max_node=44,**kwargs):
        T,N = self.f_tr.shape
        # if t is None: t = range(T)
        if t is None: t = 30
        fig = plt.figure()
        nof_axs = max(min(N,max_node),2)
        if min(N,max_node)<2: 
            print('max_node should be >=2')
        ax = fig.subplots(nof_axs ,1)
        for n in range(nof_axs):
            # ax[n].plot(t,self.f_tr[:,n],color='gray',*args,**kwargs)
            ax[n].plot(np.array(self.test_dataset.targets)[:t,n], color='gray',*args,**kwargs)
            # ax[n].set_title('node='+str(n))
            ax[n].set_title(str(n))
        fig.set_figheight(nof_axs*h)
        fig.tight_layout()
        plt.close()
        return fig
    
    def test_plot(self,*args,t=None,h=2.5,**kwargs):
        self.calculate_mse()
        fig = self._plot3(*args,t=None,h=2.5,**kwargs)
        ax = fig.get_axes()
        for i,a in enumerate(ax):
            _mse1= self.mse['train']['each_node'][i]
            _mse2= self.mse['test']['each_node'][i]
            _title = 'node: {0}, \n mse(train) = {1:.2f}, mse(test) = {2:.2f}'.format(i+1, _mse1, _mse2)
            a.set_title(_title)
            a.plot(range(t),self.yhat_test[:t,i],label='fitted (test)',color='C1')
            a.legend()
        _mse1= self.mse['train']['total']
        _mse2= self.mse['test']['total']
        _mse3= self.mse['test(base)']['total']
        _title =\
        'dataset: {0} \n method: {1} \n epochs={2} \n number of filters={3} \n lags = {4} \n mse(train) = {5:.2f}, mse(test) = {6:.2f}, mse(test_base) = {7:.2f} \n'.\
        format(self.learner.dataset_name,self.learner.method,self.learner.epochs,self.learner.nof_filters,self.learner.lags,_mse1,_mse2,_mse3)
        fig.suptitle(_title)
        fig.suptitle(_title, y=0.999)
        fig.tight_layout()
        return fig