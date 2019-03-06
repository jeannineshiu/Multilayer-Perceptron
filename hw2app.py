#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.setrecursionlimit(10000)
import os
from PyQt5.QtWidgets import *
from designerUI03 import Ui_Form    #MyFirstUI 是你的.py檔案名字

#for openfile
from PyQt5.QtCore import *
from PyQt5.QtGui import *

#drawing
import random
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap


# In[2]:


class NeuralNetMLP(object):
    
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        #original
        #"""
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T
        #"""
        
    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
    
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_samples, n_classlabels]
                sigma_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)

                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)
            
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self


# In[3]:


class AppWindow(QWidget):
    learning_rate = 0
    iters_num = 0
    hidden_n = 0
    l2 = 0
    batch_min = 0
    
    mynn = NeuralNetMLP(30,0,100,0.001,True,1,None)
    
    realtr_n = 0
    
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        #1.open file
        self.ui.openbt.clicked.connect(self.openbt_Click)
        #2.input learn,iters....
        self.ui.learninginput.editingFinished.connect( self.enterPress1 )
        self.ui.iterationinput.editingFinished.connect( self.enterPress2 )
        self.ui.hiddeninput.editingFinished.connect( self.enterPress3 )
        self.ui.l2input.editingFinished.connect( self.enterPress4 )
        self.ui.batchinput.editingFinished.connect( self.enterPress5 )
         
        #3.run
        self.ui.train_bt.clicked.connect(self.trainbt_Click)
        self.ui.test_bt.clicked.connect(self.testbt_Click)
        
        self.show()
        
    ###define my functions which i want to do when i touch the objects######
        
    def enterPress1(self):
        self.learning_rate = float(self.ui.learninginput.text())
        #print("learning="+str(self.learning_rate))
    def enterPress2(self):
        self.iters_num = int(self.ui.iterationinput.text())
        #print("iters="+str(self.iters_num))
    def enterPress3(self):
        self.hidden_n = int(self.ui.hiddeninput.text())
        #print("hidden="+str(self.hidden_n))
    def enterPress4(self):
        self.l2 = float(self.ui.l2input.text())
        #print("l2="+str(self.l2))
    def enterPress5(self):
        self.batch_min = int(self.ui.batchinput.text())
        #print("batch="+str(self.batch_min))
        
        
    def trainbt_Click(self):
        
        self.ui.datawidget.canvas.ax.clear()
        #train network, draw cost , accuracy
        self.mlptrain(self.X_train,self.y_train,self.learning_rate,self.iters_num,self.hidden_n,self.l2,self.batch_min)
        #draw data 
        self.plot_decision_regions(self.X_train, self.y_train, self.mynn)
        #self.ui.datawidget.canvas.ax.title('NeuralNetMLP_train')
        
        
    def testbt_Click(self):
        
        self.ui.datawidget.canvas.ax.clear()
        
        y_test_pred = self.mynn.predict(self.X_test)
        acc = (np.sum(self.y_test == y_test_pred)
               .astype(np.float) / self.X_test.shape[0])
        #show accuracy
        self.ui.test_acc_show.setText('Test accuracy: %.2f%%' % (acc * 100))
        #print('Test accuracy: %.2f%%' % (acc * 100))
        
        #draw data
        self.plot_decision_regions(self.X_test, self.y_test, self.mynn)
        #self.ui.datawidget.canvas.ax.title('NeuralNetMLP_test')
        
        #clear other canvas
        self.ui.costwidget.canvas.ax.clear()
        self.ui.accuracywidget.canvas.ax.clear()
        
    def openbt_Click(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setFilter( QDir.Files  )
        if dlg.exec_():
            filenames= dlg.selectedFiles()
            f = open(filenames[0], 'r') 
            #show filename
            filename = os.path.basename(filenames[0])
            self.ui.filenamelabel.setText(filename)
            
            with f:
                #process_data while reading
                data2d = []
                i = 0
                line = f.readline()
                while line!='':
                    datalist = [float(x) for x in line.split(' ')]
                    #print (datalist) => [-0.912312, 2.056434, 1.0] 
                    #datalist[0]:-0.912312
                    data2d.append([])
                    data2d[i] = datalist
                    print(data2d[i])
                    i = i + 1
                    line = f.readline()

            ##process data
            self.X_train,self.y_train,self.X_test,self.y_test = self.processData(data2d)
            #print(self.X_train,self.y_train,self.X_test,self.y_test)
            
            
    def plot_decision_regions(self,X,y,classifier,resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        
        
        
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        self.ui.datawidget.canvas.ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        self.ui.datawidget.canvas.ax.set_xlim(xx1.min(), xx1.max())
        self.ui.datawidget.canvas.ax.set_ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            self.ui.datawidget.canvas.ax.scatter(x=X[y == cl, 0], 
                        y=X[y == cl, 1],
                        alpha=0.8, 
                        c=colors[idx],
                        marker=markers[idx], 
                        label=cl, 
                        edgecolor='black')
            
        self.ui.datawidget.canvas.ax.set_xlabel('x1')
        self.ui.datawidget.canvas.ax.set_ylabel('x2')
        self.ui.datawidget.canvas.ax.legend(loc='upper left')

        #self.ui.datawidget.canvas.ax.tight_layout()
        self.ui.datawidget.canvas.draw()
            
        
    def processData(self,data2d):
        
        #用numpy來處理data
        X = np.array(data2d)
        #成功洗牌X
        np.random.shuffle(X)
        
        #把target2改成0以符合sigmoid的設定
        X_c = X
        for i in range(X.shape[0]):
            if X_c[i][2]==2.0:
                X_c[i][2] = 0.0
        #print(X_c)
        X = X_c
        
        
        #計算前2/3的資料有幾row
        trainRow = int(X.shape[0]*(2/3))
        #print(trainRow)
        testRow = X.shape[0]-trainRow
        #print(testRow)
        
        #成功將data2d分成訓練和測試資料
        training, test = X[:trainRow,:], X[trainRow:,:]
        
        
        
        #成功將data與target分開
        y_train = training[ : , 2:3 ]
        X_train = training[ : , :2 ]


        y_test = test[ : , 2:3 ]
        X_test = test[ : , :2 ]
        
        
        #二維轉一維（應該有別的方法reshape?）
        yy_train = []
        for i in range(y_train.shape[0]):
            yy_train.append(y_train[i][0])
        y_train = yy_train
        #用numpy來處理data
        y_train = np.array(y_train)
        #print(y_train)

        yy_test = []
        for i in range(y_test.shape[0]):
            yy_test.append(y_test[i][0])
        y_test = yy_test
        #用numpy來處理data
        y_test = np.array(y_test)
        #print(y_test)
        
        val_n = round(X_train.shape[0]*(1/8))
        self.realtr_n = X_train.shape[0] - val_n
        
        return X_train,y_train,X_test,y_test
    
    
    def mlptrain(self,X_train,y_train,learning_rate,iters_num,hidden_n,l2,batch_min): 

        #run!!!
        nn = NeuralNetMLP(n_hidden=hidden_n, 
                          l2=l2, 
                          epochs=iters_num, 
                          eta=learning_rate,
                          minibatch_size=batch_min, 
                          shuffle=True,
                          seed=1)

        nn.fit(X_train=X_train[:self.realtr_n], 
               y_train=y_train[:self.realtr_n],
               X_valid=X_train[self.realtr_n:],
               y_valid=y_train[self.realtr_n:])
        
        self.mynn = nn
        
        #draw cost
        self.ui.costwidget.canvas.ax.clear()
        self.ui.costwidget.canvas.ax.plot(range(nn.epochs), nn.eval_['cost'])
        self.ui.costwidget.canvas.ax.set_ylabel('Cost')
        self.ui.costwidget.canvas.ax.set_xlabel('Epochs')
        self.ui.costwidget.canvas.draw()
        
        #draw accuracy
        self.ui.accuracywidget.canvas.ax.clear()
        self.ui.accuracywidget.canvas.ax.plot(range(nn.epochs), nn.eval_['train_acc'], 
         label='training')
        self.ui.accuracywidget.canvas.ax.plot(range(nn.epochs), nn.eval_['valid_acc'], 
                 label='validation', linestyle='--')
        self.ui.accuracywidget.canvas.ax.set_ylabel('Accuracy')
        self.ui.accuracywidget.canvas.ax.set_xlabel('Epochs')
        self.ui.accuracywidget.canvas.ax.legend()
        self.ui.accuracywidget.canvas.draw()
        
        


# In[11]:





# In[4]:


## main function
if __name__ == "__main__":
    app = QApplication(sys.argv)  
    MainWindow = AppWindow()
    MainWindow.show()
    sys.exit(app.exec_())    

