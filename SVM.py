import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def train(self, train, label, epoch, alpha, verbose):
        feature_size = len(train[0]) 
        data_size = len(train)
        lambda_v = 1/epoch

        self.weight = np.ones(feature_size)
        
        
        for e in range(epoch):
            
            if verbose==1:
                sum = 0
                for i in range(feature_size):
                    sum += max(0,1-label[i]*(np.dot(train[i],self.weight))) #max(0,y<x,w>)
                loss = lambda_v*pow(np.linalg.norm(self.weight),2)/2 + sum
                
                print("Epoch {}/{}: Loss is {}".format(e+1,epoch,loss))
            
            for i in range(data_size):
                p = label[i] * (np.dot(train[i],self.weight)) # y<x,w>
    
                if p >= 0.5: #prediction is correct
                    self.weight = self.weight - alpha*(2*lambda_v*self.weight)
                else:
                    self.weight = self.weight + alpha*(label[i]*train[i] - 2*lambda_v*self.weight)
                
    def predict(self,test,target):
        
        res = []
        
        for t in test:
            res.append(np.dot(t,self.weight))

        count = 0

        for i in range(len(target)):
            if (target[i] * res[i] > 0):
                count += 1

        return res,count/len(target)











