import numpy as np
import matplotlib.pyplot as plt
import itertools
import collections

class calcMetric(object):
    
    def calcOverfit(train_acc,test_acc,_count):
        overfit_factor = train_acc-test_acc
        if   overfit_factor > 0.25:
            _count +=4
            return _count
        if   overfit_factor > 0.15:
            _count+=3
            return _count
        if   overfit_factor > 0.10:
            _count+=2
            return _count
        elif overfit_factor > 0.05:
            _count+=1
            return _count
        else :
            _count=0
            return _count
    
    def pre_rec_fs2(cm):
        # precision values # recall values # F-score values
        if (cm[0,0]!=0):
            pr0=round(cm[0,0]/(cm[0,0]+cm[1,0]),4) # sum ver 
            rec0=round(cm[0,0]/(cm[0,0]+cm[0,1]),4) # sum hor  
            Fscore0=round(2*pr0*rec0/(pr0+rec0),4)
        else :
            pr0=0
            rec0=0
            Fscore0=0
            
        if (cm[1,1]!=0):
            pr1=round(cm[1,1]/(cm[0,1]+cm[1,1]),4)  
            rec1=round(cm[1,1]/(cm[1,0]+cm[1,1]),4)  
            Fscore1=round(2*pr1*rec1/(pr1+rec1),4)  
        else :
            pr1=0
            rec1=0
            Fscore1=0
        
        return np.array([[pr0,rec0,Fscore0],[pr1,rec1,Fscore1]])
    
    
    def pre_rec_fs3(cm):
        # precision values # recall values # f-score values
        if (cm[0,0]!=0):
            pr0=round(cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]),4) # sum ver 
            rec0=round(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]),4) # sum hor
            Fscore0=round(2*pr0*rec0/(pr0+rec0),4)  
        else :
            pr0=0
            rec0=0
            Fscore0=0
            
        if (cm[1,1]!=0):
            pr1=round(cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]),4)  
            rec1=round(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]),4) 
            Fscore1=round(2*pr1*rec1/(pr1+rec1),4)
        else :
            pr1 =0
            rec1 =0
            Fscore1=0
            
        if (cm[2,2]!=0):
            pr2=round(cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]),4)  
            rec2=round(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]),4)
            Fscore2=round(2*pr2*rec2/(pr2+rec2),4) 
        else :
            pr2 =0
            rec2=0
            Fscore2=0

        return np.array([[pr0,rec0,Fscore0],[pr1,rec1,Fscore1],[pr2,rec2,Fscore2]])
    
    def pre_rec_fs4(cm):
        # precision values # recall values # f-score values
        if (cm[0,0]!=0):
            pr0=round(cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]+cm[3,0]),4) # sum ver 
            rec0=round(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]),4) # sum hor  
            Fscore0=round(2*pr0*rec0/(pr0+rec0),4)
        else :
            pr0=0
            rec=0
            Fscore0=0
        
        if (cm[1,1]!=0):
            pr1=round(cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]+cm[3,1]),4)    
            rec1=round(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]),4) 
            Fscore1=round(2*pr1*rec1/(pr1+rec1),4)
        else :
            pr1=0
            rec1=0
            Fscore1=0
            
        if (cm[2,2]!=0):
            pr2=round(cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]+cm[3,2]),4)  
            rec2=round(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]),4) 
            Fscore2=round(2*pr2*rec2/(pr2+rec2),4)
        else :
            pr2=0
            rec2=0
            Fscore2=0
        
        if (cm[3,3]!=0):
            pr3=round(cm[3,3]/(cm[0,3]+cm[1,3]+cm[2,3]+cm[3,3]),4)  
            rec3=round(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]),4)  
            Fscore3=round(2*pr3*rec3/(pr3+rec3),4) 
       
        return np.array([[pr0,rec0,Fscore0],[pr1,rec1,Fscore1],[pr2,rec2,Fscore2],[pr3,rec3,Fscore3]])
    
    def pre_rec_fs5(cm):
        # precision values # recall values # f-score values
        if (cm[0,0]!=0):
            pr0=round(cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]+cm[3,0]+cm[4,0]),4) # sum ver 
            rec0=round(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]),4) # sum hor  
            Fscore0=round(2*pr0*rec0/(pr0+rec0),4)
        else :
            pr0=0
            rec0=0
            Fscore0=0
        
        if (cm[1,1]!=0):
            pr1=round(cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]+cm[3,1]+cm[4,1]),4) 
            rec1=round(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]),4)  
            Fscore1=round(2*pr1*rec1/(pr1+rec1),4)  
        else :
            pr1=0
            rec1=0
            Fscore1=0
        
        if (cm[2,2]!=0):
            pr2=round(cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]+cm[3,2]+cm[4,2]),4)    
            rec2=round(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]),4)  
            Fscore2=round(2*pr2*rec2/(pr2+rec2),4)
        else :
            pr2=0
            rec2=0
            Fscore2=0
        
        if (cm[3,3]!=0):
            pr3=round(cm[3,3]/(cm[0,3]+cm[1,3]+cm[2,3]+cm[3,3]+cm[4,3]),4)   
            rec3=round(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]),4) 
            Fscore3=round(2*pr3*rec3/(pr3+rec3),4)
        else :
            pr3=0
            rec3=0
            Fscore3=0
        
        if (cm[4,4]!=0):
            pr4=round(cm[4,4]/(cm[0,4]+cm[1,4]+cm[2,4]+cm[3,4]+cm[4,4]),4)   
            rec4=round(cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]),4) 
            Fscore4=round(2*pr4*rec4/(pr4+rec4),4)
        else :
            pr4=0
            rec4=0
            Fscore4=0
            
        return np.array([[pr0,rec0,Fscore0],[pr1,rec1,Fscore1],[pr2,rec2,Fscore2],[pr3,rec3,Fscore3],[pr4,rec4,Fscore4]])

    def pre_rec_fs6(cm):
        # precision values # recall values # f-score values
        if (cm[0,0]!=0):
            pr0=round(cm[0,0]/(cm[0,0]+cm[1,0]+cm[2,0]+cm[3,0]+cm[4,0]+cm[5,0]),4) # sum ver 
            rec0=round(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]+cm[0,5]),4) # sum hor  
            Fscore0=round(2*pr0*rec0/(pr0+rec0),4)
        else :
            pr0=0
            rec0=0
            Fscore0=0
        
        if (cm[1,1]!=0):
            pr1=round(cm[1,1]/(cm[0,1]+cm[1,1]+cm[2,1]+cm[3,1]+cm[4,1]+cm[5,1]),4) 
            rec1=round(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]+cm[1,5]),4)  
            Fscore1=round(2*pr1*rec1/(pr1+rec1),4)  
        else :
            pr1=0
            rec1=0
            Fscore1=0
        
        if (cm[2,2]!=0):
            pr2=round(cm[2,2]/(cm[0,2]+cm[1,2]+cm[2,2]+cm[3,2]+cm[4,2]+cm[5,2]),4)    
            rec2=round(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]+cm[2,5]),4)  
            Fscore2=round(2*pr2*rec2/(pr2+rec2),4)
        else :
            pr2=0
            rec2=0
            Fscore2=0
        
        if (cm[3,3]!=0):
            pr3=round(cm[3,3]/(cm[0,3]+cm[1,3]+cm[2,3]+cm[3,3]+cm[4,3]+cm[5,3]),4)   
            rec3=round(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]+cm[3,5]),4) 
            Fscore3=round(2*pr3*rec3/(pr3+rec3),4)
        else :
            pr3=0
            rec3=0
            Fscore3=0
        
        if (cm[4,4]!=0):
            pr4=round(cm[4,4]/(cm[0,4]+cm[1,4]+cm[2,4]+cm[3,4]+cm[4,4]+cm[5,4]),4)   
            rec4=round(cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]+cm[4,5]),4) 
            Fscore4=round(2*pr4*rec4/(pr4+rec4),4)
        else :
            pr4=0
            rec4=0
            Fscore4=0
            
        if (cm[5,5]!=0):
            pr5=round(cm[5,5]/(cm[0,5]+cm[1,5]+cm[2,5]+cm[3,5]+cm[4,5]+cm[5,5]),4)   
            rec5=round(cm[5,5]/(cm[5,0]+cm[5,1]+cm[5,2]+cm[5,3]+cm[5,4]+cm[5,5]),4) 
            Fscore5=round(2*pr5*rec5/(pr5+rec5),4)
        else :
            pr5=0
            rec5=0
            Fscore5=0
                
        return np.array([[pr0,rec0,Fscore0],[pr1,rec1,Fscore1],[pr2,rec2,Fscore2],[pr3,rec3,Fscore3],[pr4,rec4,Fscore4],[pr5,rec5,Fscore5]])

    def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title,fontsize=16)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45,fontsize=16)
        plt.yticks(tick_marks, classes,fontsize=16)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm =np.matrix.round(cm,4)*100
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix')
    
        print(cm)
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center", fontsize=16,
                    color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label',fontsize=16)
        plt.xlabel('Predicted label',fontsize=16)
        plt.show()     
    
    
    def plot_test_set(test_label,filename):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        labels = ["Neutral","Positive", "Negative"]
        explode = list()
        for k in labels:
            explode.append(0.05)
        plt.pie([np.count_nonzero(test_label[:,1] == 1), np.count_nonzero(test_label[:,2] == 1),np.count_nonzero(test_label[:,0] == 1)],explode=explode, shadow=True, labels=labels,autopct='%1.1f%%')
        
        plt.title(filename[:len(filename)-5] + ' Test-set',fontsize=20)
        plt.show()
