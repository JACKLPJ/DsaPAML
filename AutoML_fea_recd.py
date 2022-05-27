from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import VotingClassifier
#from sklearn.ensemble import StackingClassifier
import time
from DsaPAML.DualAutoEncoder4_norm import DualAutoEncoder
from DsaPAML import SklearnModels_copy as sm
from DsaPAML import DataModeling_sub as dm
#from multiprocessing import Pool
#import multiprocessing
#from functools import partial
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from DsaPAML.mlxtend1.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import os 

'''
X:数据集的特征
y:数据集的标签
N:前N个best model
data_pre_processing:是否需要数据预处理 
'''
class Automl:
    #time_per_model=360

    def __init__(self, N=200, 
                 verbose=False,
                 #DoEnsembel=True, 
                 #data_pre_processing=False, 
                 time_per_model=360,                
                 N_jobs=-1,
                 system='linux'
                 #address='./metaKB/'#'/media/sia1/Elements SE/TRAIN/'
#                  address_model_metafeatures='/media/sia1/Elements SE/TRAIN/model_samples.csv', 
#                  address_data_metafeatures='/media/sia1/Elements SE/TRAIN/data_samples.csv',
#                  address_pipelines='/media/sia1/Elements SE/TRAIN/New_pipelines.json', 
                 ):
        #DualAutoEncoder_model_4, gpuStackedDualAutoEncoder_model7, DSDAE_model,DualAutoEncoder_new
        path = os.path.realpath(os.curdir)#获取当前目录的绝对路径
      #  path = os.path.join(path, "1.txt")#加上文件名
        #print(path)
        self.address=os.path.join(path, "DsaPAML/RequiredBases/")
        self.address_model_metafeatures = self.address+'EX_model_samples1.csv'#address_model_metafeatures
        self.address_data_metafeatures=self.address+'EX_data_samples1.csv'#_data_metafeatures
        self.address_pipeline = self.address+'New_pipelines1.json'#address_pipelines
        self.verbose=verbose
        self.DoEnsembel = True#DoEnsembel
        self.y = []
        #sm.time_per_model = time1
        self.ensemble_clf = []
        self.k=20
        
        self.N = N
        self.n_estimators = 200
        self.scaler = StandardScaler()
        self.data_pre_processing = False#data_pre_processing  
        self.time_per_model = time_per_model
        self.N_jobs = N_jobs
        # if system=='linux':
        #     multiprocessing.set_start_method('forkserver',force=True)
    
        self.enc = OrdinalEncoder()
        
    def fit(self, Xtrain, ytrain):
        X = Xtrain.copy(deep=True)
        y = ytrain.copy(deep=True)
        X_ = Xtrain.copy(deep=True)
        y_ = ytrain.copy(deep=True)
        if self.N < 15:
            raise ValueError('N must be more than 15')
        self.y = ytrain.copy(deep=True)
#         preprocessing_dics, model_dics = dm.data_modeling(
#             X, y, self.N).result
    
        preprocessing_dics, model_dics = dm.data_modeling(
            X_, y_, self.k, self.N, self.address_model_metafeatures, self.address_data_metafeatures, self.address_pipeline, self.address, self.verbose).result  #_preprocessor
        self.cats, self.nums = [], []
        Xtype=list(X.dtypes)
        col_num = len(Xtype)
        #self.flag = False
        for i in range(col_num):
            if Xtype[i]=='O' or Xtype[i]=='object':
                self.cats.append(i)
            else:
                self.nums.append(i)
        if self.cats and self.nums:
            X_train_some = X.iloc[:, self.cats]
            X_train_some = pd.DataFrame(self.enc.fit_transform(X_train_some))#.toarray())
            X_train_others = X.iloc[:, self.nums]
            X_train_others=X_train_others.reset_index(drop=True)# = pd.DataFrame(X_train.iloc[:,nums]) 
            X = pd.concat([X_train_some, X_train_others],axis=1)
        elif self.cats:
            X = pd.DataFrame(self.enc.fit_transform(X))
        X = pd.DataFrame(self.scaler.fit_transform(X))
        self.poly = None
        if col_num <= 5:
            self.poly = PolynomialFeatures(3)
            X=pd.DataFrame(self.poly.fit_transform(X))
        elif col_num <= 15:
            self.poly = PolynomialFeatures(2)
            X=pd.DataFrame(self.poly.fit_transform(X))
        t_fs = time.perf_counter()
        self.rfecv = RFECV(estimator=RandomForestClassifier(n_jobs = 1),          # 学习器
              min_features_to_select=X.shape[1]//5*3, # 最小选择的特征数量
              step=max(1, X.shape[1]//25),                 # 移除特征个数
              cv=StratifiedKFold(2),  # 交叉验证次数
              scoring='accuracy',     # 学习器的评价标准
              verbose = 1 if self.verbose else 0,
              n_jobs = 1
              ).fit(X, y)
        X = pd.DataFrame(self.rfecv.transform(X))
        if self.verbose:
            print("The time for features selection is: {}".format(time.perf_counter() -
                                                      t_fs))
            print('#######################################')
        n = len(preprocessing_dics)
        try:
            if self.y.iloc[:,0].dtypes == 'object' or self.y.iloc[:,0].dtypes == 'O':
                labels = list(y.unique())
                y = y.replace(labels, list(range(len(labels))))
        except:
            if self.y.dtypes == 'object' or self.y.dtypes == 'O':
                labels = list(y.unique())
                y = y.replace(labels, list(range(len(labels))))
        y = y.astype('int')
        accuracy = []
        great_models = []
        Y_hat=[]
        model_name = []
       # self.rxc = X.shape[0] * X.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25,
                                                        random_state=0)

        td = time.perf_counter()

        for i in range(n):
            if model_dics[i][0] == 'xgradient_boosting':
                if X.shape[0] >= 10 ** 5:#self.rxc > 10 ** 7:
                    try:
                        Str, Clf, acc,y_hat_sub = sm.LGB(X_train,
                                                X_test,
                                                y_train,
                                                y_test,
                                                model_dics[i],
                                                self.data_pre_processing,
                                                preprocessing_dics[i])
                    except:
                        acc = -1  
                else:
                    try:
                        Str, Clf, acc,y_hat_sub = sm.XGB(X_train,
                                                X_test,
                                                y_train,
                                                y_test,
                                                model_dics[i],
                                                self.data_pre_processing,
                                                preprocessing_dics[i])
                    except:
                        acc = -1               
                    
            elif model_dics[i][0] == 'gradient_boosting':
                if X.shape[0] < 10 ** 5:
                    try:
                        Str, Clf, acc,y_hat_sub = sm.GradientBoosting(X_train,
                                                X_test,
                                                y_train,
                                                y_test,
                                                model_dics[i],
                                                self.data_pre_processing,
                                                preprocessing_dics[i])
                    except:
                        acc = -1    
                else:
                    continue

            elif model_dics[i][0] == 'lda':
                try:
                    Str, Clf, acc,y_hat_sub = sm.LDA(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'extra_trees':
                try:
                    Str, Clf, acc,y_hat_sub = sm.ExtraTrees(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'random_forest':
                try:
                    Str, Clf, acc,y_hat_sub = sm.RandomForest(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'decision_tree':
                try:
                    Str, Clf, acc,y_hat_sub = sm.DecisionTree(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'libsvm_svc':
                try:
                    Str, Clf, acc,y_hat_sub = sm.SVM(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'k_nearest_neighbors':
                try:
                    Str, Clf, acc,y_hat_sub = sm.KNN(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'bernoulli_nb':
                try:
                    Str, Clf, acc,y_hat_sub = sm.BernoulliNB(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'multinomial_nb':
                try:
                    Str, Clf, acc,y_hat_sub = sm.MultinomialNB(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            else:
                try:
                    Str, Clf, acc,y_hat_sub = sm.QDA(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1
            if acc > -1:
                accuracy.append(acc)
                great_models.append(Clf)
                Y_hat.append(y_hat_sub)
                model_name.append(Str)

        Y_hat=np.array(Y_hat)
        sort_id = sorted(range(len(accuracy)),
                         key=lambda m: accuracy[m],
                         reverse=True)
        mean_acc = np.mean(accuracy)#np.median(accuracy)#
        #mean_f1 = np.mean(f1_scores)
        estimators_stacking = []  #[great_models[sort_id[0]]]
        #X_val_predictions = [all_results[sort_id[0]][-1]]
        id_n = len(sort_id)
        id_i = 0
        base_acc_s = [] 
        pre=[]
        while accuracy[sort_id[id_i]] > mean_acc: 
            pre.append(sort_id[id_i])
            id_i += 1
        
        Y_hat=Y_hat[pre]
        n_pre=len(Y_hat)
         
        res_=[] 
        Sort=[] 
        td = time.perf_counter()
        for i in range(n_pre):
            aa=self.Sum_diff(i,n_pre,Y_hat)
            res_.append(aa[0])
            Sort.append(aa[1])

        if self.verbose:
            print('The time of pools2 is: {}'.format(time.perf_counter() -
                                                      td))
        c = sorted(range(len(Sort)), key=lambda k: Sort[k])
        res_ = np.array(res_)[c]
        
        Rubbish=set()
        
        final=[]
        for i in range(n_pre):
            if i not in Rubbish:
                final.append(pre[i])
                for k in range(len(res_[i])):
                    if res_[i][k] == 0: 
                        Rubbish.add(i+k+1)
        
        #print(final)
        if len(final)==1:
            self.DoEnsembel=False
        estimators_stacking=[great_models[i] for i in final]#.append(great_models[sort_id0[id_i]])
        base_acc_s=[accuracy[i] for i in final]#.append(accuracy[sort_id0[id_i]])
        
       # print(self.imbalance)#, fa)
        if self.verbose:
            print(id_n, len(base_acc_s))
            print(base_acc_s, mean_acc)
        if self.DoEnsembel:
            te = time.perf_counter()
            meta_clf = RandomForestClassifier(n_jobs=1,
                                              n_estimators=self.n_estimators)
            
            eclf_stacking = StackingClassifier(classifiers=estimators_stacking,
                                               meta_classifier=meta_clf,
                                               use_probas=True,
                                               #preprocessing=self.data_pre_processing,
                                               fit_base_estimators=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            #accuracy.append(
            eclf_stacking = eclf_stacking.fit(X_train, y_train)
            if self.verbose:
                print('Ensemble val score:',
                  accuracy_score(y_test, eclf_stacking.predict(X_test)))
            self.ensemble_clf = [estimators_stacking, eclf_stacking]
            if self.verbose:
                print('The time of ensemble is: {}'.format(time.perf_counter() -
                                                       te))
            #print(self.ensemble_clf)
            #return meta_clf
        else:

            self.clf = [model_name[sort_id[0]], great_models[sort_id[0]]]
            if self.verbose:
                print(self.clf)
            #allresult = [great_models[sort_id[0]], accuracy[sort_id[0]]]
            return self
        
    def Sum_diff(self,i,n,Y_hat):
        res=[]
        for j in range(i+1,n):
            res.append(np.sum(Y_hat[i]!=Y_hat[j])) 
        return [res,i]

    def predict(self, Xtest):
        X_Test = Xtest.copy(deep=True)    
        if self.cats and self.nums:
            X_test_some = X_Test.iloc[:,self.cats]
            X_test_some = pd.DataFrame(self.enc.transform(X_test_some))#.toarray())
            X_test_others = X_Test.iloc[:, self.nums]
            X_test_others=X_test_others.reset_index(drop=True)
            X_Test = pd.concat([X_test_some, X_test_others],axis=1) 
        elif self.cats:
            X_Test = pd.DataFrame(self.enc.transform(X_Test))
        X_Test=pd.DataFrame(self.scaler.transform(X_Test))
        if self.poly:
            X_Test=pd.DataFrame(self.poly.transform(X_Test)) 
        #X_Test = self.select_fea.transform(X_Test)
        X_Test = pd.DataFrame(self.rfecv.transform(X_Test))
        #X_Test = self.pre_processing_X(X_Test)
        if self.DoEnsembel:
            # X_test_predictions = self.scaler.transform(X_test_predictions)
            ypre = self.ensemble_clf[1].predict(X_Test)
        else:
            if self.clf[0] == 'mnb':
                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()
                X_Test = min_max_scaler.fit_transform(X_Test)
            if self.data_pre_processing:
                X_Test=sm.Preprocessing(X_Test, self.clf[1][1])
           # t = time.perf_counter()            
            ypre = self.clf[1][0].predict(X_Test)
        try:
            if self.y.iloc[:,0].dtypes == 'object' or self.y.iloc[:,0].dtypes == 'O':
                b = self.y.iloc[:,0].unique()
                return [b[i] for i in ypre]
        except:
            if self.y.dtypes == 'object' or self.y.dtypes == 'O':
                b = self.y.unique()
                return [b[i] for i in ypre]
        return ypre
    
    def predict_proba(self, Xtest):
        X_Test = Xtest.copy(deep=True)
        X_Test = self.pre_processing_X(X_Test)
        if self.DoEnsembel:
            # X_test_predictions = self.scaler.transform(X_test_predictions)
            ypre = self.ensemble_clf[1].predict_proba(X_Test)
        else:
            if self.clf[0] == 'mnb':
                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()
                X_Test = min_max_scaler.fit_transform(X_Test)
            if self.data_pre_processing:
                X_Test=sm.Preprocessing(X_Test, self.clf[1][1])
           # t = time.perf_counter()
            
            ypre = self.clf[1][0].predict_proba(X_Test)
        
        return ypre
