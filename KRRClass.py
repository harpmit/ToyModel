from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("always")
import copy

class Krr():
    #Class containing methods for a standard implementation of RF/RFA krr, RF cutoff krr, or krr without feature selection
    #Defaults to RF/RFA krr, other feature selection modes can be specified by passing arguments into self.fit()
    #Usage should be:
    #   1. load_data
    #   2. fit
    #This class handles eliminating uniformative features, standardizing data, and feature selection
    #The class has default settings for the list of alphas and gammas to use in grid search
        #But, these are not always sufficient (and we wouldn't expect them to be)
        #Alternate lists of values for alpha and gamma can be specified as arguments to the fit() function
        
    def __init__(self):
        self.train_features = None
        self.train_selected_features=None
        self.train_values = None
        self.test_features = None
        self.test_selected_features=None
        self.test_values = None
        
        self.feature_scaler = preprocessing.StandardScaler()
        self.label_scaler = preprocessing.StandardScaler()
        
        self.informative_features = None
        self.selected_features = None
        
        self.rfa_results = None
        
        self.results = None
        self.regressor = None
    
    ##########################################################
    #### The first block of methods are for loading data #####
    # The user should only be using load_data from the block #
    ##########################################################
    
    def __load_train_features(self,np_array):
        #Load the features for the training data into the model
        #Features should be columns and datapoints should be rows
        
        self.informative_features = check_informative_features(np_array)[1]
        reduced_features = np_array.T[self.informative_features].T
        
        self.feature_scaler.fit(reduced_features)
        self.train_features = self.feature_scaler.transform(reduced_features)
        
    def __load_train_labels(self,np_array):
        #Load the labels for the training data
        #Should be a 2D np.array formatted as a column vector (shape = (x,1) where x is the number of datapoints)
        self.label_scaler.fit(np_array)
        self.train_values = self.label_scaler.transform(np_array)
        
    def __load_test_features(self,np_array):
        #Load the features for the test set
        #Features should be columns and datapoints should be rows
        #should be called AFTER loading the training set features
        
        self.test_features = self.feature_scaler.transform(np_array.T[self.informative_features].T)
        
    def __load_test_labels(self,np_array):
        #Load the labels for the test set
        #Should be a 2D np.array formatted as a column vector (shape = (x,1) where x is the number of datapoints)
        #Should be called AFTER loading the training set labels
        
        # ~ if type(self.train_values) == type(None):
            # ~ raise Exception('Warning, load the training set labels before test set labels!')
        self.test_values = self.label_scaler.transform(np_array)
        
    def load_data(self,train_features,train_labels,test_features,test_labels):
        #Load data into the model. This function will automatically eliminate uniformative (constant features) and standardize the features
        #All data should be input as a numpy arrary
            #The shape of feautre arrays should be: (number of data points,number of features)
                #In other words, features are columns
            #The shape of label arrays should be: (number of data points,1)
                #They should always be 2d arrays, not 1d
        
        self.__load_train_features(train_features)
        self.__load_train_labels(train_labels)
        self.__load_test_features(test_features)
        self.__load_test_labels(test_labels)

    ####################################################################
    ###### The second block of methods are for feature selection #######
    #### The user should access these by passing arguments to fit() ####
    ####################################################################
    
    def __cutoff_feature_selection(self,debug = False,cutoff=0.01):
        #Perform feature selection base on a RF importance cuttoff
        if debug:
            print('Ranking feature importance with random forests')
        regr = RandomForestRegressor(random_state=0,n_estimators=100)
        one_dimensional_train = copy.copy(self.train_values)
        one_dimensional_train.shape =(one_dimensional_train.shape[0],)
        regr.fit(self.train_features,one_dimensional_train)
        importances = regr.feature_importances_
        
        best_features = np.where(importances >= cutoff,True,False)
        
        
        self.train_selected_features = self.train_features.T[best_features].T
        self.test_selected_features = self.test_features.T[best_features].T
        
        self.selected_features = best_features
        
    def __no_feature_selection(self):
        self.train_selected_features = self.train_features
        self.test_selected_features = self.test_features
        self.selected_features = [True]*self.train_selected_features.shape[1]
        
    def __RF_RFA(self,debug = False, CV_fold = 5, alphas=[.1,1,10], gammas=[.1,1,10],show_warnings = True):
        #Perform feature selection by using random-forests ranked recursive feature addition
        #Sets attributes recording the best features and the details of this rfa run: self.selected_features and self.rfa_results
        #Also sets the self.test_selected_features and self.train_selected_features attributes
        if debug:
            print('Ranking feature importance with random forests')
        regr = RandomForestRegressor(random_state=0,n_estimators=100)
        one_dimensional_train = copy.copy(self.train_values)
        one_dimensional_train.shape =(one_dimensional_train.shape[0],)
        regr.fit(self.train_features,one_dimensional_train)
        importances = regr.feature_importances_
        ranked = importances.tolist()
        ranked.sort()
        ranked.reverse()
        
        if debug:
            print('Preparing recursive feature addition')
        #Create a list indexes corresponding to the ranked order
        ranked_order = []
        for i in ranked:
            new_feature = np.where(abs(importances - i) < 10**(-12),True,False)
            new_feature_index = np.array(range(len(ranked)))[new_feature][0]
            ranked_order.append(new_feature_index)
        features_to_evaluate = copy.copy(ranked_order[1:])
        
        ######Setup for krr##########
        def check_feature_set(feature_set,alphas=alphas,gammas=gammas,debug=debug,CV_fold = CV_fold):
            small_features = self.train_features.T[feature_set].T #Select only a few features for this round of KRR
            small_test_features = self.test_features.T[feature_set].T
            number_of_features.append(small_features.shape[1])

            #Do the KRR
            tuned_parameters = [{'kernel': ['rbf'], 'gamma' : gammas, 'alpha': alphas}]
            clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=CV_fold,iid=True, scoring = 'neg_mean_absolute_error')
            clf.fit(small_features,self.train_values)
            
            train_predictions = clf.predict(small_features)
            train_predictions = self.label_scaler.inverse_transform(train_predictions)    
            train_error = np.absolute(train_predictions - self.label_scaler.inverse_transform(self.train_values))
            train_squared_error = train_error**2
            train_MUE = np.mean(train_error)
            train_MSE = np.mean(train_squared_error)
                
            test_predictions = clf.predict(small_test_features)
            test_predictions = self.label_scaler.inverse_transform(test_predictions)    
            test_error = np.absolute(test_predictions - self.label_scaler.inverse_transform(self.test_values))
            test_squared_error = test_error**2
            test_MUE = np.mean(test_error)
            test_MSE = np.mean(test_squared_error)
            
            warning = []
            if clf.best_params_['alpha'] == max(alphas) or clf.best_params_['alpha'] == min(alphas):
                warning.append('Alpha value of: '+str(clf.best_params_['alpha'])+' indicates that a more exhaustive grid search is required in RF-RFA')
            if clf.best_params_['gamma'] == max(gammas) or clf.best_params_['gamma'] == min(gammas):
                warning.append('Gamma value of: '+str(clf.best_params_['gamma'])+' indicates that a more exhaustive grid search is required in RF-RFA')
            
            return train_MUE,test_MUE,train_MSE,test_MSE,clf.best_params_,clf.cv_results_,clf.best_score_,warning
        
        #Evaluate KRR models, recursively adding features one at a time
        number_of_features = []
        feature_addition_test_MUEs = []
        feature_addition_train_MUEs = []
        feature_addition_test_MSEs = []
        feature_addition_train_MSEs = []
        best_params = []
        cv_table = []
        cv_scores = []
        feature_set_history = []
        
        #The first feature is always kept
        keep = [ranked_order[0]]
        indexer = convert_to_indexer(keep,self.train_features.shape[1])
        train_MUE,test_MUE,train_MSE,test_MSE,best_params_,cv_results_,cv_score_,warning = check_feature_set(indexer)
        
        number_of_features.append(len(keep))
        feature_addition_test_MUEs.append(test_MUE)
        feature_addition_train_MUEs.append(train_MUE)
        feature_addition_test_MSEs.append(test_MSE)
        feature_addition_train_MSEs.append(train_MSE)
        best_params.append(best_params_)
        cv_table.append(cv_results_)
        cv_scores.append(cv_score_)
        feature_set_history.append(indexer)
        
        #Now determine whether to keep additional features
        warning = []
        added_a_feature = True
        while added_a_feature:
            added_a_feature = False
            if debug:
                print(str(len(keep))+' feature(s) added to set with score: '+str(cv_score_))
            for counter,i in enumerate(features_to_evaluate):
                indexer = convert_to_indexer(keep+[i],self.train_features.shape[1])
                train_MUE,test_MUE,train_MSE,test_MSE,best_params_,cv_results_,cv_score_,new_warning = check_feature_set(indexer)
                warning.extend(new_warning)
                if debug:
                    if (counter+1)%10 == 0 or counter == (len(features_to_evaluate)-1):
                        print('    Trying to add a feature...attempt '+str(counter+1)+'. Found CV score = '+str(cv_score_))
                
                if cv_score_ > max(cv_scores): #Scoring is defined as the negative of mean absolute error. We want largest possible values, which are those closes to zero
                    
                    for warn in new_warning:
                        if show_warnings:
                            warnings.warn(warn)
                    
                    keep.append(i)
                    del features_to_evaluate[counter]
                    
                    added_a_feature=True
                    
                    number_of_features.append(len(keep))
                    feature_addition_test_MUEs.append(test_MUE)
                    feature_addition_train_MUEs.append(train_MUE)
                    feature_addition_test_MSEs.append(test_MSE)
                    feature_addition_train_MSEs.append(train_MSE)
                    best_params.append(best_params_)
                    cv_table.append(cv_results_)
                    cv_scores.append(cv_score_)
                    feature_set_history.append(indexer)
                    break
        
        if debug:
            print('Feature selection complete with '+str(len(keep))+' features retained')
        
        best_features = convert_to_indexer(keep,self.train_features.shape[1])
                
        rfa_detailed_results = {'number_of_features':number_of_features,
                                'train_MUEs':feature_addition_train_MUEs,
                                'train_MSEs':feature_addition_train_MSEs,
                                'test_MUEs':feature_addition_test_MUEs,
                                'test_MSEs':feature_addition_test_MSEs,
                                'best_params':best_params,
                                'cv_results':cv_table,
                                'cv_scores':cv_scores,
                                'feature_set_history':feature_set_history}
        
        self.train_selected_features = self.train_features.T[best_features].T
        self.test_selected_features = self.test_features.T[best_features].T
        
        self.selected_features = best_features
        self.rfa_results = rfa_detailed_results
        
        return warning
    


    #############################################################################################
    ###### The third block of methods are for fitting data and evaluating additional data #######
    #################### The user should access all of these methods directly ###################
    #############################################################################################
    
    alphas = []
    gammas = []
    for loop in range(10):
        gammas.append(1e-8*10**loop)     
        alphas.append(1e-8*10**loop)
    def fit(self,debug=False,cutoff=None,skip_feature_selection = False, alphas=alphas, gammas=gammas, CV_fold = 10, show_warnings = True):
        #fit the final model with feature selected features
        #Evaluate error on test set and record results as a dictionary
        #If a random forests cutoff is not specified, default to rfa-rfa
        
        warning = []
        if skip_feature_selection:
            self.__no_feature_selection()
        elif not cutoff:
            warning = self.__RF_RFA(debug=debug, alphas=alphas, gammas=gammas, CV_fold = CV_fold, show_warnings = show_warnings)
        elif cutoff:
            self.__cutoff_feature_selection(debug=debug,cutoff=cutoff)
        else:
            raise ValueError('Something went wrong in choosing a feature selection method')
            
        tuned_parameters = [{'kernel': ['rbf'], 'gamma' : gammas, 'alpha': alphas}]
        clf = GridSearchCV(KernelRidge(), tuned_parameters, cv=CV_fold,iid=True)
        clf.fit(self.train_selected_features,self.train_values)
        
        if clf.best_params_['alpha'] == max(alphas) or clf.best_params_['alpha'] == min(alphas):
            warning.append('FINAL MODEL BAD: '+'Alpha value of: '+str(clf.best_params_['alpha'])+' indicates that a more exhaustive grid search is required')
            warnings.warn('FINAL MODEL BAD: '+'Alpha value of: '+str(clf.best_params_['alpha'])+' indicates that a more exhaustive grid search is required')
        if clf.best_params_['gamma'] == max(gammas) or clf.best_params_['gamma'] == min(gammas):
            warning.append('FINAL MODEL BAD: '+ 'Gamma value of: '+str(clf.best_params_['gamma'])+' indicates that a more exhaustive grid search is required')
            warnings.warn('FINAL MODEL BAD: '+ 'Gamma value of: '+str(clf.best_params_['gamma'])+' indicates that a more exhaustive grid search is required')
        
        #Record a bunch of data about the krr run
        train_predictions = clf.predict(self.train_selected_features)
        train_predictions = self.label_scaler.inverse_transform(train_predictions)
        test_predictions = clf.predict(self.test_selected_features)
        test_predictions = self.label_scaler.inverse_transform(test_predictions)  
    
        train_errors = np.absolute(train_predictions - self.label_scaler.inverse_transform(self.train_values))
        train_squared_errors = train_errors**2
        train_MUE = np.mean(train_errors)
        train_MSE = np.mean(train_squared_errors)
        
        test_errors = np.absolute(test_predictions - self.label_scaler.inverse_transform(self.test_values))
        test_squared_errors = test_errors**2
        test_MUE = np.mean(test_errors)
        test_MSE = np.mean(test_squared_errors)
        
        self.results = {'Test_MUE':test_MUE,
                        'Test_MSE':test_MSE,
                        'Train_MUE':train_MUE,
                        'Train_MSE':train_MSE,
                        'best_params':clf.best_params_,
                        'cv_results':clf.cv_results_,
                        'train_predictions':train_predictions,
                        'train_labels':self.label_scaler.inverse_transform(self.train_values),
                        'test_predictions':test_predictions,
                        'test_labels':self.label_scaler.inverse_transform(self.test_values)}
                        
        self.regressor = clf
        
        return warning
        
    def predict(self,np_array):
        #Takes a np_array of the raw features, scales them, and makes a prediction for the property of interst
        
        scaled_features = np_array.T[self.informative_features].T
        scaled_features = self.feature_scaler.transform(scaled_features)
        scaled_features = scaled_features.T[self.selected_features].T
        unscaled_predictions = self.regressor.predict(scaled_features)
        return self.label_scaler.inverse_transform(unscaled_predictions)
        
    def evaluate(self,features,labels):
        #Takes a set of (raw) features and the corresponding labels as nxd and nx1 np arrays, respectively
        #Returns the model's MUE on this data set
        
        return np.mean(np.abs(self.predict(features)-labels))


###################################################
# Additional helper functions called by the class #
###################################################

def check_informative_features(np_array):
    #Takes an np array containing the features for all datapoints
    #Features should be columns and data points should be rows
    #Returns a new np array which removes any constant features and a list of booleans detailing which features are constant
    def check_constant_array(np_array):
        #Checks to see if a 1-D array contains all identical values, returns a boolean
        constant = True
        value_reference_exists = False
        length = np_array.shape[0]
        for i in range(length):
            value = np_array[i]
            if not value_reference_exists:
                value_reference = value
                value_reference_exists = True
            else:
                difference = abs(value-value_reference)
                if difference > 10**(-10):
                    constant = False
        return constant
            
    constant = []    
    columns = np_array.shape[1]
    for i in range(columns):
        column = np_array[:,i]
        constant.append(check_constant_array(column))
    is_informative = ~np.array(constant)
    removed_non_informative = (np_array.T[is_informative]).T
    
    return removed_non_informative,is_informative
    
def convert_to_indexer(keep,size):
    #returns an indexer of a specified size where the entries in keep (a list) are retained
    indexer = [False]*size
    for i in keep:
        indexer[i] = True
    return indexer
