from DataClass import Data
from KRRClass import Krr
import groupplot as grp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import copy
import time

class ActiveLearner:

    def __init__(self, data_path=None, learning_objectives=['solvent_10_3','IP'], target_values=[1.,1.],
                 model_type='ANN', feature_type='RACs', extra_features=[],
                 random_seed = None, verbose=False):
        #data path should be a string,specifying where to load the data from
        #learning objectives should be a list of strings specifying the properies to learn
            #current valid options are: IP, EA, solvent 10.3, solvent 2.3, solvent 6.2, solvent 78.9
        #target values should be a list of floats specifying (in Hartrees) the ideal value for the property
        #model type should be a string. Either "ANN" or "KRR"
        #random_seed should be an integer, if specied
        #verbose should be a boolean
        #feature_type should be a string. Either "RACs" or (to be continues)
        #extra features allows additional values to be included in the featurization that aren't explicitly racs. It should be a list
        # Possible options include:
            #1. ox
            #2. charge

        self.data_path = data_path
        self.learning_objectives = learning_objectives
        self.target_values = target_values
        self.model_type = model_type
        self.feature_type = feature_type
        self.extra_features = extra_features
        self.random_seed = random_seed
        self.verbose = verbose

        self.model = None #Attribute which holds the current generation's ML model(s)
        self.data = None #Attribute which holds all data related to the current toy problem
        self.dropped = None #pd.DataFrame for points which are excluded from this toy problem. see 'reason' column
        self.features = None #features for all TMCs included in the current toy problem
        self.labels = None #labels for all TMCs included in the current toy problem
        self.predictions = None #values predicted by the ML model
        self.uncertainties = None #uncertainty values for each prediciton (symettric). set to the train mae for KRR models
        self.visible_maes = None #Mean average error for the prediction of visible data
        self.invisibile_maes = None #Mean avearage error for the prediction of invisible data

        self.visible_idxs = None #idxs of complexes which are visible to the current model
        self.visible_features = None #features for the complexes which are visible to the current model
        self.visible_labels = None #lables for the complexes which are visible to the current model

        self.invisible_idxs = None #idxs of complexes which are withheld from the current model
        self.invisible_features = None
        self.invisible_labels = None

        self.pareto_idxs = None #idxs of the complexes which are on the pareto front
        self.all_points_pareto_idxs = None #pareto front including invisible points

        self.old_gens = [] #copies of this active learner object as it was in previous generations

    def initialize(self):
        #Loads in data and the model class objects for this Active Learner
        #eliminates cases where the labels are unvailable for a given complex (i.e. are np.nan)
        if self.feature_type == 'RACs':
            apply_racs_filter = True
        else:
            apply_racs_filter = False
        if type(self.learning_objectives) != list or type(self.target_values) != list:
            raise Exception('learning_objectives and target_values are not specified as lists!')
        if len(self.learning_objectives) != len(self.target_values):
            raise Exception('learning_objectives and target_values are not the same length!')


        self.__get_new_model()
        self.__load_data(apply_racs_filter=apply_racs_filter)

    def __get_new_model(self):
        if self.model_type == 'ANN':
            raise Exception('ANN needs to be implemented, but is not yet available')
        elif self.model_type == 'KRR':
            self.model = [Krr()]*len(self.learning_objectives)
        else:
            raise Exception(self.model_type+' is not implemented as a possible model type in ActiveLearner')

    def __load_data(self,apply_racs_filter=False):
        #initializes a new dataclass object and binds it to thes self.data attribute
        if self.verbose:
            print('---- Loading Data and Applying Geometry and Spin Checks ----')
        self.data=Data(self.data_path,verbose=self.verbose,apply_racs_filter=apply_racs_filter,extra_features=self.extra_features)
        if self.verbose:
            print('Finished loading Data')

        if self.feature_type == 'RACs':
            self.features = self.data.racs
        else:
            raise Exception(self.feature_type+' is not fully implemented!')

        labels = []
        for objective in self.learning_objectives:
            if objective == 'IP':
                labels.append(self.data.ip)
            elif objective == 'EA':
                labels.append(self.data.ea)
            elif objective == 'Solvent 10.3':
                labels.append(self.data.solvent_10_3)
            elif objective == 'Solvent 2.3':
                labels.append(self.data.solvent_2_3)
            elif objective == 'Solvent 6.2':
                labels.append(self.data.solvent_6_2)
            elif objective == 'Solvent 78.9':
                labels.append(self.data.solvent_78_9)
            else:
                raise Excpetion(objective+' is not fully implemented!')
        labels = [i.reshape(i.shape[0],1) for i in labels]
        self.labels = np.concatenate(labels,axis=1)

        if self.verbose:
            print('---- checking for missing labels now ----')
            print('Initial number of points: '+str(self.features.shape[0]))
        good_labels,bad_labels = [],[]
        all_labels = np.arange(0,self.labels.shape[0])
        for i in all_labels:
            if not np.any(np.isnan(self.labels[i,:])):
                good_labels.append(i)
            else:
                bad_labels.append(i)

        self.labels = self.labels[good_labels]
        self.features = self.features[good_labels]
        if self.verbose:
            print('After dropping cases where a label is missing: '+str(self.features.shape[0]))
            print('----- Eliminated complexes due to missing labels -----')

        self.dropped = self.data.dropped
        new_dropped = self.data.data.iloc[bad_labels].copy()
        new_dropped['reason'] = 'missing label'
        self.dropped = pd.concat([self.dropped,new_dropped],axis=0)

    def setup_gen_0(self,init_fraction,
                    gen_0_selection_method='random'):
        #init fraction should be a float between 0 and 1
            #it describes the fraction of overall points which are included in visible_points

        all_idxs = np.arange(0,self.features.shape[0])
        if gen_0_selection_method == 'random':
            np.random.seed(self.random_seed)
            self.visible_idxs = np.random.choice(all_idxs,int(init_fraction*self.features.shape[0]),replace=False)
        else:
            raise Exception('The gen_0_selection_method: '+str(gen_0_selection_method)+' is not implemented!')
        self.invisible_idxs = np.array(list(set(all_idxs)-set(self.visible_idxs)))


        self.visible_features = self.features[self.visible_idxs]
        self.visible_labels = self.labels[self.visible_idxs]

        self.invisible_features = self.features[self.invisible_idxs]
        self.invisible_labels = self.labels[self.invisible_idxs]

        #populate the pareto_idxs attribute

        self.__find_pareto_front(include_invisible=True)
        self.__find_pareto_front()

    def __find_pareto_front(self,include_invisible = False):
        #finds the overall pareto front, flexible to specific target values and arbitrary number of dimensions
        #if include_invisible=False, considers only visible points and sets the self.pareto_idxs attribute
        #if include_invisible=True, considers all points and set the self.all_points_pareto_idxs attribute

        #the quadrants variable should be a list of lists with lengths 2**n where n is the number of objectives
        #each entry in the quardants list is length n and composed of -1 and 1
        #a negative 1 corresponds to points which are less than the optimal point on that axis
        #a positive 1 corresponds to points which are greater than the optimal point on that axis
        quadrants = [[]]
        for quad_idx in range(self.labels.shape[1]):
            new_quadrants = []
            for sub_quad in quadrants:
                new_quadrants.append(sub_quad+[1])
                new_quadrants.append(sub_quad+[-1])
            quadrants = new_quadrants
        if self.verbose:
            print('---- Find the pareto front ----')
            print('---- Dividing property space into '+str(len(quadrants))+' based on '+str(len(self.target_values))+' objectives ----')

        #break the visible points up into 2**n quadrants, finding a pareto front within each quadrant
        #reorient each quadrant so that the problem is equivalent to maximizing all objectives
        #pass the reoriented points into __find pareto_Front_quadrant by quadrant
        pareto_idxs = []
        for quad in quadrants:
            if self.verbose:
                print('--- Finding pareto points in a quadrant ---')
            if include_invisible:
                in_quadrant_labels = self.labels.copy()
                in_quadrant_idxs = np.arange(0,self.features.shape[0])
            else:
                in_quadrant_labels = self.visible_labels.copy()
                in_quadrant_idxs = self.visible_idxs.copy()
            for col in range(in_quadrant_labels.shape[1]):
                if quad[col] == -1:
                    filterer = [True if row[col] < self.target_values[col] else False for row in in_quadrant_labels]
                elif quad[col] == 1:
                    filterer = [False if row[col] < self.target_values[col] else True for row in in_quadrant_labels]
                else:
                    raise Exception('bad quadrant value')
                in_quadrant_labels[:,col] = in_quadrant_labels[:,col]*-1*quad[col]
                in_quadrant_labels = in_quadrant_labels[filterer]
                in_quadrant_idxs = in_quadrant_idxs[filterer]

            relative_pareto_idxs = self.__find_pareto_front_quardrant(in_quadrant_labels) #relative to the in_quadrant_labels array
            new_pareto_idxs = in_quadrant_idxs[relative_pareto_idxs].tolist()

            #find the best way to stitch together the pareto idxs to minimize long connections (for graphing only)
            def find_distance_between_idx(idx1,idx2):
                coord1 = self.labels[idx1]
                coord2 = self.labels[idx2]
                distances = coord2-coord1
                return np.min(np.abs(distances))

            if len(pareto_idxs) > 1 and len(new_pareto_idxs) > 1:
                reverse = [False,False]
                min_distance = find_distance_between_idx(pareto_idxs[-1],new_pareto_idxs[0])
                if find_distance_between_idx(pareto_idxs[-1],new_pareto_idxs[-1]) < min_distance:
                    min_distance = find_distance_between_idx(pareto_idxs[-1],new_pareto_idxs[-1])
                    reverse = [False,True]
                elif find_distance_between_idx(pareto_idxs[0],new_pareto_idxs[0]) < min_distance:
                    min_distance = find_distance_between_idx(pareto_idxs[0],new_pareto_idxs[0])
                    reverse = [True,False]
                elif find_distance_between_idx(pareto_idxs[0],new_pareto_idxs[-1]) < min_distance:
                    min_distance = find_distance_between_idx(pareto_idxs[0],new_pareto_idxs[-1])
                    reverse = [True,True]
                if reverse[0]:
                    pareto_idxs = pareto_idxs[::-1]
                if reverse[-1]:
                    new_pareto_idxs = new_pareto_idxs[::-1]
                pareto_idxs = pareto_idxs+new_pareto_idxs
            else:
                pareto_idxs = pareto_idxs+new_pareto_idxs

        if self.verbose:
            print('--- ' +str(len(pareto_idxs))+' Pareto points found ---')
        if include_invisible:
            self.all_points_pareto_idxs = np.array(pareto_idxs)
        else:
            self.pareto_idxs = np.array(pareto_idxs)

    def __find_pareto_front_quardrant(self,labels):
        #finds the pareto front oriented toward maximizing all quantities
        #acts on the np.array "labels"
        #returns a list of indexes for points to include in the array
        def remove_row_from_array(array,row_idx):
            return np.concatenate([array[:row_idx,:],array[row_idx+1:,:]],axis=0)
        def remove_entry_from_list(lst,entry_idx):
            return lst[:entry_idx]+lst[entry_idx+1:]

        pareto_idxs = []
        ranked_idxs = np.flip(np.argsort(labels[:,0])).tolist()
        possible_pareto_points = labels.copy()
        possible_pareto_points = possible_pareto_points[ranked_idxs]
        while possible_pareto_points.shape[0] > 0:
            current_pareto_values = possible_pareto_points[0,:].copy()
            pareto_idxs.append(ranked_idxs[0])
            ranked_idxs = remove_entry_from_list(ranked_idxs,0)
            possible_pareto_points = remove_row_from_array(possible_pareto_points,0)

            entries_to_remove = []
            for row in range(possible_pareto_points.shape[0]):
                row_values = possible_pareto_points[row,:]
                keep_row = False
                for value,pareto_value in zip(row_values,current_pareto_values):
                    if value > pareto_value: #must beat the pareto front in 1 dimension to be retained
                        keep_row = True
                if not keep_row:
                    entries_to_remove.append(row)
            entries_to_remove.reverse()
            for row in entries_to_remove:
                ranked_idxs = remove_entry_from_list(ranked_idxs,row)
                possible_pareto_points = remove_row_from_array(possible_pareto_points,row)
        return pareto_idxs


    def fit(self):
        #fits a model and populates the self.predictions attribute

        #Fit the model(s) and make predictions
        if type(self.model) == list:
            predictions = []
            for idx in range(self.labels.shape[1]):
                if self.verbose:
                    print('Fitting model '+str(idx+1)+'/'+str(self.labels.shape[1])+'...')

                visible_labels = self.visible_labels[:,idx].reshape(self.visible_labels.shape[0],1)
                invisible_labels = self.invisible_labels[:,idx].reshape(self.invisible_labels.shape[0],1)

                self.model[idx].load_data(self.visible_features,visible_labels,
                                          self.invisible_features,invisible_labels)
                self.model[idx].fit(cutoff=0.01)

                predictions.append(self.model[idx].predict(self.features))
            self.predictions = np.concatenate(predictions,axis=1)


        else:
            if self.verbose:
                print('Fitting model...')
            self.model.load_data(self.visible_features,self.visible_labels,
                                 self.invisible_features,self.invisible_labels)
            self.model.fit(cutoff=0.01)
            self.predictions = self.model.predict(self.features)

        #Calculate the maes for the visible and invisible points
        self.visible_maes,self.invisible_maes = [],[]
        for idx in range(self.predictions.shape[1]):
            visible_errors = self.labels[self.visible_idxs][:,idx]-self.predictions[self.visible_idxs][:,idx]
            self.visible_maes.append(np.mean(np.abs(visible_errors)))
            invisible_errors = self.labels[self.invisible_idxs][:,idx]-self.predictions[self.invisible_idxs][:,idx]
            self.invisible_maes.append(np.mean(np.abs(invisible_errors)))
        self.visible_maes,self.invisible_maes = np.array(self.visible_maes),np.array(self.invisible_maes)

        #Calculate the uncertainties for each point
        if self.model_type == 'KRR':
            self.uncertainties = np.zeros(self.labels.shape)
            for idx in range(self.uncertainties.shape[1]):
                self.uncertainties[:,idx] = self.uncertainties[:,idx]+self.visible_maes[idx]
        else:
            raise Excpetion('uncertainties are not implemented for: '+self.model_type)


    def copy(self,active_learner):

        self.data_path = copy.copy(active_learner.data_path)
        self.learning_objectives = copy.copy(active_learner.learning_objectives)
        self.target_values = copy.copy(active_learner.target_values)
        self.model_type = copy.copy(active_learner.model_type)
        self.feature_type = copy.copy(active_learner.feature_type)
        self.extra_features = copy.copy(active_learner.extra_features)
        self.random_seed = copy.copy(active_learner.random_seed)
        self.verbose = copy.copy(active_learner.verbose)

        self.model = active_learner.model #Attribute which holds the current generation's ML model(s)
        self.data = active_learner.data #Attribute which holds all data related to the current toy problem
        self.dropped = active_learner.dropped.copy()
        self.features = active_learner.features.copy() #features for all TMCs included in the current toy problem
        self.labels = active_learner.labels.copy() #labels for all TMCs included in the current toy problem
        self.predictions = active_learner.predictions.copy()
        self.uncertainties = active_learner.uncertainties.copy() #uncertainty values for each prediciton (symettric). set to the train mae for KRR models
        self.visible_maes = active_learner.visible_maes.copy() #Mean average error for the prediction of visible data
        self.invisible_maes = active_learner.invisible_maes.copy() #Mean avearage error for the prediction of invisible data

        self.visible_idxs = active_learner.visible_idxs.copy() #idxs of complexes which are visible to the current model
        self.visible_features = active_learner.visible_features.copy() #features for the complexes which are visible to the current model
        self.visible_labels = active_learner.visible_labels.copy() #lables for the complexes which are visible to the current model

        self.invisible_idxs = active_learner.invisible_idxs.copy() #idxs of complexes which are withheld from the current model
        self.invisible_features = active_learner.invisible_features.copy()
        self.invisible_labels = active_learner.invisible_labels.copy()

        self.pareto_idxs = active_learner.pareto_idxs.copy()
        self.all_points_pareto_idxs = active_learner.all_points_pareto_idxs.copy()

        self.old_gens = active_learner.old_gens #copies of this active learner object as it was in previous generations

    def advance_generation(self, selection_size=10,
                           selection_strategy='EGO'):
        #Selects new points to add to the visible points
        #Saves the sate of the model to old_models
        #Rests appropriate attributes to "nonde"

        old_learner = ActiveLearner()
        old_learner.copy(self)
        self.old_gens.append(old_learner)

        if selection_strategy == 'PI-EGO':
            total_points = self.invisible_idxs.shape
            improvement_probablities = []
            for idx in self.invisible_idxs:
                improvement_probablities.append(self.find_PI(idx))
            sorter = np.argsort(improvement_probablities).tolist()
            sorter.reverse()
            sorted_invisible_idxs = self.invisible_idxs[sorter]
            new_visible_idxs = sorted_invisible_idxs[:selection_size]

        elif selection_strategy.capitalize() == 'Best':
            #selects the points which appear to be the best candidates, ignoring uncertainty
            invisible_predictions = self.predictions[self.invisible_idxs]
            distances_from_goal = [] #euclidean distances with all properties in Ha
            for point in invisible_predictions:
                distances_from_goal.append(np.linalg.norm(point-np.array(self.target_values)))
            sorter = np.argsort(distances_from_goal)
            sorted_invisible_idxs = self.invisible_idxs[sorter]
            new_visible_idxs = sorted_invisible_idxs[:selection_size]

        elif selection_strategy.capitalize() == 'Random':
            #selects new points randomly
            np.random.seed(self.random_seed)
            new_visible_idxs = np.random.choice(self.invisible_idxs,selection_size,replace=False)

        else:
            raise Exception('Selection strategy: '+str(selection_strategy)+' not implemented')

        if self.verbose:
            print('---- Adding new points to visible points ---')
            print(new_visible_idxs)

        self.visible_idxs = np.array(self.visible_idxs.tolist()+new_visible_idxs.tolist())
        self.invisible_idxs = np.array(list(set(self.invisible_idxs)-set(new_visible_idxs)))

        self.visible_features = self.features[self.visible_idxs]
        self.visible_labels = self.labels[self.visible_idxs]
        self.invisible_features = self.features[self.invisible_idxs]
        self.invisible_labels = self.labels[self.invisible_idxs]

        self.__find_pareto_front()

        self.predictions = None
        self.uncertainties = None
        self.visible_maes = None
        self.invisibile_maes = None

    def numeric_gaussian(self,centers,variances,points_per_dimension=10,print_timings = False):
        #makes a numerical represenation of a guassian, as a numpy array respresenting a grid
        #centers should be a list of floats describing the center of the gaussian in each dimension
        #variances should be a list of floats describing the variance of the gaussian along each dimension
        #the lengths of centers and variances should match
        #returns a tuple of three objects
            #1. a numpy array representing the gaussian as a grid. # of dimesions matches the length of centers and variances
            #2. a numpy array with boolean values. True indicates that this point on the grid is in the dominated region. Matches shape with array 1
            #3. a float, indicating the volume/area/4-d equivalent of each point on the grid, for integration
        #this function is implemented for 2-4 objectives. It can be extended to more, but may get unacceptably slow
        if print_timings:
            t0 = time.time()
        if len(centers) < 2 or len(centers) > 4 or len(centers) != len(variances):
            raise Exception('Numeric gaussian can not be consstructed')

        #the gaussian will start 4 standard deviations away from the center, spanning over 8 total standard deviations
        #this captures .9999 of the gaussian's integral
        starts = [float(i)-4*ii for i,ii in zip(centers,variances)]
        ends = [float(i)+4*ii for i,ii in zip(centers,variances)]
        widths = [float(i)*8/points_per_dimension for i in variances]#width of each point along each axis
        volume_per_point = widths[0]
        if len(widths) > 1:
            for i in widths[1:]:
                volume_per_point = volume_per_point*i

        #Assemble the gaussian function grid
        dims = [] #dimensions
        dim1 = (np.array(range(points_per_dimension))*widths[0])+starts[0]
        dim2 = (np.array(range(points_per_dimension))*widths[1])+starts[1]

        if len(starts) <= 2:
            mesh = np.meshgrid(dim1,dim2)
        if len(starts) > 2 and len(starts) <= 3:
            dim3 = (np.array(range(points_per_dimension))*widths[2])+starts[2]
            mesh = np.meshgrid(dim1,dim2,dim3)
        if len(starts) > 3:
            dim3 = (np.array(range(points_per_dimension))*widths[2])+starts[2]
            dim4 = (np.array(range(points_per_dimension))*widths[3])+starts[3]
            mesh = np.meshgrid(dim1,dim2,dim3,dim4)

        prefactor = (2.*np.pi)**(float(len(mesh))/2.)
        for var in variances:
            prefactor = prefactor*var
        prefactor = 1/prefactor

        exponential_term = np.zeros(len(centers)*[points_per_dimension])
        for dim in range(len(mesh)):
            exponential_term = exponential_term + ((mesh[dim]-centers[dim])/variances[dim])**2
        exponential_term = (-.5*exponential_term)
        gaussian = prefactor*np.exp(exponential_term)

        if print_timings:
            print('Time to assemble numeric gaussian: '+str(time.time()-t0)+' seconds')
            t0 = time.time()

        #Assemble th grid of booleans specifying if a point is dominated or not
        new_shape = list(mesh[0].shape)+[1]
        lifted_mesh = [i.reshape(new_shape) for i in mesh]
        lifted_mesh = np.concatenate(lifted_mesh,axis=len(new_shape)-1)
        dominated_booleans = np.apply_along_axis(self.__is_dominated,(len(new_shape)-1),lifted_mesh)

        if print_timings:
            print('Time to identify dominated region: '+str(time.time()-t0)+' seconds')

        return gaussian,dominated_booleans,volume_per_point

    def __is_dominated(self,coords):
        #Returns true if the coords are in the dominated region
        #returns false otherwise
        pareto_points = self.labels[self.pareto_idxs]
        for pareto in pareto_points:
            dominates = True
            for pareto_dim,point_dim,target in zip(pareto,coords,self.target_values):
                pareto_error = target - pareto_dim
                point_error = target - point_dim
                if pareto_error < 0 and point_error < 0: # to the right of the point
                    if abs(point_error) < abs(pareto_error):
                        dominates =  False
                elif pareto_error > 0 and point_error > 0: #to the left of the target piont
                    if abs(point_error) < abs(pareto_error):
                        dominates = False
                else:
                    dominates = False
            if dominates:
                return True
        return False

    def find_PI(self,idx):
        #find the probablity of improvement for a point with a given index
        prediction = self.predictions[idx]
        uncertainty = self.uncertainties[idx]
        gaussian,dominated_booleans,volume = self.numeric_gaussian(prediction,uncertainty)
        return np.sum(gaussian*~dominated_booleans*volume)

    def plot(self,ax,x_units = None, y_units = None,
             xlabel=None,ylabel=None,
             xticks=None,yticks=None,
             xlim=None,ylim=None):
        #takes a matplotlib axes object
        #returns the axes object with a plot of the current state of the optimization problem drawn over it
        #if uncertainty_point_dix is defined, 
        def convert_units(array,units):
            array = array.copy()
            if not units:
                return array
            elif units == 'eV':
                return array*27.2114
            elif units == 'kcal/mol':
                return array*627.509
            else:
                raise Exception('Cannot understand unit: '+str(units))

        visible = self.labels[self.visible_idxs]
        invisible = self.labels[self.invisible_idxs]
        pareto = self.labels[self.pareto_idxs]
        found_from_best_pareto = [i for i in self.pareto_idxs if i in self.all_points_pareto_idxs]
        found_from_best_pareto = self.labels[found_from_best_pareto]
        if len(self.old_gens)>0:
            new_visible_idxs = list(set(self.visible_idxs)-set(self.old_gens[-1].visible_idxs))
            new_points = self.labels[new_visible_idxs]
        else:
            new_points = self.labels[[]]
        target = np.array([self.target_values])

        if x_units:
            visible[:,0] = convert_units(visible[:,0],x_units)
            invisible[:,0] = convert_units(invisible[:,0],x_units)
            pareto[:,0] = convert_units(pareto[:,0],x_units)
            found_from_best_pareto[:,0] = convert_units(found_from_best_pareto[:,0],x_units)
            new_points[:,0] = convert_units(new_points[:,0],x_units)
            target[:,0] = convert_units(target[:,0],x_units)
        if y_units:
            visible[:,1] = convert_units(visible[:,1],y_units)
            invisible[:,1] = convert_units(invisible[:,1],y_units)
            pareto[:,1] = convert_units(pareto[:,1],y_units)
            found_from_best_pareto[:,1] = convert_units(found_from_best_pareto[:,1],y_units)
            new_points[:,1] = convert_units(new_points[:,1],y_units)
            target[:,1] = convert_units(target[:,1],y_units)
        markers = []
        # markers.append(ax.scatter(invisible[:,0],invisible[:,1],color='gray',alpha=.1,edgecolor='k'))
        markers.append(ax.scatter(visible[:,0],visible[:,1],color='gray',alpha=.1,edgecolor='k'))
        # markers.append(ax.scatter(new_points[:,0],new_points[:,1],color='green',alpha=1,edgecolor='k'))
        markers.append(ax.scatter(pareto[:,0],pareto[:,1],color='red',alpha=1,edgecolor='k'))
        markers.append(ax.scatter(found_from_best_pareto[:,0],found_from_best_pareto[:,1],color='blue',alpha=1,edgecolor='k'))
        markers.append(ax.scatter(target[:,0],target[:,1],
                    color='yellow',edgecolor='k',marker='*',s=100))

        #plot the pareto front
        def plot_pareto(coords1,coords2):
            possible1 = np.array([coords1[0],coords2[1]])
            possible2 = np.array([coords2[0],coords1[1]])
            target = self.target_values
            target_x = convert_units(np.array([target[0]]),x_units)
            target_y = convert_units(np.array([target[1]]),y_units)
            target = np.array([target_x[0],target_y[0]])
            if np.linalg.norm(possible1-target) < np.linalg.norm(possible2-target):
                best = possible2
            else:
                best = possible1
            ax.plot([coords1[0],best[0]],[coords1[1],best[1]],color='blue')
            ax.plot([coords2[0],best[0]],[coords2[1],best[1]],color='blue')

        pareto_coordinate_idxs = range(pareto.shape[0])
        for point1,point2 in zip(pareto_coordinate_idxs[1:],pareto_coordinate_idxs[:-1]):
            coords1 = pareto[point1]
            coords2 = pareto[point2]
            plot_pareto(coords1,coords2)
        coords1 = pareto[-1]
        coords2 = pareto[0]
        plot_pareto(coords1,coords2)


        grp.apply_std_format(ax,xlabel=xlabel,ylabel=ylabel,fontsize=10,
                             xticks=xticks,yticks=yticks)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        bold_font = {'family':'helvetica','weight':'bold','size':10}

        return ax,markers

    def plot_2d_gaussian(self,ax,coords,uncertainty,
                         x_units = None, y_units = None,
                         exclude_dominated_region=True,
                         points_per_dimension=100,
                         cmap='Blues'):

        #plots a gaussian onto an existing axes object
        #used best in combination with a self.plot
        coords = coords.copy()
        uncertainty = uncertainty.copy()

        cmap = cm.get_cmap(cmap,256)
        new_colors = cmap(np.linspace(0,1,256))
        white = np.array([1,1,1,1])
        new_colors[:5,:] = white
        cmap = ListedColormap(new_colors)

        if len(coords) != 2 or len(uncertainty) != 2:
            raise Exception('Bad arguments passed to self.plot_2d_gaussian')

        def convert_units(array,units):
            array = array.copy()
            if not units:
                return array
            elif units == 'eV':
                return array*27.2114
            elif units == 'kcal/mol':
                return array*627.509
            else:
                raise Exception('Cannot understand unit: '+str(units))

        gaussian,dominated,_ = self.numeric_gaussian(coords,uncertainty,points_per_dimension=points_per_dimension)
        if exclude_dominated_region:
            gaussian = gaussian*~dominated #zero out the points in the dominated region
        starts = [float(i)-4*ii for i,ii in zip(coords,uncertainty)]
        widths = [float(i)*8/points_per_dimension for i in uncertainty]
        dim1 = (np.array(range(points_per_dimension))*widths[0])+starts[0]
        dim2 = (np.array(range(points_per_dimension))*widths[1])+starts[1]
        dim1 = convert_units(dim1,x_units)
        dim2 = convert_units(dim2,y_units)
        ax.contourf(dim1,dim2,gaussian,100,cmap=cmap)










# AL = ActiveLearner('../raw_data/MD2-checked-unqiue.csv',['IP','Solvent 10.3'],[.5,-.35],
#                     model_type='KRR', feature_type='RACs', extra_features=['ox','charge'],
#                     verbose=True, random_seed=0)
# AL.initialize()
# AL.setup_gen_0(0.1,0.1)

# figure = plt.figure(figsize=(3.25,2.5))
# ax = figure.add_axes([.1,.1,.8,.8])
# AL.plot(ax,x_units='kcal/mol',y_units='eV',xlabel='Solvent 10.3 (kcal/mol)',ylabel='Vertical IP (eV)',
#         xlim=(100,500),ylim=(-30,10))
# plt.show()
# # AL.fit()




