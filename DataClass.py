import pandas as pd
import numpy as np

class Data:
    #Class created to hold data for the active learner class object
    #Takes the path to a csv file to initialize
    #Capable of holding a variety of featurizations for a given dataset simulatenously
    #Holds a variety of potential learning objectives simultaneously as well
    def __init__(self,data_path,verbose=False,apply_racs_filter=False,
                 extra_features=[]):
        #data path should be a string
        #verbose and apply rac filter should be booleans
        #apply rac filter is necessary for the rac featurization to be good (no np.nan's), but will interfere with other representations
        #extra fearues allows additional values to be included in the featurization that aren't explicitly racs
        # Possible options include:
            #1. ox
            #2. charge
        self.verbose = verbose

        #find the featurization definitions
        featurization_definitions = self.make_featurization_definitions(extra_features=extra_features)
        self.racs_labels = featurization_definitions['RACs']
        self.eracs_labels = featurization_definitions['eRACs']

        #load in the pertinent data from the csv (aside from feature vectors)
        self.data = pd.read_csv(data_path,na_values=['None','none',''])
        self.names = self.data['unique_name'].tolist()
        self.names = np.array([str(i) for i in self.names])
        self.tag = self.data['tag'].tolist()
        self.tag = np.array([str(i) for i in self.tag])
        self.subtag = self.data['subtag'].tolist()
        self.subtag = np.array([str(i) for i in self.subtag])
        self.converged = self.data['converged'].tolist()
        self.converged = [True if i =='True' else i for i in self.converged]
        self.converged = [False if i=='False' else i for i in self.converged]
        for i in self.converged:
            if type(i) != bool:
                raise Exception('Non boolean found in "converged" column: '+str(i))
        self.converged = np.array(self.converged)
        self.spin = self.data['spin'].tolist()
        self.spin = np.array([float(i) for i in self.spin])
        self.charge = self.data['charge'].tolist()
        self.charge = np.array([float(i) for i in self.charge])
        self.oxstate= self.data['ox'].tolist()
        self.oxstate = np.array([float(i) for i in self.oxstate])
        charge_spin_params = np.concatenate([self.charge.reshape(self.charge.shape[0],1),self.spin.reshape(self.spin.shape[0],1)],axis=1)
        self.check_all_floats(charge_spin_params)
        self.unfold_geo_check_metrics()

        #load in potential learning objectives
        self.solvent_10_3 = self.coerce_strings(self.data['solvent.10_3'].values)
        self.solvent_2_3 = self.coerce_strings(self.data['solvent.2_3'].values)
        self.solvent_6_2 = self.coerce_strings(self.data['solvent.6_2'].values)
        self.solvent_78_9 = self.coerce_strings(self.data['solvent.78_9'].values)
        self.ip = None
        self.ea = None
        self.load_ip_and_ea()
        self.ip = self.coerce_strings(self.ip)
        self.ea = self.coerce_strings(self.ea)

        #apply geo check
        tmp_dropped = self.pre_geometry_check()
        tmp_dropped['reason'] = 'pre-geocheck'
        self.dropped = tmp_dropped
        geo_params = self.data[['num_coord_metal','dist_del_eq','dist_del_all',
                                'max_del_sig_angle','oct_angle_devi_max',
                                'devi_linear_avrg','devi_linear_max','rmsd_max']].values
        self.check_all_floats(geo_params,exception_name='Geo params') #make sure all geo params are floats
        tmp_dropped = self.geometry_check()
        tmp_dropped['reason'] = 'geocheck'
        self.dropped = pd.concat([self.dropped,tmp_dropped],axis=0)

        #apply spin contamination check
        spin_check_params = self.data[['new_ss_act','new_ss_target']].values
        self.check_all_floats(spin_check_params,exception_name='Spin contamination check')
        tmp_dropped = self.spin_contamination_check()
        tmp_dropped['reason'] = 'spin_contamination_check'
        self.dropped = pd.concat([self.dropped,tmp_dropped],axis=0)

        #match other attributes to the newly trimmed dataset
        self.names = self.names[self.data.index]
        self.tag = self.tag[self.data.index]
        self.subtag = self.subtag[self.data.index]
        self.spin = self.spin[self.data.index]
        self.charge = self.charge[self.data.index]
        self.oxstate = self.oxstate[self.data.index]

        self.solvent_10_3 = self.solvent_10_3[self.data.index]
        self.solvent_2_3 = self.solvent_2_3[self.data.index]
        self.solvent_6_2 = self.solvent_6_2[self.data.index]
        self.solvent_78_9 = self.solvent_78_9[self.data.index]
        self.ip = self.ip[self.data.index]
        self.ea = self.ea[self.data.index]

        #Subset out feature sets of interest
        self.racs = self.data[self.racs_labels].values
        # self.eracs = self.raw_data[self.eracs_labels].values #NumB is not currently available in the data frame

        if apply_racs_filter: #RACs are known to contain some errors, this filters those cases out if requested.
            self.rac_filter = [] #racs failed on some complexes, so we need to exclude those
            for i in range(self.racs.shape[0]):
                test = self.racs[i,:]
                if not np.any(np.isnan(test)):
                    self.rac_filter.append(i)

            self.names = self.names[self.rac_filter]
            self.tag = self.tag[self.rac_filter]
            self.subtag = self.subtag[self.rac_filter]
            self.spin = self.spin[self.rac_filter]
            self.charge = self.charge[self.rac_filter]
            self.oxstate = self.oxstate[self.rac_filter]

            self.solvent_10_3 = self.solvent_10_3[self.rac_filter]
            self.solvent_2_3 = self.solvent_2_3[self.rac_filter]
            self.solvent_6_2 = self.solvent_6_2[self.rac_filter]
            self.solvent_78_9 = self.solvent_78_9[self.rac_filter]
            self.ip = self.ip[self.rac_filter]
            self.ea = self.ea[self.rac_filter]

            self.racs = self.racs[self.rac_filter]

        self.check_all_floats(self.racs,exception_name='RACs')
        # self.check_all_floats(self.eracs,exception_name='eRACs')

        self.eliminate_constant_features()



    def make_featurization_definitions(self,extra_features=[]):
        #Finds a list of features to include in each representation type
        #i.e. the features included in RACs-155 are different than eRACs
        #returns a dictionary containing lists of the names of the features in each representation (column-labels in the csv)
        featurization_definitions = {}

        ### Isovalent racs (eRACs)
        isovalent_racs = []
        for i in ['D_mc','D_lc','f','mc','lc']:
            if i == 'D_mc':
                properties = ['S','T','Z','chi','NumB','Zeff']
                depths = ['1','2','3']
            elif i == 'D_lc':
                properties = ['S','T','Z','chi','NumB','Zeff']
                depths = ['1','2','3']
            elif i == 'mc':
                properties = ['I','S','T','Z','chi','NumB','Zeff']
                depths = ['0','1','2','3']
            else:
                properties = ['I','S','T','Z','chi','NumB','Zeff']
                depths = ['0','1','2','3']
                
            if i == 'D_lc' or i == 'lc':
                destinations = ['ax','eq']
            elif i== 'D_mc' or i == 'mc':
                destinations = ['all']
            else:
                destinations = ['ax','eq','all']
                
            for ii in properties:
                for iii in depths:
                        for iiii in destinations:
                            isovalent_racs.append(i+'-'+ii+'-'+iii+'-'+iiii)

        # RACs
        racs = [i for i in isovalent_racs if 'NumB' not in i]
        racs = [i for i in racs if 'Zeff' not in i]

        #modify naming scheme for consistency with how it's stored
        racs = ['RACs.'+i for i in racs]
        isovalent_racs = ['RACs.'+i for i in isovalent_racs]

        featurization_definitions['RACs'] = np.array(racs + extra_features)
        featurization_definitions ['eRACs'] = np.array(isovalent_racs + extra_features)
        return featurization_definitions

    def unfold_geo_check_metrics(self):
        #the geo check metrics are a dictionary cast as a string by default, convert these to seperate columns in the dataframe
        #"banned_by_user","lig_mismatch","bad_init_geo"  and  are passed through as a strings
        #adds the dictionary entries as columns to self.data
        geo_dicts = self.data['new_geo_metrics'].tolist()
        def convert_to_dict(string_dict):
            geo_params = ['num_coord_metal','dist_del_eq','dist_del_all',
                          'max_del_sig_angle','oct_angle_devi_max',
                          'devi_linear_avrg','devi_linear_max','rmsd_max',
                          'atom_dist_max']
            new_dict = {}
            string_dict = string_dict[1:-1].split(',')
            for entry in string_dict:
                if entry.split(':')[1] in [r"'banned_by_user'",r" 'banned_by_user'"]:
                    key = entry.split(':')[0].split("'")[1]
                    new_dict[key]=entry.split(':')[1]
                elif entry.split(':')[1] in [r"'lig_mismatch''",r" 'lig_mismatch'"]:
                    key = entry.split(':')[0].split("'")[1]
                    new_dict[key]=entry.split(':')[1]
                elif entry.split(':')[1] in [r"'bad_init_geo'",r" 'bad_init_geo'"]:
                    key = entry.split(':')[0].split("'")[1]
                    new_dict[key]=entry.split(':')[1]
                else:
                    key = entry.split(':')[0].split("'")[1]
                    new_dict[key]=float(entry.split(':')[1])

            for key in geo_params:
                if key not in new_dict.keys():
                    new_dict[key] = np.nan
            if len(new_dict.keys()) != len(geo_params):
                print(len(new_dict.keys()))
                print(len(geo_params))
                print(new_dict)
                raise Exception('String parsing of dictionary failed!')

            return new_dict

        geo_dicts = [convert_to_dict(i) for i in geo_dicts] #convert to real dictionaries
        geo_params = ['num_coord_metal','dist_del_eq','dist_del_all',
              'max_del_sig_angle','oct_angle_devi_max',
              'devi_linear_avrg','devi_linear_max','rmsd_max']
        for param in geo_params:
            entries = [i[param] for i in geo_dicts]
            self.data[param] = entries

    def load_ip_and_ea(self):
        #populates the self.ip and self.ea attributes
        #searches out valid ip and ea options, and takes the minimum if all are available (not np.nan)
        vertIPs = self.data[['vertIP.1','vertIP.2','vertIP.3','vertIP.4','vertIP.5','vertIP.6']].values
        vertEAs = self.data[['vertEA.1','vertEA.2','vertEA.3','vertEA.4','vertEA.5','vertEA.6']].values
        vertIPs = np.where(vertIPs!=False,vertIPs,np.nan) #replace "False" with np.nan
        vertIPs = np.where(vertIPs!='False',vertIPs,np.nan)
        vertEAs = np.where(vertEAs!=False,vertEAs,np.nan)
        vertEAs = np.where(vertEAs!='False',vertEAs,np.nan)
        valid_destination_spins = [[int(i+.1)-1,int(i+.1)+1] for i in self.spin]

        def find_ip_and_ea(valid_spin1,valid_spin2,ips,eas):
            if valid_spin1 == 0 or valid_spin1>6:
                idxs = [valid_spin2-1]
            elif valid_spin2 == 0 or valid_spin2>6:
                idxs = [valid_spin1-1]
            else:
                idxs = [valid_spin1-1,valid_spin2-1]

            ips = np.array([float(ips[i]) for i in idxs])
            ips = ips[~np.isnan(ips)]
            eas = np.array([float(eas[i]) for i in idxs])
            eas = eas[~np.isnan(eas)]

            if len(ips) != len(idxs):
                ip = np.nan
            else:
                ip = np.min(ips)
            if len(eas) != len(idxs):
                ea = np.nan
            else:
                ea = np.min(eas)

            return [ip,ea]

        self.ip,self.ea = [],[]
        for i in range(len(valid_destination_spins)):
            ip,ea = find_ip_and_ea(valid_destination_spins[i][0],valid_destination_spins[i][1],
                                   vertIPs[i,:],vertEAs[i,:])

            self.ip.append(ip)
            self.ea.append(ea)

        self.ip = np.array(self.ip)
        self.ea = np.array(self.ea)

    def pre_geometry_check(self):
        #applied before the geo check, eliminates cases where one of the geo params is assigned to a string
        #updates the points in self.data
        #returns a dataframe of eliminated complexes
        if self.verbose:
            print('---- pre-checking geometries now ----')
            print('Initial dataframe shape: '+str(self.data.shape))

        dropped_frame = self.data[~((self.data['num_coord_metal'].apply(type)==float)&
                                   (self.data['dist_del_eq'].apply(type)==float)&
                                   (self.data['dist_del_all'].apply(type)==float)&
                                   (self.data['max_del_sig_angle'].apply(type)==float)&
                                   (self.data['oct_angle_devi_max'].apply(type)==float)&
                                   (self.data['devi_linear_avrg'].apply(type)==float)&
                                   (self.data['devi_linear_max'].apply(type)==float)&
                                   (self.data['rmsd_max'].apply(type)==float))
                                  ]
        self.data = self.data[((self.data['num_coord_metal'].apply(type)==float)&
                                   (self.data['dist_del_eq'].apply(type)==float)&
                                   (self.data['dist_del_all'].apply(type)==float)&
                                   (self.data['max_del_sig_angle'].apply(type)==float)&
                                   (self.data['oct_angle_devi_max'].apply(type)==float)&
                                   (self.data['devi_linear_avrg'].apply(type)==float)&
                                   (self.data['devi_linear_max'].apply(type)==float)&
                                   (self.data['rmsd_max'].apply(type)==float))
                                  ]
        if self.verbose:
            print('After dropping cases where the geo check output contains a string dataframe shape: '+str(self.data.shape))
            print('----- Eliminated complexes due to pre-geo check -----')
        return dropped_frame

    def geometry_check(self, custom_geo_dict = {}):
        #eliminates cases with bad geometery from self.data
        #returns a dataframe of the removed complexes
        if self.verbose:
            print('---- checking geometries now ----')

        if len(custom_geo_dict) > 0:
            constraint_dictionary = custom_geo_dict
        else:
            constraint_dictionary = {'num_coord_metal': 6,
                                     'rmsd_max': 0.3, 'atom_dist_max': 0.45,
                                     'oct_angle_devi_max': 12, 'max_del_sig_angle': 22.5,
                                     'dist_del_eq': 0.35, 'dist_del_all': 1,
                                     'devi_linear_avrg': 20, 'devi_linear_max': 28}

        if self.verbose:
            print('Initial dataframe shape: '+str(self.data.shape))

        dropped_frame = self.data[~((self.data['num_coord_metal'].astype(float)==constraint_dictionary['num_coord_metal'])&
                                   (self.data['dist_del_eq'].astype(float)<=constraint_dictionary['dist_del_eq'])&
                                   (self.data['dist_del_all'].astype(float)<= constraint_dictionary['dist_del_all'])&
                                   (self.data['max_del_sig_angle'].astype(float)<= constraint_dictionary['max_del_sig_angle'])&
                                   (self.data['oct_angle_devi_max'].astype(float)<= constraint_dictionary['oct_angle_devi_max'])&
                                   (self.data['devi_linear_avrg'].astype(float)<= constraint_dictionary['oct_angle_devi_max'])&
                                   (self.data['devi_linear_max'].astype(float)<= constraint_dictionary['devi_linear_max'])&
                                   (self.data['rmsd_max'].astype(float)<= constraint_dictionary['rmsd_max']))
                                  ]
        self.data = self.data[((self.data['num_coord_metal'].astype(float)==constraint_dictionary['num_coord_metal'])&
                                   (self.data['dist_del_eq'].astype(float)<=constraint_dictionary['dist_del_eq'])&
                                   (self.data['dist_del_all'].astype(float)<= constraint_dictionary['dist_del_all'])&
                                   (self.data['max_del_sig_angle'].astype(float)<= constraint_dictionary['max_del_sig_angle'])&
                                   (self.data['oct_angle_devi_max'].astype(float)<= constraint_dictionary['oct_angle_devi_max'])&
                                   (self.data['devi_linear_avrg'].astype(float)<= constraint_dictionary['oct_angle_devi_max'])&
                                   (self.data['devi_linear_max'].astype(float)<= constraint_dictionary['devi_linear_max'])&
                                   (self.data['rmsd_max'].astype(float)<= constraint_dictionary['rmsd_max']))
                                  ]
        if self.verbose:
            print('After dropping bad geo dataframe shape: '+str(self.data.shape))
            print('----- Eliminated complexes due to geometric reason -----')
        return dropped_frame

    def spin_contamination_check(self, limit=1.0):
        #eliminates points with spin contamination from self.data
        #returns a pd.DataFrame of the removed complexes
        if self.verbose:
            print('---- checking spin contamination now ----')
            print('Initial dataframe shape: '+str(self.data.shape))
        dropped_frame = self.data[~(abs(self.data['new_ss_act']-self.data['new_ss_target'])<=limit)]
        self.data = self.data[(abs(self.data['new_ss_act']-self.data['new_ss_target'])<=limit)]
        if self.verbose:
            print('After dropping spin contamination problems dataframe shape: '+str(self.data.shape))
            print('----- Eliminated complexes due to spin contamination -----')
        return dropped_frame

    # def spin_on_metal_check(self, limit=1.0): #### make sure to make this mode dependent (check empty ss)
    #     print('---- checking spin on metal now ----')
    #     print('Initial dataframe shape: '+str(self.frame.shape))
    #     dropped_frame = self.data[~(abs(self.frame['net_metal_spin'].astype(float)-self.frame['spin'].astype(float)+1)<=limit)]
    #     self.data = self.frame[(abs(self.frame['net_metal_spin'].astype(float)-self.frame['spin'].astype(float)+1)<=limit)]
    #     print('After dropping spin on metal problems dataframe shape: '+str(self.frame.shape))
    #     print('----- Eliminated complexes due to spin deviation from metal reason -----')
    #     return dropped_frame

    def check_all_floats(self,float_check,exception_name='None'):
        #takes an np.array to check
        #ensures that all entries fromt these columns are floats and aren't np.nan
        #raises an exception if this is not True

        isnan = np.vectorize(np.isnan)
        def isfloat(item):
            if type(item) == float:
                return True
            else:
                return False
        isfloat = np.vectorize(isfloat)

        for idx in range(float_check.shape[0]):
            entries = float_check[idx,:]
            if not np.all(isfloat(entries)): #if triggered we are raising an exception
                raise Exception('Non float found in: '+self.names[idx]+' during check: '+exception_name)
            if np.any(isnan(entries)): #if triggered we are raising an exception
                raise Exception('np.nan found in: '+self.names[idx]+' during check: '+exception_name)

    def coerce_strings(self,array):
        #iterates over a 1-dimensional array, converts 'False' and 'True' to np.nan
        #converts numbers as strings to floats

        array = [np.nan if i=='True' else i for i in array]
        array = [np.nan if i=='False' else i for i in array]
        array = [float(i) for i in array]

        return np.array(array)

    def eliminate_constant_features(self,threshold=0.01,):
        #eliminates constant features for all possible featurizations
        #acts in-place on the self.racs,self.eracs,etc.

        #Make a dictionary of feature sets to eliminate constant features in
        #dictionary should look like
        #{<name of featurization>:[<feature array>,<feature labels>]}
        feature_sets = {'RACs':[self.racs,self.racs_labels]}

        for name in feature_sets.keys():
            feature_array = feature_sets[name][0]
            feature_labels = feature_sets[name][1]
            if self.verbose:
                print('---- eliminating constant features in '+name+' now ----')
                print('---- initial feature count: '+str(feature_array.shape[1])+ ' ----')

            column_wise_range = np.max(feature_array,axis=0)-np.min(feature_array,axis=0)
            non_constant = np.where(column_wise_range > threshold,True,False)

            eliminated = np.array(feature_labels)[~non_constant]
            if self.verbose:
                print('Constant features eliminated from '+name+': '+str(len(eliminated)))
                if len(eliminated) > 0:
                    print(eliminated)
                print('---- Final feature count '+str(len(feature_labels[non_constant]))+' ----')

            feature_array = feature_array.T[non_constant].T
            feature_labels = feature_labels[non_constant]
            feature_sets[name] = [feature_array,feature_labels]

        #assign the object's attributes based on the contents of the dictionary
        self.racs = feature_sets['RACs'][0]
        self.racs_labels = feature_sets['RACs'][1]

#Code for testing this class's functionality

# data = Data('../raw_data/MD2-checked-unqiue.csv',apply_racs_filter=True,verbose=True,
#             extra_features=['ox','charge'])


# print('names: '+str(data.names.shape))
# print('tag: '+str(data.tag.shape))
# print('subtag: '+str(data.subtag.shape))
# print('spin: '+str(data.spin.shape))
# print('charge: '+str(data.charge.shape))
# print('oxstate: '+str(data.oxstate.shape))

# print('solvent 10.3: '+str(data.solvent_10_3.shape))
# print('solvent 2.3: '+str(data.solvent_2_3.shape))
# print('solvent 6.2: '+str(data.solvent_6_2.shape))
# print('solvent 78.9: '+str(data.solvent_78_9.shape))
# print('ip: '+str(data.ip.shape))
# print('ea: '+str(data.ea.shape))

# print('RACs: '+str(data.racs.shape))









