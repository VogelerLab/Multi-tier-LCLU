#####################################################################################################################
# Title: LCLUC project feature analysis script
# --------------------------------------------
# This script reads a set of reference data and check different feature selections using Random Forest (RF)
# classification model. Before doing the selection, the features correlation matrix is calculated and a dendrogram
# is generated that shows clustering of features based on their correlation (both correlation matrix and dendrogram
# plot are saved as output). We then select one feature from each cluster randomly and build a feature set. This
# selection is repeated for some iterations with an RF model being trained and tested for each feature set. The
# training and testing parameters (ratio, years, etc.) can be set through the variables defined in variable setting
# block. The output log is written to a .txt file. It can then be manually copied to an excel file so the different
# feature selections can be sorted based on best score and the best candidate(s) be picked.
#
# Required input data:
#   - Training dataset (.p and .txt files) created by MakeSampleData.py
#   - Training data labels (.csv file) created manually by interpreters
# Generated output file(s):
#   - Features correlation matrix (.csv) and dendrogram figure (.png)
#   - A .txt file containing clustering and selected feature sets model evaluation results
#
# Note 1: Reference data consists of two files: a binary pickle file (and its companion text file) including features
#         extracted from remote sensing data sources for each point that is generated using makeSampleData.py script,
#         and a csv file containing the reference labels assigned to those points that is prepared by manual
#         interpreters.
# Note 2: At this stage all model train and test is done using training reference set. Final model validation will
#         be conducted later using an independent validation set.
# Note 3: Variables in the variables setting block are set to the values used in the last program run and may not
#         represent the values used to create the files given in the SampleData folder.
####################################################################################################################
# Shahriar S. Heydari, Vogeler Lab, 4/24/2024

import pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
import random

# Variables setting block
##########################
# input features specification (file name, desired features to extract, and time-series reduction method)
data_file_name = 'SA_urbanTrainSet'
features_to_pick = 'all'  # either 'all' or a list of features to pick
reduceTSflag = 1
# if 0, median value of all yearly observations are used,
# if 1, the min/mean/max/range yearly values are calculated and used,
# if 2, the observation closest to selectDOY elements are used.
selectDOY = np.array([106, 288])/366  # (15, 106, 197, 288 for 4 seasons)

# reference labels file specification
labels_file_name = 'SA_urbanTrainSetLabels.csv'
response_columns = ['LCP_16', 'LCP_17', 'LCP_18', 'LCP_19', 'LCP_20']                       # columns to hold reference labels per year
confidence_columns = ['conf_2016', 'conf_2017', 'conf_2018', 'conf_2019', 'conf_2020']      # columns to hold labels confidence per year
# list of confidence values to consider (other terms will not be included in analysis)
conf_to_pick = ['High','high','Low','low']
# you can combine labels to make a more general label through a dictionary definition here. for example, you can
# # set combine_labels = {'Impervious': ['Building', 'Pavement'], 'Vegetation': ['Short_vegetation', 'Tall_vegetation']}
combine_labels = None
labels_first_year = 2016
labels_number_of_years = 5

# train/test parameters definition
test_size = 0.2           # Note: test_size should be scalar, but multiple train_sizes are allowed
test_years = 'all'        # either 'all' or list of years (or a list of just one year)
train_sizes = [0.8]       # multiple training ratios can be specified as a list
train_years = 'all'       # either 'all' or list of years (or a list of just one year)
# Note: If you define specific years for test and train, the two lists should be mutually exclusive.

# if num_train_points_groups > 0, each training point is assigned a random group number between zero and
# num_train_points_groups in such a way that the total number of points in the groups are almost the same. This
# capability is useful if you want to do a k-fold cross-validation later.
num_train_points_groups = 0

# Random Forest classifier definition
n_estimators = 100
min_samples_leaf = 3
clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, class_weight='balanced_subsample')

# other variables
clustering_thresholds = [0.2, 0.5, 1]
num_clustering_iterations = 2
num_training_iterations = 2
outputFolder = ''
save_flattened_labels = False

################################################################################################################
# Helper functions
################################################################################################################

def closest(lst, K):
    # will be used to pick suitable time stamps in case reduceDate != 0
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx

def generate_sample_wrapper_function(allPoints_allFeatures, all_feature_names):
    # wrapper function to help generating yearly stratified samples for each train and test set. This function calls
    # generate_samples function to create the final training and test data.
    # Note: It is assumed that the first four columns of matrix allPoints_allFeatures are 'PointID', 'XLon', 'YLat',
    # and 'Year', and the last column is 'Label.

    # as input matrix has one row for each point for each year, we first select just one year of data
    # to partition the points to train and test points so the train and test data will be spatially disjoint
    # (but they can share data from the same year)
    year_ind = 3
    label_ind = -1
    points_base = allPoints_allFeatures[allPoints_allFeatures[:,year_ind] == labels_first_year,]

    # do stratified sampling to pick test points
    p_rest, p_test, y_rest_p, y_test_p = train_test_split(points_base[:, 0], points_base[:, label_ind],
                                                          test_size=test_size, stratify=points_base[:, label_ind])

    p_train_list = []
    # for each train size specified in the input train size list, an updated ratio is calculated to
    # adjust the initial value to the number of points available after setting aside the test points
    updated_train_sizes = [train_size * len(points_base) / len(p_rest) for train_size in train_sizes]
    # for each train size do the stratified sampling to pick training points
    for train_size in updated_train_sizes:
        if train_size >= 1:  # the last train_size calculation may get bigger than 1 due to rounding errors,
            # so just use the "rest" of points in such a case
            p_train = p_rest
        else:
            p_train, p_unused, y_train_p, y_unused_p = train_test_split(p_rest, y_rest_p,
                                                                        train_size=train_size, stratify=y_rest_p)
        # create point groups if num_train_points_groups > 0, or add a vector of all zeros
        p_train_group = np.zeros(p_train.shape[0])
        if num_train_points_groups > 0:
            bin_size = int(p_train.shape[0] / num_train_points_groups)
            for i in range(num_train_points_groups - 1):
                p_train_group[i * bin_size:(i + 1) * bin_size] = i
            p_train_group[(num_train_points_groups - 1) * bin_size:] = num_train_points_groups-1
        p_train = np.hstack((p_train[:,np.newaxis], p_train_group[:,np.newaxis]))
        p_train_list.append(p_train)

    # now we have an array containing test points (p_test) and a list of arrays containing training points for each
    # train size (p_train_list). These lists are passed along with input features matrix to the sampling function
    X_train_list, y_train_list, g_train_list, X_test, y_test, features_names = \
        generate_samples(p_train_list, p_test, allPoints_allFeatures, all_feature_names)

    # returning variables include:
    # X_train_list: A list containing training features matrix for each selected train size
    # y_train_list: point labels corresponding to X_train_list items
    # g_train_list: point group numbers corresponding to X_train_list items
    # X_test: test data features matrix
    # y_test: test data labels
    # features_names: A list containing name of features in the features matrix
    return X_train_list, y_train_list, g_train_list, X_test, y_test, features_names

def generate_samples(train_points_list, test_points, allPoints_allFeatures, all_feature_names):
    # function to extract yearly samples for selected points from globally defined allPoints_allFeatures variable.
    # Features can be dropped or selected as well by setting related variables.
    # test_points is a list of test point IDs,
    # train_points_list is a list of arrays including train point IDs and their group numbers.
    # Note: It is assumed that the first four columns of matrix allPoints_allFeatures are 'PointID', 'XLon', 'YLat',
    # and 'Year', and the last column is 'Label.

    # Below array is duplicate array for test samples (arrays will be reduced later by dropping unwanted points/years)
    test_features = np.copy(allPoints_allFeatures)
    # Look at each point's available samples and decide to drop some of them from test samples or not.
    # If a fixed train year is specified, that year's sample will be dropped from test samples. Otherwise,
    # all years are used for testing.
    # keep_ind is the variable used to flag qualified entries. Those entries that will remain zero after the loop will
    # be dropped, including all entries related to the TEST points.
    keep_ind = np.zeros(test_features.shape[0])
    for pID in test_points:
        if test_years != 'all':
            ind = []
            for year in test_years:
                loc = np.where(np.logical_and(test_features[:, 0] == pID, test_features[:, 3] == year))[0].tolist()
                ind = ind + loc
        else:
            ind = np.where(test_features[:, 0] == pID)[0]
        keep_ind[ind] = 1
    # drop non-qualified entries
    entries_to_drop_test = np.where(keep_ind == 0)[0]
    test_features = np.delete(test_features, entries_to_drop_test, axis=0)
    # randomise samples
    np.random.shuffle(test_features)
    # drop point ID, coordinate, and year from features to create train/test data vectors and targets for
    # classification. Output feature names will be updated as well.
    X_test = test_features[:, feature_start_index:label_ind]
    y_test = test_features[:, label_ind]
    features_out = all_feature_names[feature_start_index:label_ind]

    X_train_list = []
    y_train_list = []
    g_train_list = []
    # Now do the same for each set of training points
    for train_points in train_points_list:
        train_features = np.copy(allPoints_allFeatures)
        group_features = np.zeros(train_features.shape[0])
        keep_ind = np.zeros(train_features.shape[0])
        for pID, group in train_points:
            if train_years != 'all':
                ind = []
                for year in train_years:
                    loc = np.where(np.logical_and(train_features[:, 0] == pID, train_features[:, 3] == year))[0].tolist()
                    ind = ind + loc
            else:
                ind = np.where(train_features[:, 0] == pID)[0]
            keep_ind[ind] = 1
            group_features[ind] = group
        entries_to_drop_train = np.where(keep_ind == 0)[0]
        train_features = np.delete(train_features, entries_to_drop_train, axis=0)
        group_features = np.delete(group_features, entries_to_drop_train, axis=0)
        rnd_index = np.arange(train_features.shape[0])
        np.random.shuffle(rnd_index)
        train_features = train_features[rnd_index]
        group_features = group_features[rnd_index]
        X_train = train_features[:, feature_start_index:label_ind]
        y_train = train_features[:, label_ind]
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        g_train_list.append(group_features)

    return X_train_list, y_train_list, g_train_list, X_test, y_test, features_out

def reduce_time_series(time_series, band_names_in, TS_index, flag):
    # function to reduce time series into aggregate values

    if flag == 0:  # make the median of all observations
        reduced_time_series = np.median(time_series, axis=0)
        band_names_out = band_names_in
    elif flag == 1: # extract min/max/mean/range statistics
        try:
            _min = np.nanmin(time_series, axis=0)
            _mean = np.nanmean(time_series, axis=0)
            _max = np.nanmax(time_series, axis=0)
            _range = _max - _min
        except:
            _min = np.array(np.repeat(np.nan, len(band_names_in)))
            _mean = _min
            _max = _min
            _range = 0
        _min_bands = [x + '_YearMin' for x in band_names_in]
        _mean_bands = [x + '_YearMean' for x in band_names_in]
        _max_bands = [x + '_YearMax' for x in band_names_in]
        _range_bands = [x + '_YearRange' for x in band_names_in]
        reduced_time_series = np.hstack((_min, _mean, _max, _range))
        band_names_out = _min_bands + _mean_bands + _max_bands  + _range_bands
    elif flag == 2:  # pick suitable observations and attach them together horizontally
        index = [closest(time_series[:, TS_index], l) for l in selectDOY]
        series_temp = []
        series_bands = []
        for k in range(len(index)):
            series_temp = series_temp + time_series[index[k]].tolist()
            series_bands = series_bands + [x + '_' + str(k) for x in band_names_in]
        reduced_time_series = np.array(series_temp)
        band_names_out = series_bands
    return reduced_time_series, band_names_out

def prepare_data(data_file_name, labels_file_name, features_to_pick, combine_labels):
    # function to read points reference labels and points extracted data from reference datasets and generate
    # a complete features matrix for all points and years. Each row in this matrix will contain all extracted
    # features for each point in one year.

    # reading points labels from csv file and process it to a flattened list (the input file has multiple years listed
    # in the same row for each point, i.e. each point is represented by one line. This structure is flattened and
    # the empty fields are filled and a flattened array is created in which each point and year have a separate row.
    ###############################################################################################################
    labels = pd.read_csv(labels_file_name)
    response_names = np.unique(labels.loc[:, response_columns[0]].dropna()).tolist()
    if combine_labels != None:
        response_names_new = response_names
        for new_label in list(combine_labels.keys()):
            response_names_new = [x if x not in combine_labels.get(new_label) else new_label for x in response_names_new]
        response_names_new_unique = np.unique(response_names_new).tolist()
    points_and_labels = []  # the full table of available points and labels for each year
    for i in labels.index:
        pID = labels.loc[i, 'pointID']
        first_confident_year_index = 0
        response_base_confidence = labels.loc[i, confidence_columns[first_confident_year_index]]
        while ((response_base_confidence not in conf_to_pick) or pd.isnull(response_base_confidence)) and (first_confident_year_index < labels_number_of_years-1):
            first_confident_year_index += 1
            response_base_confidence = labels.loc[i, confidence_columns[first_confident_year_index]]
        if ((response_base_confidence not in conf_to_pick) or pd.isnull(response_base_confidence)) and (first_confident_year_index == labels_number_of_years-1):
            continue
        response_base = labels.loc[i, response_columns[first_confident_year_index]] # retrieve the base response (for first year)
        response_base_index = response_names.index(response_base)
        for j in range(first_confident_year_index, labels_number_of_years):  # add other response values in subsequent years
            response_year = labels.loc[i, response_columns[j]]
            if pd.isnull(response_year):
                response_year = response_base
                response_year_index = response_base_index
            else:
                response_year_index = response_names.index(response_year)
            response_year_confidence = labels.loc[i, confidence_columns[j]]
            if pd.isnull(response_year_confidence):
                response_year_confidence = response_base_confidence
            if (response_year_confidence not in conf_to_pick):
                response_base_confidence = response_year_confidence
                continue
            if (response_year_confidence in conf_to_pick): # change label
                response_base = response_year
                response_base_index = response_names.index(response_year)
                response_base_confidence = response_year_confidence
            if combine_labels == None:
                points_and_labels.append([pID, labels_first_year + j, response_year_index])
            else:
                response_base_index_new = response_names_new_unique.index(response_names_new[response_year_index])
                points_and_labels.append([pID, labels_first_year + j, response_base_index_new])
    # convert list to numpy array
    points_and_labels = np.array(points_and_labels)

    if save_flattened_labels:
        pl = pd.DataFrame(points_and_labels, columns=['ID','Year','class'])
        pl.to_csv(outputFolder + 'labels_flattened.csv')
    # response_names will contain final assignment for reference label names after processing and label merging
    if combine_labels != None:
        response_names = response_names_new_unique

    # reading extracted features from pickle file
    #############################################
    point_data = pickle.load(open(data_file_name + '.p', 'rb'))

    # reading data configuration text file and parse it to get feature names and some other information
    ###################################################################################################
    with open(data_file_name + '.txt', 'r') as f:
        contents = f.read()
        # reading feature names
        p1 = contents.find('Sentinel-1 bands:')
        temp = contents[p1 + 18:]
        p2 = temp.find('\n')
        Sentinel1BandNames = eval(temp[:p2].strip())
        p1 = contents.find('Sentinel-2 bands:')
        temp = contents[p1 + 18:]
        p2 = temp.find('\n')
        Sentinel2BandNames = eval(temp[:p2].strip())
        p1 = contents.find('Annual bands:')
        temp = contents[p1 + 14:]
        p2 = temp.find('\n')
        annualBandNames = eval(temp[:p2].strip())
        p1 = contents.find('Static bands:')
        temp = contents[p1 + 14:]
        p2 = temp.find('\n')
        staticBandNames = eval(temp[:p2].strip())
    Sentinel1_TS_index = Sentinel1BandNames.index('timestamp')
    Sentinel2_TS_index = Sentinel2BandNames.index('timestamp')

    pointVector = []
    Sentinel1DataVector = []
    Sentinel2DataVector = []
    annualDataVector = []
    staticDataVector = []
    outputLabels = []

    # converting retrieved data from pickle file to the required 2-D features matrix structure
    ###########################################################################################
    for i in range(len(point_data)):
        # extracting data parts
        data = point_data[i]
        pID = data['point_ID']
        coord = data['coordinates']
        variable_data = data['variable_data']
        static_data = data['fixed_data']['static_data']

        N = len(variable_data)
        for j in range(N):
            year = variable_data[j]['year']
            try:
                label = points_and_labels[np.where(
                    np.logical_and(points_and_labels[:, 0] == pID, points_and_labels[:, 1] == year)), 2].item()
            except:
                continue

            Sentinel1_data = variable_data[j]['Sentinel1_data']
            Sentinel2_data = variable_data[j]['Sentinel2_data']

            # convert fractional year to DOY
            Sentinel1_data[:, Sentinel1_TS_index] = (366 * (Sentinel1_data[:, Sentinel1_TS_index] - year)).astype(int)
            Sentinel2_data[:, Sentinel2_TS_index] = (366 * (Sentinel2_data[:, Sentinel2_TS_index] - year)).astype(int)

            Sentinel1_data, Sentinel1NewBandNames = reduce_time_series(Sentinel1_data, Sentinel1BandNames,
                                                                   Sentinel1_TS_index, reduceTSflag)
            timestamp_indices = [Sentinel1NewBandNames.index(x) for x in Sentinel1NewBandNames if x.find('timestamp')>-1]
            Sentinel1_data = np.delete(Sentinel1_data, timestamp_indices)
            Sentinel1NewBandNames = [Sentinel1NewBandNames[x] for x in range(len(Sentinel1NewBandNames)) if x not in timestamp_indices]

            Sentinel2_data, Sentinel2NewBandNames = reduce_time_series(Sentinel2_data, Sentinel2BandNames,
                                                                     Sentinel2_TS_index,  reduceTSflag)
            timestamp_indices = [Sentinel2NewBandNames.index(x) for x in Sentinel2NewBandNames if
                                 x.find('timestamp') > -1]
            Sentinel2_data = np.delete(Sentinel2_data, timestamp_indices)
            Sentinel2NewBandNames = [Sentinel2NewBandNames[x] for x in range(len(Sentinel2NewBandNames)) if
                                   x not in timestamp_indices]

            annual_data = variable_data[j]['annual_data']

            pointVector.append([pID] + coord + [year])
            Sentinel1DataVector.append(Sentinel1_data)
            Sentinel2DataVector.append(Sentinel2_data)
            annualDataVector.append(annual_data)
            staticDataVector.append(static_data)
            outputLabels.append(label)

    outputFeatures = np.hstack((np.array(pointVector), np.array(Sentinel1DataVector), np.array(Sentinel2DataVector),
                                np.array(annualDataVector), np.array(staticDataVector),
                                np.array(outputLabels)[:,np.newaxis]))

    yearly_feature_names = Sentinel1NewBandNames + Sentinel2NewBandNames + annualBandNames
    static_feature_names = staticBandNames
    output_feature_names = ['PointID', 'XLon', 'YLat', 'Year'] + yearly_feature_names + static_feature_names + ['Label']

    if features_to_pick != 'all':
        pick_indices = [0,1,2,3] + [output_feature_names.index(i) for i in features_to_pick] + [label_ind]
        outputFeatures = outputFeatures[:, pick_indices]
        output_feature_names = ['PointID', 'XLon', 'YLat', 'Year'] + features_to_pick + ['Label']

    # now allPoints_allFeatures is a 2D numpy array containing all features for each point
    # remove rows with NaN values
    outputFeatures = outputFeatures[~np.isnan(outputFeatures).any(axis=1)]
    # shuffle it, in case we want to use it directly for cross-validation
    np.random.shuffle(outputFeatures)
    return response_names, outputFeatures, output_feature_names

##########################################################################################
# Main program
##########################################################################################

# reading input data files and generating features matrix
response_names, outputFeatures, output_feature_names = prepare_data(data_file_name, labels_file_name,
                                                                    features_to_pick, combine_labels)
# NOTE: It is assumed that the first four columns of matrix outputFeatures are 'PointID', 'XLon', 'YLat',
# and 'Year', and the last column is 'Label. Therefore, features are all other columns.
year_ind = 3
feature_start_index = 4
label_ind = -1

# drop non-feature columns for correlation analysis
Xs = outputFeatures[:,feature_start_index:label_ind]
# calculate feature correlation and save it to an excel file
corr = spearmanr(Xs).correlation
corr_df = pd.DataFrame(np.abs(corr), columns=output_feature_names[feature_start_index:label_ind])
corr_df.to_excel(outputFolder + 'correlations.xls')

# clustering data and generating features dendrogram:
# first ensure the correlation matrix is symmetric:
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
# then convert the correlation matrix to a distance matrix
distance_matrix = 1 - np.abs(corr)
# performing hierarchical clustering using Ward's linkage
dist_linkage = hierarchy.ward(squareform(distance_matrix))
# creating and plotting dendrogram
fig, ax1 = plt.subplots(1, 1, figsize=(30,30))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=output_feature_names[feature_start_index:label_ind], ax=ax1, orientation='right'
)
dendro_idx = np.arange(0, len(dendro["ivl"]))
plt.savefig(outputFolder + 'Dendrogram.png', dpi=600)

# setup clustering variables and log file
cluster_ids = [0] * len(clustering_thresholds)
f = open(outputFolder + 'clustering_iter'+str(num_training_iterations)+'x'+str(num_clustering_iterations)+'.txt','w')
f.write('Points data file: {}\n'.format(data_file_name))
f.write('Points label file: {}\n'.format(labels_file_name))
f.write('Picked features: \n{}\n'.format(features_to_pick))

# scores matrix will contain each simulation run's evaluated score
scores = np.zeros((num_training_iterations, len(clustering_thresholds), num_clustering_iterations))
for j in range(num_training_iterations):
    # each training iteration will pick a new random train and test dataset from the main data pool
    X_train_list, y_train_list, g_train_list, X_test, y_test, features_names = \
        generate_sample_wrapper_function(outputFeatures, output_feature_names)
    for k in range(len(clustering_thresholds)):
        # for each specified clustering threshold size, the dendrogram is cut at that level and the partitioning
        # of correlated features (at that level of distance) is obtained and stored in a list of lists variable
        start = time.time()
        cluster_ids[k] = hierarchy.fcluster(dist_linkage, clustering_thresholds[k], criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids[k]):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        max_score = 0
        for iter in range(num_clustering_iterations):
            # through another iteration loop, one random sample from each cluster is picked and a random forest
            # model is trained based on those specific selected features. The model performance is evaluated using
            # test data set and the results are stores in scores matrix and best found features.
            selected_features = [random.choice(v) for v in cluster_id_to_feature_ids.values()]
            selected_features_names = [features_names[u] for u in selected_features]
            X_train_sel = X_train_list[0][:, selected_features]
            X_test_sel = X_test[:, selected_features]
            clf.fit(X_train_sel, y_train_list[0])
            new_score = clf.score(X_test_sel, y_test)
            scores[j, k, iter] = new_score
            if new_score > max_score:
                max_score = new_score
                best_selected_features_names = selected_features_names
        print('Iteration# {}, threshold: {}, best score = {:.2f}, features: {}'
              .format(j, clustering_thresholds[k],max_score, best_selected_features_names))
        f.write('Iteration# {}, threshold: {}, best score = {:.2f}, features: {}\n'
              .format(j, clustering_thresholds[k],max_score, best_selected_features_names))
for k in range(len(clustering_thresholds)):
    print('Cluster distance threshold setting: {:.2f}, # of selected features: {}, '
      'mean accuracy over iterations on test data: {:.2f}'.
      format(clustering_thresholds[k], max(cluster_ids[k]),np.mean(scores[:,k,:])))
    f.write('Cluster distance threshold setting: {:.2f}, # of selected features: {}, '
          'mean accuracy over iterations on test data: {:.2f}\n'.
          format(clustering_thresholds[k], max(cluster_ids[k]), np.mean(scores[:, k, :])))
f.close()

