import dotenv
import os
import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import ot

#Function to perform the pairwise distance function of all points within 'points' parameter

def pairwise_wasserstein(pointsA, pointsB):
    wass_dist = np.empty((len(pointsA),len(pointsB)))
    for first_index in range(0,pointsA.shape[0]):
      for second_index in range(0,pointsB.shape[0]):
        sp = np.arange(len(pointsA[first_index]))
        wass_dist[first_index, second_index]=ot.wasserstein_1d(sp, sp, a=pointsA[first_index], b=pointsB[second_index])
    return wass_dist

def compute_average_precision_table_given_dist_metric (metric, df, train, test):  
    if metric == 'wasserstein': 
        dist_df = pd.DataFrame(pairwise_wasserstein(test.iloc[:, 1:].values, train.iloc[:, 1:].values),
                      columns = train['sample_id'], index = test['sample_id'])
    elif metric != 'random': 
        dist_df = pd.DataFrame(pairwise_distances(test.iloc[:, 1:].values, train.iloc[:, 1:], metric=metric),
                      columns = train['sample_id'], index = test['sample_id'])
    precision = []
    final_number_of_recommended_drugs = []
    results_neighbors_sensitivity = np.empty((len(TEST_SAMPLES),5))
    results_neighbors_resistance = np.empty((len(TEST_SAMPLES),5))
    for top_samples in N_TOP_CLOSEST_SAMPLES: #for top_samples neighbors
        i=-1
        for sample in TEST_SAMPLES: #for each sample from the testing set
            i=i+1
            if metric == 'random':               
                # randomly sample top_samples from train
                top_most_similar_samples_in_train = random.sample(TRAIN_SAMPLES, top_samples)
            else:
                # select the top neighbors
                top_most_similar_samples_in_train = list(dist_df.loc[sample].sort_values(ascending=True)[:top_samples].index)           
            #count the occurrence of drugs in the similar samples
            counts_of_possible_sensitive_drugs_from_train = df[(df['sample_id'].isin(top_most_similar_samples_in_train)) &\
                                                     (df['response'] == RESPONSE)]['drug'].value_counts()
            #select the drugs for which we know the resistance/sensitivity value in the test sample
            drug_in_test_and_train=df[(df['sample_id'] == sample) & \
                                       (df['drug'].isin(counts_of_possible_sensitive_drugs_from_train.index))]
            prediction=counts_of_possible_sensitive_drugs_from_train.reindex(drug_in_test_and_train['drug'])
            drug_in_test_and_train.loc[:,'counts_drugs']=prediction.array
            #order the drugs per occurrence in the similar samples
            drug_in_test_and_train_sensitivity=drug_in_test_and_train.sort_values(by=['counts_drugs'], ascending=False)
            drug_in_test_and_train_resistance=drug_in_test_and_train.sort_values(by=['counts_drugs'])
            # take the most common sensitive drugs among all top patients
            max_occurrence_number = drug_in_test_and_train_sensitivity['counts_drugs'].max()
            boolean_counts = drug_in_test_and_train_sensitivity['counts_drugs'] == max_occurrence_number
            final_drug_recommendation=drug_in_test_and_train_sensitivity[(drug_in_test_and_train_sensitivity['counts_drugs'] == max_occurrence_number)]['drug']
            if len(final_drug_recommendation) == 0: #if there is no drugs in common in similar samples and the test sample
                final_drug_recommendation = list(range(0,10000)) # assign very high number so precision is zero
            drugs_that_actually_work = df[(df['sample_id'] == sample) & \
                                          (df['drug'].isin(final_drug_recommendation)) & \
                                          (df['response'] == RESPONSE)].shape[0]
            #Compute the precision
            precision.append(drugs_that_actually_work/len(final_drug_recommendation))
            final_number_of_recommended_drugs.append(len(final_drug_recommendation))   
            for j in range(0,5): #drug cutoff
             if len(drug_in_test_and_train_resistance)>j: #if more than j drugs in our recommandation
                 #compute the precision at cutoff j for the sensitivity and the resistance settings
                 results_neighbors_sensitivity[i, j]=(j+1-sum(drug_in_test_and_train_sensitivity['response'][0:j+1]))/(j+1)
                 results_neighbors_resistance[i, j]=sum(drug_in_test_and_train_resistance['response'][0:j+1])/(j+1)
             else: #otherwise, do not compute the precision at cutoff j
                 results_neighbors_sensitivity[i, j]=np.nan
                 results_neighbors_resistance[i, j]=np.nan
    #Gather the results
    precision = pd.DataFrame(np.array(precision).reshape(len(N_TOP_CLOSEST_SAMPLES), test.shape[0]),
                            index = N_TOP_CLOSEST_SAMPLES)
    final_number_of_recommended_drugs = pd.DataFrame(np.array(final_number_of_recommended_drugs).reshape(len(N_TOP_CLOSEST_SAMPLES), test.shape[0]),
                            index = N_TOP_CLOSEST_SAMPLES)
    return precision, final_number_of_recommended_drugs, results_neighbors_sensitivity, results_neighbors_resistance

#Load data
path='/home/'
train = pd.read_csv('/home/spectrum_train.csv')
test = pd.read_csv('/home/spectrum_test.csv')
df = pd.read_csv('/home/df_B.csv')

#Parameters
TEST_SAMPLES = list(test['sample_id'])
TRAIN_SAMPLES = list(train['sample_id'])
RESPONSE = 0
#N_TOP_CLOSEST_SAMPLES = [1, 5, 10, 30, 50, 75, 100]
N_TOP_CLOSEST_SAMPLES = [30]
METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'braycurtis', 'canberra', 'chebyshev', 
            'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'minkowski', 'wasserstein']

for metric in METRICS:
    precision, final_number, results_neighbors_sensitivity, results_neighbors_resistance = compute_average_precision_table_given_dist_metric (metric = metric, df = df, train = train, test=test)
    ax.plot(N_TOP_CLOSEST_SAMPLES, precision.mean(axis=1), label = metric+" Spectrum", linewidth=3)
    np.save(path+'/precision_test_'+metric+'.npy', precision)
    np.save(path+'/final_number_test_'+metric+'.npy', final_number)
    np.save(path+'/results_neighbors_sensitivity_'+metric+'.npy', results_neighbors_sensitivity)
    np.save(path+'/results_neighbors_resistance_'+metric+'.npy', results_neighbors_resistance)

    
for metric in ['random']:
    for i in range(100):
        precision, final_number, results_neighbors_sensitivity, results_neighbors_resistance = compute_average_precision_table_given_dist_metric (metric = metric, df = df,
                                                       train = train, test=test)
        ax.plot(N_TOP_CLOSEST_SAMPLES, precision.mean(axis=1), linewidth=2, alpha=0.3, color = 'grey')
        np.save(path+'/precision_test_random_'+ str(i)+'.npy', precision)
        np.save(path+'/final_number_test_random_'+ str(i)+'.npy', final_number)
        np.save(path+'/results_neighbors_sensitivity_random'+ str(i)+'.npy', results_neighbors_sensitivity)
        np.save(path+'/results_neighbors_resistance_random'+ str(i)+'.npy', results_neighbors_resistance)

  

