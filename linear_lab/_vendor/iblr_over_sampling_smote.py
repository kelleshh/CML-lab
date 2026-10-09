# Vendored from ImbalancedLearningRegression/over_sampling_smote.py (GPL-3.0); modifications documented in PROVENANCE.md.
## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm
from .sampling_math import interpolate_target

## load dependencies - internal
from ImbalancedLearningRegression.dist_metrics import euclidean_dist, heom_dist, overlap_dist

## generate synthetic observations
def over_sampling_smote(
    
    ## arguments / inputs
    data,       ## training set
    index,      ## index of input data
    perc,       ## oversampling percentage
    k           ## num of neighs for over-sampling
    
    ):
    
    """
    generates synthetic observations and is the primary function underlying the
    over-sampling technique utilized in the higher main function 'smote()', the
    4 step procedure for generating synthetic observations is:
    
    1) pre-processing: temporarily removes features without variation, label 
    encodes nominal / categorical features, and subsets the training set into 
    two data sets by data type: numeric / continuous, and nominal / categorical
    
    2) distances: calculates the cartesian distances between all observations, 
    distance metric automatically determined by data type (euclidean distance 
    for numeric only data, heom distance for both numeric and nominal data, and 
    hamming distance for nominal only data) and determine k nearest neighbors
    
    3) over-sampling: 'smote' only applies to numeric / continuous features, 
    for nominal / categorical features, synthetic values are generated at random 
    from sampling observed values found within the same feature
    
    4) post processing: restores original values for label encoded features, 
    reintroduces constant features previously removed, converts any interpolated
    negative values to zero in the case of non-negative features
    
    returns a pandas dataframe containing synthetic observations of the training
    set which are then returned to the higher main function 'smote()'
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
    
    Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013, September). 
    Smote for regression. In Portuguese conference on artificial intelligence (pp. 378-389). 
    Springer, Berlin, Heidelberg. 
    https://researchcommons.waikato.ac.nz/bitstream/handle/10289/8518/smoteR.pdf?sequence=23
    """
    
    ## Check the domain on all input training rows, before selecting a relevance bump.
    original_nonnegative_columns = {j for j in range(len(data.columns))
                                    if pd.api.types.is_numeric_dtype(data.iloc[:, j]) and (data.iloc[:, j] >= 0).all()}
    ## subset original dataframe by bump classification index
    data = data.iloc[index]
    
    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)
    
    ## store original data types
    feat_dtypes_orig = [None] * d
    
    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype
    
    ## find non-negative numeric features
    feat_non_neg = [] 
    num_dtypes = ["int64", "float64"]
    
    for j in range(d):
        if j in original_nonnegative_columns:
            feat_non_neg.append(j)
    
    ## find features without variation (constant features)
    feat_const = data.columns[(data.nunique() == 1) & (data.columns != data.columns[-1])]
    
    ## temporarily remove constant features
    if len(feat_const) > 0:
        
        ## create copy of orignal data and omit constant features
        data_orig = data.copy()
        data = data.drop(data.columns[feat_const], axis = 1)
        
        ## store list of features with variation
        feat_var = list(data.columns.values)
        
        ## reindex features with variation
        for i in range(d - len(feat_const)):
            data.rename(columns = {
                data.columns[i]: i
                }, inplace = True)
        
        ## store new dimension of feature space
        d = len(data.columns)
    
    ## create copy of data containing variation
    data_var = data.copy()
    
    ## create global feature list by column index
    feat_list = list(data.columns.values)
    
    ## create nominal feature list and
    ## label encode nominal / categorical features
    ## (strictly label encode, not one hot encode) 
    feat_list_nom = []
    nom_dtypes = ["object", "bool", "datetime64"]

    # Unknown warning, may be handled later
    
    for j in range(d):
        if data.dtypes[j] in nom_dtypes:
            feat_list_nom.append(j)
            data.iloc[:, j] = pd.Categorical(pd.factorize(
                data.iloc[:, j])[0])
    
    data = data.apply(pd.to_numeric)
    
    ## create numeric feature list
    feat_list_num = list(set(feat_list) - set(feat_list_nom))
    
    ## calculate ranges for numeric / continuous features
    ## (includes label encoded features)
    feat_ranges = list(np.repeat(1, d))
    
    if len(feat_list_nom) > 0:
        for j in feat_list_num:
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    else:
        for j in range(d):
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    
    ## subset feature ranges to include only numeric features
    ## (excludes label encoded features)
    feat_ranges_num = [feat_ranges[i] for i in feat_list_num]
    
    ## subset data by either numeric / continuous or nominal / categorical
    data_num = data.iloc[:, feat_list_num]
    data_nom = data.iloc[:, feat_list_nom]
    
    ## get number of features for each data type
    feat_count_num = len(feat_list_num)
    feat_count_nom = len(feat_list_nom)
    
    ## calculate distance between observations based on data types
    ## store results over null distance matrix of n x n
    dist_matrix = np.ndarray(shape = (n, n))
    
    for i in tqdm(range(n), disable = True, ascii = True, desc = "dist_matrix"):
        for j in range(n):
            
            ## utilize euclidean distance given that 
            ## data is all numeric / continuous
            if feat_count_nom == 0:
                dist_matrix[i][j] = euclidean_dist(
                    a = data_num.iloc[i],
                    b = data_num.iloc[j],
                    d = feat_count_num
                )
            
            ## utilize heom distance given that 
            ## data contains both numeric / continuous 
            ## and nominal / categorical
            if feat_count_nom > 0 and feat_count_num > 0:
                dist_matrix[i][j] = heom_dist(
                    
                    ## numeric inputs
                    a_num = data_num.iloc[i],
                    b_num = data_num.iloc[j],
                    d_num = feat_count_num,
                    ranges_num = feat_ranges_num,
                    
                    ## nominal inputs
                    a_nom = data_nom.iloc[i],
                    b_nom = data_nom.iloc[j],
                    d_nom = feat_count_nom
                )
            
            ## utilize hamming distance given that 
            ## data is all nominal / categorical
            if feat_count_num == 0:
                dist_matrix[i][j] = overlap_dist(
                    a = data_nom.iloc[i],
                    b = data_nom.iloc[j],
                    d = feat_count_nom
                )
    
    ## determine indicies of k nearest neighbors
    ## and convert knn index list to matrix
    knn_index = [None] * n
    
    for i in range(n):
        knn_index[i] = np.argsort(dist_matrix[i])[1:k + 1]
    
    knn_matrix = np.array(knn_index)
    
    ## number of new synthetic observations for each rare observation
    x_synth = int(perc - 1)
    
    ## total number of new synthetic observations to generate
    n_synth = int(n * (perc - 1 - x_synth))
    
    ## randomly index data by the number of new synthetic observations
    r_index = np.random.choice(
        a = tuple(range(0, n)), 
        size = n_synth, 
        replace = False, 
        p = None
    )

    
    ## create null matrix to store new synthetic observations
    synth_matrix = np.zeros(shape = ((x_synth * n + n_synth), d))
    
    if x_synth > 0:
        for i in tqdm(range(n), disable = True, ascii = True, desc = "synth_matrix"):
            
            for j in range(x_synth):
                
                ## randomly select a k nearest neighbor
                neigh = int(np.random.choice(
                    a = tuple(range(k)), 
                    size = 1))
                
                ## conduct synthetic minority over-sampling
                ## technique for regression (smote)
                diffs = data.iloc[
                    knn_matrix[i, neigh], 0:(d - 1)] - data.iloc[
                    i, 0:(d - 1)]
                synth_matrix[i * x_synth + j, 0:(d - 1)] = data.iloc[
                    i, 0:(d - 1)] + rd.random() * diffs
                
                ## randomly assign nominal / categorical features from
                ## observed cases and selected neighbors
                for x in feat_list_nom:
                    synth_matrix[i * x_synth + j, x] = [data.iloc[
                        knn_matrix[i, neigh], x], data.iloc[
                        i, x]][round(rd.random())]
                
                ## generate synthetic y response variable by
                ## inverse distance weighted
                synth_matrix[i * x_synth + j, d - 1] = interpolate_target(data, i, knn_matrix[i, neigh], synth_matrix[i * x_synth + j, :d - 1], feat_ranges[:-1])
                    
    
    if n_synth > 0:
        count = 0
        
        for i in tqdm(r_index, disable = True, ascii = True, desc = "r_index"):
            
            ## randomly select a k nearest neighbor
            neigh = int(np.random.choice(
                a = tuple(range(0, k)), 
                size = 1))
            
            ## conduct synthetic minority over-sampling 
            ## technique for regression (smote)
            diffs = data.iloc[
                knn_matrix[i, neigh], 0:(d - 1)] - data.iloc[i, 0:(d - 1)]
            synth_matrix[x_synth * n + count, 0:(d - 1)] = data.iloc[
                i, 0:(d - 1)] + rd.random() * diffs
            
            ## randomly assign nominal / categorical features from
            ## observed cases and selected neighbors
            for x in feat_list_nom:
                synth_matrix[x_synth * n + count, x] = [data.iloc[
                    knn_matrix[i, neigh], x], data.iloc[
                    i, x]][round(rd.random())]
            
            ## generate synthetic y response variable by
            ## inverse distance weighted
            synth_matrix[x_synth * n + count, d - 1] = interpolate_target(data, i, knn_matrix[i, neigh], synth_matrix[x_synth * n + count, :d - 1], feat_ranges[:-1])
        
            ## close loop counter
            count = count + 1
    
    ## convert synthetic matrix to dataframe
    data_new = pd.DataFrame(synth_matrix)
    
    ## synthetic data quality check
    if sum(data_new.isnull().sum()) > 0:
        raise ValueError("oops! synthetic data contains missing values")
    
    ## replace label encoded values with original values
    for j in feat_list_nom:
        code_list = data.iloc[:, j].unique()
        cat_list = data_var.iloc[:, j].unique()
        
        for x in code_list:
            data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])
    
    ## reintroduce constant features previously removed
    if len(feat_const) > 0:
        data_new.columns = feat_var
        
        for j in range(len(feat_const)):
            data_new.insert(
                loc = int(feat_const[j]),
                column = feat_const[j], 
                value = np.repeat(
                    data_orig.iloc[0, feat_const[j]], 
                    len(synth_matrix))
            )
    
    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower = 0)
    
    ## return over-sampling results dataframe
    return data_new
