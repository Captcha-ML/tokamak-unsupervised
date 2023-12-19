import pandas as pd
import glob 
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#import umap
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

sns.color_palette("Set2")

Machine_inputs = ['isbaffled', 'ip', 'b0', 'nel', 'ptot', 'pdiv', 'q95', 'betan', 'kappa', 'deltaavg', 'deltaupp', 'deltalow', 'gapin', 'gapout', 'zmag', 'rmag', 'rmin', 'lpar_ot', 'zeff']
Physical_inputs = ["TS_Te_87_93", "TS_Ne_87_93", "TS_Te_93_99", "TS_Ne_93_99", "nel", "ptot", "pmid", "q95", "betan", "lpar_ot", "lpol_io", "prad", "zeff", "nu_sol"]


def get_training_data(include_time, include_shotnumber, shot_indices = range(0,30)):
    """
    Given a list of indices in the 0 to 60 (excluded) range, 
    this function returns DataFrame X and Series y composed of the shots at those given indices
    
    sometimes we may or may not want to include time and and shotnumber which is why we include the two boolean arguments
    
    return:
        X: Dataframe with the machine inputs as columns/features and samples as rows
        y: Series with the labels of the associated samples in X
        desired_columns: list of feature names, aka the columns of X and y in that order
    """
    file_names = glob.glob(os.path.join('', f'QCEH_data/TCV_DATAno*.parquet'))
    df_list = [pd.read_parquet(x).drop(columns=['alpha', 'H98y2calc'], errors='ignore') for x in file_names]
    df_training = pd.concat([df_list[idx] for idx in shot_indices], ignore_index=True)

    desired_columns = Machine_inputs + ["LHD_label"]
    if include_time:
        desired_columns.insert(0,"time")
    if include_shotnumber:
        desired_columns.insert(0,"shotnumber")
        
    df_data_analysis = df_training[desired_columns] 

    X = df_data_analysis.drop(["LHD_label"], axis=1)
    y = df_data_analysis["LHD_label"]
    
    scaler = StandardScaler()
    # we don't want to normalize the time and shotnumber columns
    if include_time or include_shotnumber:
        unchanged_columns = X.iloc[:, :include_time+include_shotnumber].values
        
        X_standardized_columns = scaler.fit_transform(X.iloc[:,include_time+include_shotnumber:])
        
        X_standardized = np.column_stack((unchanged_columns, X_standardized_columns))
    else:
        X_standardized = scaler.fit_transform(X)
        
    return X_standardized, y, desired_columns

def perform_umap(X, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=0
    ).fit(X)
    return fit, fit.transform(X)

def perform_pca(X, n_components = 2):
    return PCA().fit_transform(X)

def draw_reduced_space(components, s_y, n_components=2, legend_labels = None, legend_title = "LHD Label", title=''):
    """
    Draws reduced data (components) on a 2D graph or 3D depending on the number of components (n_components)
    and displays the labels (s_y)
    
    legend_labels must be the labels of all the unique values in s_y (increasing order)
    """
    if legend_labels is None:
        legend_mapping = {1: 'L-mode', 2: 'QCE H-mode', 3: 'ELMy H-mode'}
        unique_labels = sorted(np.unique(s_y))
        legend_labels = [legend_mapping[label] for label in unique_labels if label in legend_mapping]

        # If there's a label in s_y not in legend_mapping, handle it appropriately
        for label in unique_labels:
            if label not in legend_mapping:
                print(f"Warning: Label {label} is not defined in legend_mapping.")
                legend_labels.append(f"Unknown Label {label}")

    
    
    if n_components == 1:
        print("That's not a very interesting plot, go for more than a single component")
        return
        #plt.scatter(components[:,0], range(len(components)), c=s_y)
    elif n_components == 2:
        fig = plt.figure(figsize=(10,5))
        sc = plt.scatter(components[:,0], components[:,1], c=s_y, s=10, alpha=0.5)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif n_components == 3:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(projection='3d')
        #ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(components[:,0], components[:,1], components[:,2], c=s_y, s=10, alpha=0.5)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_box_aspect(aspect=None, zoom=0.85) # otherwise component 3 is cutoff
    else:
        print("That number of components can't be displayed on a graph")
    
    plt.legend(handles=sc.legend_elements()[0], labels=legend_labels, loc='upper right', title = legend_title)
    plt.title(title, fontsize=12)
    plt.show()