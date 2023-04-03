import os
# os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from PIL import Image
import random
import math


def n_folds_calc(dataset, folds):
    '''

    :param dataset: Dataframe containing the entire dataset
    :param folds: Number of folds
    :return: Returns a list of indexes equal to the number of folds
    '''
    use_dataframe = dataset

    # Determine the number of entries in the dataset
    entries = use_dataframe.shape[0]
    samples_per_fold = math.floor(entries / folds)

    # Create index list
    entries_list = list(range(0, entries))

    # Shuffle the entries list to randomize sample selection in N-Fold
    random.shuffle(entries_list)
    return_entries = []
    for fold in range(folds):
        return_entries.append(entries_list[0:samples_per_fold])
        entries_list = entries_list[samples_per_fold:]

    return return_entries


def n_folds_split(x_in, indexes, fold):
    # Get test and train samples
    # Get test and train labels
    use_dataframex = pd.DataFrame(x_in)
    tr_x = use_dataframex.drop(indexes[fold])
    tst_x = use_dataframex.loc[use_dataframex.index[indexes[fold]]]

    return tr_x.values, tst_x.values


def scoring_function(estimator, hyper_cube):
    return estimator.score(hyper_cube)


if __name__ == '__main__':
    target_image_path = r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Homework4\pyramid.jpg'
    image_name = 'Pyramid'
    OUTPATH_CSV = r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Homework4\problem2csv\cross_val_gmm'
    OUTPATH_FIGURES = r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Homework4\problem2figures'
    n_components_list = [2,3,4,5,6,7,8,9,10]
    covariance_type_list = ["full", "spherical", "diag"]

    load_csv = False
    image = Image.open(target_image_path)
    image = np.asarray(image)
    hyper_cube = []

    for row_index, row_vector in enumerate(image):

        for pixel_index, pixel in enumerate(row_vector):
            hyper_cube.append([row_index, pixel_index, pixel[0], pixel[1], pixel[2]])

    hyper_cube = np.asarray(hyper_cube)
    hyper_cube = pd.DataFrame(MinMaxScaler().fit_transform(hyper_cube),
                              columns=['row_index', 'column_index', 'r', 'g', 'b'])
    if not load_csv:
        # Perform K-fold cross validation
        # metrics are taken from the scikit learn cross validation recomended metrics (covariance type)
        parameter_grid = {"n_components": n_components_list, "covariance_type": covariance_type_list}
        cv = KFold(n_splits=10, shuffle=True, random_state=1)

        grid = GridSearchCV(GaussianMixture(), param_grid=parameter_grid, verbose=3, cv=cv, n_jobs=-1, scoring=scoring_function)

        grid.fit(hyper_cube.values)

        print(f'Best Parameters: {grid.best_params_}')
        best_params = grid.best_params_
        dataframe_results = pd.DataFrame(grid.cv_results_)
        dataframe_results.to_csv(OUTPATH_CSV + f'_{image_name}.csv')
        print(f'Score: {grid.best_score_}')

    dataframe_results = pd.read_csv(OUTPATH_CSV + f'_{image_name}.csv')
    print(f'Results Loaded...')

    interest_rows = dataframe_results.loc[dataframe_results['rank_test_score'] == 1]
    selected_column = dataframe_results.loc[interest_rows['mean_fit_time'].idxmin()]
    n_component = eval(selected_column['params'])['n_components']
    covariance_type = eval(selected_column['params'])['covariance_type']
    print(f'Hyper parameters selected: {n_component=}, {covariance_type=}')
    # Sort out desired params
    label_list = dataframe_results['params']
    plot_results = []
    for index, row in dataframe_results.iterrows():
        temp_dict = eval(row['params'])
        if temp_dict['covariance_type'] == covariance_type:
            plot_results.append(row['mean_test_score'])
    # Plot likelihoods for different numbers of components and covariance types
    fig0 = plt.figure(0, figsize=(15, 12))
    plt.title(f'Log Likelihood Score vs. Gaussian Components, {covariance_type} Covariance')
    plt.plot(plot_results)
    label_list = dataframe_results['params']
    plt.xticks(np.arange(len(n_components_list)), n_components_list)
    plt.xlabel("Number of Gaussian Components")
    plt.ylabel("Log Likelihood")
    fig0.savefig(OUTPATH_FIGURES + f'/Score_vs_Components' + "_" + image_name)
    fig0.clear()

    classifiers = []
    test_scores = []
    gaussian_range = list(range(n_component, n_component+1))

    for test in gaussian_range:
        print(f'Testing Order of {test}')
        clf = GaussianMixture(n_components=test, covariance_type=covariance_type)
        clf.fit(hyper_cube.values)
        test_scores.append(clf.score(hyper_cube.values))
        classifiers.append((clf, n_component, covariance_type, clf.score(hyper_cube.values)))

    max_score = max(test_scores)
    best_classifier = classifiers[test_scores.index(max_score)]
    best_component = best_classifier[1]
    best_covariance_type = covariance_type
    results = best_classifier[0].predict(hyper_cube.values)

    results = results.reshape(image.shape[0], image.shape[1])
    print(f"Results Calculated...")

    fig1 = plt.figure(0, figsize=(12,9))
    display_image = plt.imshow(image)
    plt.title(f'Original Image {image_name}')
    fig2 = plt.figure(1, figsize=(12,9))
    display = plt.imshow(results)
    plt.title(f'GMM {best_component} Components {image_name} {best_covariance_type} Covariance')
    fig1.savefig(OUTPATH_FIGURES+f'/_{image_name}')
    fig2.savefig(OUTPATH_FIGURES + f'/_{image_name}_{best_component}_best_components_{best_covariance_type}_covariance')

    # 2 Gaussian Distribution Example
    fig3 = plt.figure(2, figsize=(12,9))
    clf = GaussianMixture(n_components=2, covariance_type=covariance_type)
    clf.fit(hyper_cube.values)
    results = clf.predict(hyper_cube.values)
    results = results.reshape(image.shape[0], image.shape[1])
    display = plt.imshow(results)
    plt.title(f'GMM 2 Components {image_name} {best_covariance_type} Covariance')
    fig3.savefig(OUTPATH_FIGURES + f'/_{image_name}_2_components_{best_covariance_type}_covariance')

    # 5 Gaussian Distribution Example
    fig4 = plt.figure(3, figsize=(12,9))
    clf = GaussianMixture(n_components=5, covariance_type=covariance_type)
    clf.fit(hyper_cube.values)
    results = clf.predict(hyper_cube.values)
    results = results.reshape(image.shape[0], image.shape[1])
    display = plt.imshow(results)
    plt.title(f'GMM 5 Components {image_name} {best_covariance_type} Covariance')
    fig4.savefig(OUTPATH_FIGURES + f'/_{image_name}_5_components_{best_covariance_type}_covariance')

    # 5 Gaussian Distributions Spherical Covariance
    fig5 = plt.figure(4, figsize=(12,9))
    clf = GaussianMixture(n_components=5, covariance_type='spherical')
    clf.fit(hyper_cube.values)
    results = clf.predict(hyper_cube.values)
    results = results.reshape(image.shape[0], image.shape[1])
    display = plt.imshow(results)
    plt.title(f'GMM 5 Components {image_name} spherical Covariance')
    fig5.savefig(OUTPATH_FIGURES + f'/_{image_name}_5_components_spherical_covariance')

    # 5 Gaussian Distributions diag Covariance
    fig6 = plt.figure(4, figsize=(12,9))
    clf = GaussianMixture(n_components=5, covariance_type='diag')
    clf.fit(hyper_cube.values)
    results = clf.predict(hyper_cube.values)
    results = results.reshape(image.shape[0], image.shape[1])
    display = plt.imshow(results)
    plt.title(f'GMM 5 Components {image_name} diag Covariance')
    fig6.savefig(OUTPATH_FIGURES + f'/_{image_name}_5_components_diag_covariance')

