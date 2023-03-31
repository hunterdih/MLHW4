import os

os.environ["OMP_NUM_THREADS"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import random
import math
import pandas as pd
import matplotlib.pyplot as plt

RN1 = 2

RP1 = 4

SIGMA = 1

OUTPATH_CSV_SVM = r'D:\Classes\MLResults\HW4\Problem1CSV\svm_results.csv'
OUTPATH_FIGURES_SVM = r'D:\Classes\MLResults\HW4\Problem1Figures'


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


def n_folds_split(x_in, y_in, indexes, fold):
    # Get test and train samples
    # Get test and train labels
    use_dataframex = pd.DataFrame(x_in)
    use_dataframey = pd.DataFrame(y_in)
    tr_x = use_dataframex.drop(indexes[fold])
    tr_y = use_dataframey.drop(indexes[fold])
    tst_x = use_dataframex.loc[use_dataframex.index[indexes[fold]]]
    tst_y = use_dataframey.loc[use_dataframey.index[indexes[fold]]]

    return tr_x, tst_x, tr_y, tst_y


def generate_circle_data(num_samples):
    identity = SIGMA * np.identity(2)
    x_return = []
    y_return = []
    for sample in range(num_samples):
        choice = random.random()
        if choice < 0.5:  # Choose -1
            theta = random.uniform(-np.pi, np.pi)
            n = np.random.multivariate_normal((0, 0), identity, 1)
            cos_sin_theta = np.array((np.cos(theta), np.sin(theta))).T
            x = (RN1 * cos_sin_theta) + n
            x_return.append(x)
            y_return.append(-1)
        else:
            theta = random.uniform(-np.pi, np.pi)
            n = np.random.multivariate_normal((0, 0), identity, 1)
            cos_sin_theta = np.array((np.cos(theta), np.sin(theta))).T
            x = (RP1 * cos_sin_theta) + n
            x_return.append(x)
            y_return.append(1)
    x_return = np.stack(x_return, axis=1)[0]
    y_return = np.asarray(y_return)

    return x_return, y_return


def plot_svm_data_accuracy(scores, c_range, g_range, plot_title='Gamma_vs_C'):
    scores = np.asarray(scores)
    scores = scores.reshape(len(c_range), len(g_range))

    fig0 = plt.figure(0, figsize=(12, 9))
    plt.tight_layout()
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.xticks(np.arange(len(g_range)), g_range, rotation=45)
    plt.yticks(np.arange(len(c_range)), c_range)
    plt.title(plot_title)
    plt.colorbar()
    fig0.savefig(OUTPATH_FIGURES_SVM + '/' + plot_title)
    fig0.clear()


def plot_classifications(test_x, test_y, results, plot_title_cm='Confusion_Matrix', plot_title_class='Classifications'):
    fig1 = plt.figure(1, figsize=(12, 9))
    cm = confusion_matrix(test_y, results, normalize='true')
    plt.imshow(cm, cmap='BuPu')
    for (i, j), label in np.ndenumerate(cm):
        plt.text(j, i, str(round(label, 4)), ha='center', va='center')
    plt.colorbar
    error = round(np.sum(1 - cm.diagonal()) / cm.shape[0], 4)
    plt.title(plot_title_cm)
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label, Error: {error}')
    fig1.savefig(OUTPATH_FIGURES_SVM + '/' + plot_title_cm)
    fig1.clear()

    fig2 = plt.figure(2, figsize=(12, 9))
    marker_list = ['o', '<', 's', 'x', '*']
    plotting_df = pd.DataFrame()
    plotting_df['x0'] = test_x[:, 0]
    plotting_df['x1'] = test_x[:, 1]
    plotting_df['y_test'] = test_y
    plotting_df['results'] = results == test_y
    for l in [-1, 1]:
        label_df = plotting_df[plotting_df['y_test'] == l]
        label_miss = label_df[label_df['results'] == False]
        label_match = label_df[label_df['results'] == True]
        plt.scatter(label_match['x0'], label_match['x1'], color='g', marker=marker_list[l], label=f'Correct Class {l}')
        plt.scatter(label_miss['x0'], label_miss['x1'], color='r', marker=marker_list[l], label=f'Correct Class {l}')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title(plot_title_class)
    plt.tight_layout()
    plt.legend()
    fig2.savefig(OUTPATH_FIGURES_SVM + '/' + plot_title_class)
    fig2.clear()


def mlp_cross_val(x_train, y_train, node_list = [1], activation_function_list = ['relu'],  n_folds = 10):
    indexes = n_folds_calc(x_train, 10)
    num_parameters = x_train.shape[1]

    best_fold_for_node_list = []
    node_loss_list = []
    node_acc_list = []

    for nodes in node_list:

        fold_loss_list = []
        fold_acc_list = []

        for indexes_chosen in indexes:
            x_trn, x_tst, y_trn, y_tst = n_folds_split(x_train, y_train, indexes_chosen)
            model = Sequential()
            model.add(Dense(units=nodes, kernel_initializer='random_uniform', activation='elu', input_dim=num_parameters))
            model.add(Dense(units=2, kernel_initializer='random_uniform', activation='softmax'))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='val_loss')
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
            history = model.fit(x_trn, y_trn, validation_data=(x_tst, y_tst), epochs=10000, verbose=0, callbacks=[es])

            test_loss, test_acc = model.evaluate(x_tst, y_tst, verbose=0)
            fold_loss_list.append(test_loss)
            fold_acc_list.append(test_acc)
            print('Loss: %.3f, Accuracy: %.3f' % (test_acc, test_acc))

        best_fold = fold_loss_list.index(min(fold_loss_list))
        best_fold_for_node_list.append(best_fold)
        node_loss_list.append(fold_loss_list[best_fold])
        node_acc_list.append(fold_acc_list[best_fold])

    # PLOT LOSS VS NODE COUNT
    # PLOT ACCURACY VS NODE COUNT

    optimal_node_range = node_loss_list.index(min(node_loss_list))






if __name__ == '__main__':
    x_train, y_train = generate_circle_data(1000)
    x_test, y_test = generate_circle_data(10000)

    load_from_csv = True
    run_svm = True
    run_mlp = True
    if run_svm:
        if not load_from_csv:
            # Calculate SVM Results Using builtin svm k-fold splitting and cross validation in sklearn
            C_range = np.logspace(-4, 5, 10)
            gamma_range = np.logspace(-6, -2, 5)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = KFold(n_splits=10, shuffle=True, random_state=3)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, verbose=3)
            grid.fit(x_train, y_train)

            print(f'Best Parameters: {grid.best_params_}')
            best_params = grid.best_params_
            dataframe_results = pd.DataFrame(grid.cv_results_)
            dataframe_results.to_csv(OUTPATH_CSV_SVM)
            print(f'Score: {grid.best_score_}')

        else:
            dataframe_results = pd.read_csv(OUTPATH_CSV_SVM)
            print(f'Results Loaded...')

        interest_rows = dataframe_results.loc[dataframe_results['rank_test_score'] == 1]
        selected_column = dataframe_results.loc[interest_rows['mean_fit_time'].idxmin()]
        c = eval(selected_column['params'])['C']
        gamma = eval(selected_column['params'])['gamma']
        print(f'Hyper parameters selected: {c=}, {gamma=}')

        c_test_range = np.linspace(c - 0.5 * c, c + 0.5 * c, 10)
        g_test_range = np.linspace(gamma - 0.5 * gamma, gamma + 0.5 * gamma, 10)
        classifiers = []
        test_scores = []

        print(f'Starting precision evaluation fitting...')
        for C in c_test_range:
            for G in g_test_range:
                clf = SVC(C=C, gamma=G)
                clf.fit(x_train, y_train)
                test_scores.append(clf.score(x_train, y_train))
                classifiers.append((clf, C, G, clf.score(x_train, y_train)))

        max_score = max(test_scores)
        best_classifier = classifiers[test_scores.index(max_score)]
        best_c = best_classifier[1]
        best_g = best_classifier[2]

        print(f'Best classifier parameters and score: {best_c=}, {best_g=}, score={max_score}')
        plot_svm_data_accuracy(test_scores, c_test_range, g_test_range, plot_title='Validation_Accuracy_100_Models')

        print(f'Starting precision evaluation on test data...')
        test_scores = []
        for classifier in classifiers:
            classifier = classifier[0]
            test_scores.append(classifier.score(x_test, y_test))

        plot_svm_data_accuracy(test_scores, c_test_range, g_test_range, plot_title='Test_Accuracy_100_Models')
        predictions = best_classifier[0].predict(x_test)
        plot_classifications(x_test, y_test, predictions, plot_title_cm='10000_Sample_Test_Set_Confusion_Matrix', plot_title_class='SVM_Classifications')

    if run_mlp:


