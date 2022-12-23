from scipy.stats import ttest_ind
import numpy as np

def welch_test(a, b):
    """Perform a welch test which assumes inequal population variances"""
    res = ttest_ind(a, b, equal_var=False)
    print("T-value: ", np.round(res.statistic,2), " , p-value: ", np.round(res.pvalue,5))


def import_data(filepath, k):
    """
    Import the data from csv file to a numpy array.
    :param filepath: The csv-file where the generated data are stored.
    :param k: (int) the row index (= temperature level) where the simulation data should be retrieved
    :return: (numpy.array) 25*1 array with the current route lengths at temperature level k.
    """
    data_array = np.genfromtxt(filepath, delimiter=',')
    return data_array[k,:]


if __name__ == '__main__':
    filenames = ['data/boltzmann_numit25_a280.csv', 'data/cauchy_numit25_a280.csv',
                 'data/exponential05_numit25_a280.csv',
                 'data/exponential09_numit25_a280.csv',
                 'data/exponential095_numit25_a280.csv',
                 'data/fast_numit25_a280.csv']
    data = np.zeros((25, len(filenames)))
    for i, fn in enumerate(filenames):
        data[:, i] = import_data(fn, 50)

    for i in range(len(filenames)):
        print(f"-------------------- Descriptive statistics of {filenames[i]} ---------------------------")
        print(f"M= ", np.round(np.mean(data[:,i]),2), " , SD = ", np.round(np.std(data[:,i]),2))

    for i in range(len(filenames)):
        for j in range(i+1, len(filenames)):
            print(f"------------- Test {filenames[i]} against {filenames[j]}----------------------")
            welch_test(data[:,i], data[:,j])



