import pandas as pd
import user_data_loader as udl
from itertools import product


def make_filename(dir_, experiment, user):
    ''' Produces a filename from input arguments'''
    return dir_ + experiment + '-' + user + '.arff'

#def get_filenames(dir_, experiments, users):
#    ''' Produces a generator with all the filenames '''
#    for exp, user in it.product(experiments, users):
#        yield make_filename(dir_, exp, user)


def build_store(storename, data_dir, experiments, users):
    ''' Builds an HDF store from the data in arff format
        @param storename: the filename of the HDFStore
        @param data_dir: directory where the arff files are located
        @param experiments: list containing the experiment names
        @type experiments: list of strings
        @param users: list with the user ids
        @type users: list of strings
        @return: the hdfstore object with all the datasets from the users
        @rtype: pandas.HDFStore
    '''
    store = pd.HDFStore(storename)
    for exp, user in product(experiments, users):
        filename = make_filename(data_dir, exp, user)
        print filename,
        df = udl.load_user_file(filename)
        store.put(exp + '/' + user, df, format='table')
    return store
