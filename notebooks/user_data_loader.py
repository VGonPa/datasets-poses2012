import numpy as np
import pandas as pd
import arff as arff   # Downloaded from: http://code.google.com/p/arff/
# import matplotlib.pyplot as plt
import itertools as it
import toolz as tlz
from functools import partial

users = ['user' + str(i+1).zfill(2) for i in np.arange(30)]
users.pop(users.index('user12'))  # There is no 'user12' data, so we remove it

header = tuple(['h_seqNum', 'h_stamp', 'user_id'])
joints = tuple(['head', 'neck', 'torso',
                'left_shoulder', 'left_elbow', 'left_hand',
                'right_shoulder', 'right_elbow', 'right_hand',
                'left_hip', 'left_knee', 'left_foot',
                'right_hip', 'right_knee', 'right_foot'])
attribs = tuple(['confidence', 'pos_x', 'pos_y', 'pos_z',
                 'orient_x', 'orient_y', 'orient_z', 'orient_w'])

index = list(it.chain(header,
                      it.imap('_'.join, it.product(joints, attribs)),
                      ['pose', ]))

positions = [i for i in index if '_pos_' in i]
orientations = [i for i in index if '_orient_' in i]
confidences = [i for i in index if '_confidence' in i]
pose = index[-1]

ind_pos_x = [i for i in index if '_pos_x' in i]
ind_pos_y = [i for i in index if '_pos_y' in i]
ind_pos_z = [i for i in index if '_pos_z' in i]


def _prepare_multiindex(joints, attribs):
    '''Helper function that returns all attribs in two iterables.
       1st iterable will be used as the 1st level of the index.
       2nd iterable will be used as the 2nd level of the index.'''
    all_columns = it.izip(*it.product(joints, attribs))
    first_level = \
        it.chain(['header'] * len(header), all_columns.next(), ['pose', ])
    second_level = it.chain(header, all_columns.next(), ['pose', ])
    return first_level, second_level


def make_multiindex(joints, attribs):
    ''' Returns a pandas.MultiIndex from the entered attributes
        @param joints: first level of multiindex
        @type joints: list of strings
        @param joints: second level of multiindex
        @type joints: list of strings
        @return: the Multiindex itself
        @rtype: pandas.MultiIndex
    '''
    multiind1, multiind2 = _prepare_multiindex(joints, attribs)
    return pd.MultiIndex.from_arrays([list(multiind1), list(multiind2)],
                                     names=['joint', 'attrib'])


def normalize_joints(df, from_joint):
    ''' Returns a normalized DataFrame. '''
    return (df[from_joint].copy(),
            df.drop(['header', 'pose', from_joint], axis=1, level='joint')
            .sub(df[from_joint], level=1))


def load_user_file(file_):
    ''' Loads an ARFF file from an experiment
        and returns a DataFrame associated to it '''
    data = arff.load(file_)
    # Converting it to a Numpy array
    data = np.array([list(d) for d in data])
    # Converting the array to a pandas dataFrame
    df = pd.DataFrame(data, columns=index)
    # Setting dtype to float for the numeric columns
    numeric_cols = list(index[3:-1])
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

# Function to clean the index removing the leading 'STAND_' substring
stand_cleaner = lambda x: x.lstrip('STAND_')


def clean_stand_prefix(x):
    '''Removes 'STAND_' prefix of a string'''
    x.lstrip('STAND_')


def prepare_df(df):
    ''' Groups by pose, removes Header and Confidences,
        calculates the mean for all instances of each group,
        and cleans the index names '''
    return df.groupby(pose) \
        .mean().drop(confidences, axis=1).rename(index=stand_cleaner)


def accumulate_indices(df_list):
    ''' Concatenates the indices of a list of pandas.Dataframes
        and returns them as an numpy.array '''
    # get_ind = lambda x: x.index
    # return np.concatenate(map(get_ind, df_list))
    return np.concatenate([x.index() for x in df_list])


def accumulate_users(users):
    ''' Accumulates a list of dataframes in a single dataframe '''
    return pd.concat(users)


def insert_user_name(df, user):
    ''' returns a copy of the entered dataframe,
        but with a column added whith the user id '''
    new_df = df.copy()
    new_df.insert(0, 'user', [user] * len(df))
    return new_df


# parse_filename = lambda user, experiment, dir_: \
#                         dir_ + '/' + experiment + '-' + user + '.arff'
def parse_filename(user, experiment, dir_):
    return dir_ + '/' + experiment + '-' + user + '.arff'

parse_exp03_filename = partial(parse_filename, experiment='exp03',
                               dir_='../data/users_separated', )


def user_pipe(filename):
    ''' A pipe that produces a user dataset (pd.DataFrame)
        starting from a filename.
        Note that the data in the dataset corresponds
        to the means of data found in the file '''
    return tlz.pipe(filename, load_user_file, prepare_df)


def load_all_users():
    ''' Returns a pd.DataFrame with the information of all the users'''
    map = tlz.curry(map)
    dataset = tlz.pipe(users, map(parse_exp03_filename), map(user_pipe),
                       accumulate_users)
    dataset.insert(0, 'user', sorted(users * 3))
    return dataset
