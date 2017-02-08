'''
Do some decoding stuff.
'''
from pymeg import preprocessing
from sklearn import cross_validation, svm, pipeline, preprocessing as skpre
from sklearn import decomposition
import numpy as np
import pandas as pd
from joblib import Memory


sensors = dict(
        all= lambda x:x,
        occipital= lambda x:[ch for ch in x if ch.startswith('MLO') or ch.startswith('MRO')],
        posterior= lambda x:[ch for ch in x if ch.startswith('MLP') or ch.startswith('MRP')],
        central= lambda x:[ch for ch in x if ch.startswith('MLC')
            or ch.startswith('MRC') or ch.startswith('MZC')],
        frontal= lambda x:[ch for ch in x if ch.startswith('MLF')
            or ch.startswith('MRF') or ch.startswith('MZF')],
        temporal= lambda x:[ch for ch in x if ch.startswith('MLT') or ch.startswith('MRT')])


def clf():
    return pipeline.Pipeline([
            ('scale', skpre.StandardScaler()),
            ('PCA', decomposition.PCA(n_components=.99)),
            ('SVM', svm.LinearSVC())])

def cv(x):
    return cross_validation.StratifiedShuffleSplit(x, n_iter=10, test_size=0.1)



def decode(classifier, data, labels, train_time, predict_times,
        cv=cross_validation.StratifiedKFold, collapse=np.mean):
    '''
    Apply a classifier to data and predict labels with cross validation.
    Train classifier from data at data[:, :, train_time] and apply to all
    indices in predict_times. Indices are interpreted as an index into
    the data matrix.

    train_time can be a slice object to allow averaging across time or using
    time_sequences for prediction. In this case indexing data with train_time
    potentially results in a (n_epochs, n_channels, n_time) and the time
    dimension needs to be collapsed to obtain a (n_epochs, n_channels) matrix.
    How to do this can be controlled by the collapse keyword. If collapes=='reshape'
    data is coercd into  (n_epochs, n_channels*n_matrix), else collapse is applied
    to data like so:

        >>> X = collapse(data, axis=2)

    If train_time is a slice object, predict_times should be a list of slice objects
    and the test set gets the same treatment as the training set.

    Returns a vector of average accuracies for all indices in predict_idx.

    Parameters
    ----------
    classifier : sklearn classifier object
    data : np.array, (#trials, #features, #time)
    labels : np.array, (#trials,)
    train_time : int
    predict_times : iterable of ints
    '''
    assert len(labels) == data.shape[0]
    results = []

    # With only one label no decoding is possible.
    if len(np.unique(labels)) == 1:
        return pd.DataFrame({
            'fold':[np.nan],
            'train_time':[np.nan],
            'predict_time':[np.nan],
            'accuracy':[np.nan]
            })

    for i, (train_indices, test_indices) in enumerate(cv(labels)):
        np.random.shuffle(train_indices)
        fold = []
        clf = classifier()
        l1, l2 = np.unique(labels)
        l1 = train_indices[labels[train_indices]==l1]
        l2 = train_indices[labels[train_indices]==l2]
        if len(l1)>len(l2):
            l1 = l1[:len(l2)]
        else:
            l2 = l2[:len(l1)]
        assert not any([k in l2 for k in l1])
        train_indices = np.concatenate([l1, l2])

        train = data[train_indices, :, train_time]
        if len(train.shape) == 3:
            if collapse == 'reshape':
                train = train.reshape((train.shape[0], np.prod(train.shape[1:])))
            else:
                train = collapse(train, axis=2)
            train_time = train_time.stop

        clf=clf.fit(train, labels[train_indices])

        for pt in predict_times:
            fold_result = {}
            test = data[test_indices, :, pt]
            if len(test.shape) == 3:
                if collapse == 'reshape':
                    test = test.reshape((test.shape[0], np.prod(test.shape[1:])))
                else:
                    test = collapse(test, axis=2)
                fold_result['predict_time'] = pt.stop
            else:
                fold_result['predict_time'] = pt
            fold_result['train_time'] = train_time
            fold_result.update({
                'fold':i,
                'accuracy': clf.score(test, labels[test_indices])})
            results.append(fold_result)
    return pd.DataFrame(results)


def generalization_matrix(epochs, labels, dt, classifier=clf, cv=cv, slices=False):
    '''
    Get data for a generalization across time matrix.

    Parameters
    ----------
        epochs: mne.epochs object
    Epochs to use for decoding.
        labels : np.array (n_epochs,)
    Target labels to predict
        dt : int
    Time resolution of the decoding in ms (!).
        slices : False, 'reshape' or function
    Indicates how time dimension should be treated during decoding.
    False implies single time point decoding, reshape implies
    using time sequence data for decoding and function can be used
    to reduce time series data to single point.
    '''
    data = epochs._data
    sfreq = epochs.info['sfreq']

    tlen = data.shape[-1]/(float(sfreq)/1000.)
    nsteps = np.around(float(tlen)/dt)
    #steps = np.linspace(0, data.shape[-1]-1, nsteps).astype(int)
    steps = np.arange(0, data.shape[-1], int(data.shape[-1]/nsteps))
    if slices:
        steps = [slice(s, e) for s, e in zip(steps[0:-1], steps[1:])]
        decoder = lambda x: decode(clf, data, labels, x, steps, cv=cv, collapse=slices)
    else:
        decoder = lambda x: decode(clf, data, labels, x, steps, cv=cv)
    return pd.concat([decoder(tt) for tt in steps])


def apply_decoder(func, snum, epoch, label, channels=sensors['all']):
    '''
    Apply a decoder function to epochs from a subject and decode 'label'.

    Parameters
    ----------
        func: function object
    A function that performs the desired decoding. It needs to take two arguments
    that the epoch object and labels to use for the decoing. E.g.:

        >>> func = lambda x,y: generalization_matrix(x, y, 10)

        snum: int
    Subject number to indicate which data to load.
        epoch: str
    One of 'stimulus', 'response', or 'feedback'
        label: str
    Which column in the metadata to use for decoding. Labels will recoded to
    0-(num_classes-1).
    '''
    s, m = preprocessing.get_epochs_for_subject(snum, epoch) #This will cache.
    s = s.pick_channels(channels(s.ch_names))

    # Drop nan labels
    nan_loc = m.index[np.isnan(m.loc[:, label])]
    use_loc = m.index[~np.isnan(m.loc[:, label])]

    m = m.drop(nan_loc)
    s = s[list(use_loc.astype(str))]

    # Sort order index to align epochs with labels.
    m = m.loc[s.events[:, 2]]
    if not all(s.events[:, 2] == m.index.values):
        raise RuntimeError('Indices of epochs and meta do not match! Task: ' + str(snum) + ' ' + epoch + ' ' + label)
    # Recode labels to 0-(n-1)
    labels = m.loc[:, label]
    labels = skpre.LabelEncoder().fit(labels).transform(labels)
    return func(s, labels)
