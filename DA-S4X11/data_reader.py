import numpy as np 
import torch

def normalize_data(data):
    data = (data - data.min()) / (data.max() - data.min())
    data = (data - 0.5) * 2 
    return data

def read_bci_data():
    """
    Import and clean the data of BCI competition 3 3b which has two subjects: S4b, X11b
    The experiment consists of 3 sessions for each subject. Each session consists of 4 to 9 runs
    """
    S4b_train = np.load('./S4b_train.npz')
    X11b_train = np.load('./X11b_train.npz')
    S4b_test = np.load('./S4b_test.npz')
    X11b_test = np.load('./X11b_test.npz')

    source_data = np.concatenate((S4b_train['signal'], S4b_test['signal']), axis=0)
    source_label = np.concatenate((S4b_train['label'], S4b_test['label']), axis=0)
    target_data = np.concatenate((X11b_train['signal'], X11b_test['signal']), axis=0)
    target_label = np.concatenate((X11b_train['label'], X11b_test['label']), axis=0)

    source_label = source_label - 1
    target_label = target_label - 1
    source_data = np.transpose(np.expand_dims(source_data, axis=1), (0, 1, 3, 2))
    target_data = np.transpose(np.expand_dims(target_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(source_data))
    source_data[mask] = np.nanmean(source_data)


    mask = np.where(np.isnan(target_data))
    target_data[mask] = np.nanmean(target_data)


    source_data = torch.from_numpy(source_data).float()
    target_data = torch.from_numpy(target_data).float()
    source_label = torch.tensor(source_label, dtype=torch.long)
    target_label = torch.tensor(target_label, dtype=torch.long)
    val_data = target_data
    val_label = target_label

    print(source_data.shape, source_label.shape, val_data.shape, val_label.shape, target_data.shape, target_label.shape)

    source_data = normalize_data(source_data)
    val_data    = normalize_data(val_data)
    target_data = normalize_data(target_data)

    return source_data, source_label, val_data, val_label, target_data, target_label
