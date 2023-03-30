import numpy as np
from satio.features import Features
from sklearn.decomposition import PCA
from skimage.feature import greycoprops


# define a function to calculate PCA of set of features


def pca(features, num_components, scaling_range=None):
    """
    Calculates principal components from a set of features
    :param features -> should be an object of the Features class
    :param num_components -> number of PC components to retain
    :param scaling_range -> provides minimum and maximum of each feature
    for normalization. Should be a dict with the names of the features as keys
    and [min, max] as value.
    Features not included in the dict won't be normalized.
    """

    # prepare the data
    data = features.data
    names = features.names
    nfeat = data.shape[0]
    lenx = data.shape[1]
    leny = data.shape[2]
    new_data = data.copy()

    # normalize data if needed
    if scaling_range is not None:
        for k, v in scaling_range.items():
            idx = features.get_feature_index(k)
            new_data[idx, :, :] = (data[idx, :, :] - v[0]) / (v[1] - v[0])

    # convert data to 2d
    data_2d = new_data.transpose(1, 2, 0).reshape(lenx * leny, nfeat)

    # apply pca and retain n components
    pca = PCA(n_components=num_components)
    pcs_2d = pca.fit_transform(data_2d)

    # reshape to shape required by features
    pcs = pcs_2d.reshape(lenx, leny, num_components).transpose(2, 0, 1)
    pc_names = ["PC{}".format(x) for x in range(1, num_components+1)]

    return Features(pcs, pc_names)


def offset(length, angle):
    """Return the offset in pixels for a given length and angle"""
    dv = length * np.sign(-np.sin(angle)).astype(np.int32)
    dh = length * np.sign(np.cos(angle)).astype(np.int32)
    return dv, dh


def crop(img, center, win):
    """Return a square crop of img centered at center (side = 2*win + 1)"""
    row, col = center
    side = 2*win + 1
    first_row = row - win
    first_col = col - win
    last_row = first_row + side
    last_col = first_col + side
    return img[first_row: last_row, first_col: last_col]


def cooc_maps(img, center, win, d=[1], theta=[0], levels=256):
    """
    Return a set of co-occurrence maps for different d and theta in a square
    crop centered at center (side = 2*w + 1)
    """
    shape = (2*win + 1, 2*win + 1, len(d), len(theta))
    cooc = np.zeros(shape=shape, dtype=np.int32)
    row, col = center
    Ii = crop(img, (row, col), win)
    for d_index, length in enumerate(d):
        for a_index, angle in enumerate(theta):
            dv, dh = offset(length, angle)
            Ij = crop(img, center=(row + dv, col + dh), win=win)
            cooc[:, :, d_index, a_index] = encode_cooccurrence(Ii, Ij, levels)
    return cooc


def encode_cooccurrence(x, y, levels=256):
    """Return the code corresponding to co-occurrence of intensities x and y"""
    return x*levels + y


def decode_cooccurrence(code, levels=256):
    """Return the intensities x, y corresponding to code"""
    return code//levels, np.mod(code, levels)


def compute_glcms(cooccurrence_maps, levels=256):
    """Compute the cooccurrence frequencies of the cooccurrence maps"""
    Nr, Na = cooccurrence_maps.shape[2:]
    glcms = np.zeros(shape=(levels, levels, Nr, Na),
                     dtype=np.float32)
    for r in range(Nr):
        for a in range(Na):
            codes, freqs = np.unique(cooccurrence_maps[:, :, r, a],
                                     return_counts=True)
            freqs = freqs/float(freqs.sum())
            i, j = decode_cooccurrence(codes, levels=levels)
            glcms[i, j, r, a] = freqs
    return glcms


def compute_props(glcms, props, avg):
    """Return a feature vector corresponding to a set of GLCM"""
    Nr, Na = glcms.shape[2:]
    features = np.zeros(shape=(Nr, Na, len(props)))
    for index, prop_name in enumerate(props):
        features[:, :, index] = greycoprops(glcms, prop_name)
    if avg:
        features = np.average(features, axis=(0, 1))
    return features.ravel()


def haralick_features(features,
                      win=2, d=[1], theta=[0, np.pi/4], levels=256,
                      metrics=('contrast',), avg=True,
                      scaling_range={}):
    """
    Returns a set of Haralick features for each input feature
    :param features -> Features object containing all input features
    :param win -> spatial window included when calculating texture
    for given pixel
    The window is defined as (2*win + 1) -> win=2 means a 5x5 pixel window!
    """
    # prepare data
    data = features.data
    names = features.names
    nfeat = data.shape[0]

    text_features = []
    for fname in names:
        idx = features.get_feature_index(fname)
        img = data[idx, :, :]
        # normalize data and transform to byte
        if fname in scaling_range:
            # use scaling range provided
            srange = scaling_range[fname]
            # make sure all values in img fall in this range
            img[img < srange[0]] = srange[0]
            img[img > srange[1]] = srange[1]

        else:
            # calculate minimum and maximum to be used for scaling
            srange = [np.min(img), np.max(img)]
        img = (img - srange[0]) / (srange[1] - srange[0]) * 256
        img = img.astype('uint8')

        rows, cols = img.shape
        margin = win + max(d)
        arr = np.pad(img, margin, mode='reflect')
        if avg:
            n_text_feat = len(metrics)
            text_feat_names = [fname + '_tx_' + p[0:3] for p in metrics]
        else:
            n_text_feat = len(d) * len(theta) * len(metrics)
            text_feat_names = []
            for r in d:
                for t in theta:
                    for p in metrics:
                        text_feat_names.append(
                            fname + '_tx_' + p[0:3] + '_' +
                            str(r) + '_' + str(int(t * 180 / np.pi)))

        feature_map = np.zeros(shape=(rows, cols, n_text_feat),
                               dtype=np.float32)
        for m in range(rows):
            for n in range(cols):
                coocs = cooc_maps(arr, (m + margin, n + margin),
                                  win, d, theta, levels)
                glcms = compute_glcms(coocs, levels)
                feature_map[m, n, :] = compute_props(glcms, metrics, avg)

        feature_map = feature_map.transpose(2, 0, 1)
        text_features.append(Features(feature_map, text_feat_names))

    if len(text_features) > 1:
        text_features = Features.from_features(*text_features)
    elif len(text_features) == 1:
        text_features = text_features[0]

    return text_features
