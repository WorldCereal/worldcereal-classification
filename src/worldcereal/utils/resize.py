import numpy as np


def AgERA5_resize(data, scaling=32, shape=None):

    nb, nt, nx, ny = data.shape

    if shape is not None:
        scalex = int(np.ceil(shape[0] / nx))
        scaley = int(np.ceil(shape[1] / ny))
    else:
        scalex = scaling
        scaley = scaling

    arri = []
    for i in range(nx):
        arrj = []
        for j in range(ny):
            arrj.append(np.broadcast_to(data[..., i, j, None, None], (nb, nt, scalex, scaley)))
        arri.append(np.concatenate(arrj, axis=3))
    newdata = np.concatenate(arri, axis=2)

    if shape is not None:
        newdata = newdata[..., :shape[0], :shape[1]]

    return newdata
