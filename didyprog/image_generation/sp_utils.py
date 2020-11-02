import numpy as np


def compute_diff(image):
    '''
    Parameters
    ----------
    image: numpy.ndarray of shape nxn
    Returns
    -------
    below: numpy.ndarray represents image[i][j]-image[i+1][j]
    to_right: numpy.ndarray represents image[i][j]-image[i][j+1]
    '''


    zips = np.zeros(image.shape[-1])[np.newaxis, :]
    zips_above = np.concatenate([zips, image])
    zips_below = np.concatenate([image, zips])

    minus_below = zips_above - zips_below
    minus_below = minus_below[1:, :]
    minus_below[-1, :] = 0

    zips = np.zeros(image.shape[-1])[:, np.newaxis]
    right_side = np.concatenate([image, zips], axis=1)
    left_side = np.concatenate([zips, image], axis=1)
    minus_right = left_side - right_side
    minus_right = minus_right[:, 1:]
    minus_right[:, -1] = 0

    zips = np.zeros(image.shape[-1])[:, np.newaxis]
    zips_and = np.zeros(image.shape[-1]+1)[:, np.newaxis]
    zips_and_across= np.zeros(image.shape[-1]+1)[np.newaxis,:]
    to_right=np.concatenate([image,zips],axis=1)
    zip_se=np.concatenate([to_right,zips_and_across])
    to_left=np.concatenate([zips,image],axis=1)
    zip_nw=np.concatenate([zips_and_across,to_left])
    minus_se = zip_nw - zip_se
    minus_se = minus_se[1:,1:]
    minus_se[-1,:]=0
    minus_se[:,-1]=0

    zips_ne = np.concatenate([zips_and_across,to_right])
    zips_sw = np.concatenate([to_left,zips_and_across])
    minus_sw = zips_ne-zips_sw
    minus_sw = minus_sw[1:,:-1]
    minus_sw[:,0] = 0
    minus_sw[-1,:] = 0



    return minus_right,minus_se,minus_below,minus_sw

if __name__=='__main__':
    a=np.array([[4,7],[13,6]])
    print(compute_diff(a))