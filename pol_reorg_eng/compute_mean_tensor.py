import numpy as np

def compute_mean_tensor(xx, xy, xz, yy, yz, zz):
    """
    Computes the mean tensor from the given components.

    Parameters:
        xx (numpy.ndarray): Array of xx components.
        xy (numpy.ndarray): Array of xy components.
        xz (numpy.ndarray): Array of xz components.
        yy (numpy.ndarray): Array of yy components.
        yz (numpy.ndarray): Array of yz components.
        zz (numpy.ndarray): Array of zz components.

    Returns:
        numpy.ndarray: A 3x3 tensor with the mean of the six components.
    """
    # Compute the mean of each component
    mean_xx = np.mean(xx)
    mean_xy = np.mean(xy)
    mean_xz = np.mean(xz)
    mean_yy = np.mean(yy)
    mean_yz = np.mean(yz)
    mean_zz = np.mean(zz)
    
    # Create the mean tensor
    mean_tensor = np.array([[mean_xx, mean_xy, mean_xz],
                            [mean_xy, mean_yy, mean_yz],
                            [mean_xz, mean_yz, mean_zz]])
    
    return mean_tensor
