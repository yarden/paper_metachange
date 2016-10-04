import scipy
import scipy.interpolate 
import numpy

def get_spline_fit(x, y, k=4, s=1):
    """
    Get spline fit.
    """
    spl = scipy.interpolate.UnivariateSpline(x, y, k=k, s=s)
    return spl(x)

def get_spline_derivs_fit(x, y, k=4, s=1):
    """
    Get spline derivatives fit.
    """
    spl = scipy.interpolate.UnivariateSpline(x, y, k=k, s=s)
    derivs = spl.derivative()
    return derivs(x)
