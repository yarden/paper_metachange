##
## Representation of time
##
import numpy as np
import utils

class Time:
    """
    Representation of time interval.

    Terminology:

    - 'bins' represent bins of time interval.
    - 'step size' means the duration that is relevant
      for an event to occur in.
    """
    def __init__(self, t_start, t_end, step_size,
                 rate_unit=1):
        self.t_start = t_start
        self.t_end = t_end
        self.total_len = float(self.t_end - self.t_start)
        self.step_size = step_size
        # duration of rate
        self.rate_unit = rate_unit
        # calculate number of bins to achieve step size
        self.num_bins = \
          round((t_end - t_start) / float(self.step_size)) + 1
        binned_time = np.linspace(t_start, t_end, self.num_bins,
                                  retstep=True)
        self.t = binned_time[0]
        # get exact step size used
        self.step_size = binned_time[1]
        self.num_steps = len(self.t)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Time(%.2f, %.2f, step=%.2f)" %(self.t_start,
                                               self.t_end,
                                               self.step_size)

    def get_num_bins_to(self, interval_len):
        """
        Return number steps (bins) to get to interval
        of the given length.
        """
        num_steps = \
          round((interval_len * (self.num_bins - 1)) / self.total_len)
        return int(num_steps)

    def iter_interval(self, interval_len):
        """
        Iterate by bins of size 'interval_len'
        """
        # Need to add 1 to number of bins here
        # because of how Python slice indexing works.
        # Example: in [0, 1, 2, 3, 4], there are 4 bins.
        # To get to 2, we need 2 bins, but the corresponding
        # slice is from 0:3
        n = self.get_num_bins_to(interval_len) + 1
        for curr_interval in utils.grouper_nofill(self.t, n):
            yield curr_interval

    def iter_interval_ind(self, interval_len, start=0):
        """
        Iterate through time by time interval of length 'interval_len'.
        Yield the time interval (in time space) as well as the start and
        end coordinate of the time interval.
        """
        n = self.get_num_bins_to(interval_len) 
        ind = start
        for curr_interval in utils.grouper_nofill(self.t[start:], n):
            yield curr_interval, (ind, min(ind + n, self.num_steps))
            ind += len(curr_interval)

if __name__ == "__main__":
    my_time = Time(0, 5, 1)
    print my_time
    for n in range(5):
        nbins = my_time.get_num_bins_to(n)
        print "To get %d need %d bins" %(n, nbins)
        assert (nbins == n), "Error"
    my_time2 = Time(0, 10, 11)
    for c in my_time2.iter_interval(3):
        print "time interval: ", c
