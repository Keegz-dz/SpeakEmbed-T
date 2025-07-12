"""
Profiler utility for measuring and summarizing execution time of code blocks/functions.
"""

from time import perf_counter as timer
from collections import OrderedDict
import numpy as np


class Profiler:
    def __init__(self, summarize_every=5, disabled=False):
        """
        Initialize the Profiler.
        Args:
            summarize_every (int): Number of ticks before summarizing.
            disabled (bool): If True, disables profiling.
        """
        self.last_tick = timer()
        self.logs = OrderedDict()
        self.summarize_every = summarize_every
        self.disabled = disabled
    
    def tick(self, name):
        """
        Record the time since the last tick for a named code section.
        Args:
            name (str): Name of the code section/function.
        """
        if self.disabled:
            return
        
        # Log the time needed to execute that function
        if name not in self.logs:
            self.logs[name] = []
        if len(self.logs[name]) >= self.summarize_every:
            self.summarize()
            self.purge_logs()
        self.logs[name].append(timer() - self.last_tick)
        
        self.reset_timer()
        
    def purge_logs(self):
        """
        Clear all logs for all code sections.
        """
        for name in self.logs:
            self.logs[name].clear()
    
    def reset_timer(self):
        """
        Reset the timer to the current time.
        """
        self.last_tick = timer()
    
    def summarize(self):
        """
        Print a summary of the average and stddev of execution times for each code section.
        """
        n = max(map(len, self.logs.values()))
        assert n == self.summarize_every
        print("\nAverage execution time over %d steps:" % n)

        name_msgs = ["%s (%d/%d):" % (name, len(deltas), n) for name, deltas in self.logs.items()]
        pad = max(map(len, name_msgs))
        for name_msg, deltas in zip(name_msgs, self.logs.values()):
            print("  %s  mean: %4.0fms   std: %4.0fms" % 
                  (name_msg.ljust(pad), np.mean(deltas) * 1000, np.std(deltas) * 1000))
        print("", flush=True)    
        