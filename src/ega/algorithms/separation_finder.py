import abc 

class SeparationFinder(object):
    """ Abstract class for computing balanced separations. """
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def compute_separation(self, adjacency_lists):
        """
        The output of the compute_separation method is a list of length N (N being the number of nodes of the graph) 
        of the values: {0,1,2} where 0 is a label for the first class, 1 is the label for the second class 
        and 2 is thee label for the separator-set.
        """
        raise NotImplementedError("Abstract method")
      
