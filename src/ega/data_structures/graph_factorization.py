import abc

class Slicer(object):
    """
    _clustering - a list of the form  [cl_1,....,cl_{h_1}] , where an element  cl_i  corresponds to these vertices for which 
                  the slicer was constructed that correspond to cluster of id i; 
                  the element cl_i is a 2d-array (with rows corresponding to slices) of the form:  
                      cl_i = [l_0, l_1, ...], where l_i is the 1d-array of ids of vertices within distance i  
                      (measured in pre-defined units) from the sub-sampled separator corresponding to the slicer.
    """
    def __init__(self):
        self._clustering = None           # If not None, a list of 2d-arrays


class NodeContent(object):
    """
    _left_slicer - an object of the class Slicer responsible for organizing vertices of the set  A  
                   of the separation  (A,B,S′) given by the sub-sampled separator  S′  
                   (in particular via partitioning them into slices) in a way supporting efficient computation of 
                   this part of the graph field integration that corresponds to cross-terms between  A  and  B ,
    _right_slicer - the analogoue of the above, but for the vertex set  B ,
    _cl_relator - the table  T \in N^{h1 × h2} , where:  
                  h1  and  h2  stand for the number of clusters of the sets:  A  and  B  respectively,
                  and  T[i][j] = min_k(η_i[k] + η_j[k])  for the signature vectors  η_i, η_j in N^{|S′|}  
                  of the representatives of the ith cluster of  A  and jth cluster of  B  respectively 
                  (the signature vector  η  of a vertex  v  is given as:  
                  η = (d_1 − min_{l=1,...,|S′|}d_l, ..., d_{|S′|}, ..., d_{|S′|} − min_{l=1, ..., |S′|}d_l)  for  d_i  
                  being the shortest-path-distance in pre-defined units from the ith vertex of the sub-sampled separator to  v ),
    _bfgf_integrator - object of the class BFGFIntegrator deriving from GFIntegrator (responsible for the brute-force 
                       field integration; used only when the sub-graph corresponding to the tree-node is small enough).
    """    
    def __init__(self):
        self._left_slicer = None         # If not None, of the type Slicer
        self._right_slicer = None        # If not None, of the type Slicer
        self._cl_relator = None          # If not None, a 2d-numpy-array
        self._bfgf_integrator = None     # If not None, of the type BFGFIntegrator


class GraphFactorization(object):
    """
    The presented class, called GraphFactorization, is the core class encoding structural graph decomposition 
    and used by those derivatives of GFIntegrator that apply balanced separations for efficient graph field integration.
    
    It is a binary tree with nodes consisting of the fields: 
        _left_child - pointer to the left sub-tree (None if the sub-tree does not exist), 
        _right_child - pointer to the right-subtree (None if the sub-tree does not exist), 
        _data - an object of the type NodeContent with field containing node data (None if there is no data).
    """
      
    def __init__(self):
        self._left_child = None    # If not None, of the type GraphFactorization
        self._right_child = None   # If not None, of the type GraphFactorization
        self._data = None          # If not None, of the type NodeContent
          

