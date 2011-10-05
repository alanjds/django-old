"""
A class for storing a tree graph. Primarily used for filter constructs in the
ORM.
"""

import copy

class Node(object):
    """
    A single internal node in the tree graph. A Node should be viewed as a
    connection (the root) with the children being either leaf nodes or other
    Node instances.
    """
    # Standard connector type. Clients usually won't use this at all and
    # subclasses will usually override the value.
    default = 'DEFAULT'

    def __init__(self, children=None, connector=None, negated=False):
        """
        Constructs a new Node. If no connector is given, the default will be
        used.

        Warning: You probably don't want to pass in the 'negated' parameter. It
        is NOT the same as constructing a node and calling negate() on the
        result.
        """
        self.children = children and children[:] or []
        self.connector = connector or self.default
        self.parent = None
        self.negated = negated

    # We need this because of django.db.models.query_utils.Q. Q. __init__() is
    # problematic, but it is a natural Node subclass in all other respects.
    def _new_instance(cls, children=None, connector=None, negated=False):
        """
        This is called to create a new instance of this class when we need new
        Nodes (or subclasses) in the internal code in this class. Normally, it
        just shadows __init__(). However, subclasses with an __init__ signature
        that is not an extension of Node.__init__ might need to implement this
        method to allow a Node to create a new instance of them (if they have
        any extra setting up to do).
        """
        obj = Node(children, connector, negated)
        obj.__class__ = cls
        return obj
    _new_instance = classmethod(_new_instance)

    def empty(cls):
        return cls._new_instance([])
    empty = classmethod(empty)

    def clone(self, recurse=False):
        assert recurse == False, 'recurse=True not implemented'
        obj = self.empty()
        obj.children = self.children[:]
        obj.parent = self.parent
        obj.connector = self.connector
        obj.negated = self.negated
        return obj
    
    def __deepcopy__(self, memodict):
        """
        Utility method used by copy.deepcopy().
        """
        obj = Node(connector=self.connector, negated=self.negated)
        obj.__class__ = self.__class__
        obj.children = copy.deepcopy(self.children, memodict)
        obj.parent = copy.deepcopy(self.parent, memodict)
        return obj
        
    def __repr__(self):
        return self.as_tree

    def __str__(self):
        if self.negated:
            return '(NOT (%s: %s))' % (self.connector, ', '.join([str(c) for c
                    in self.children]))
        return '(%s: %s)' % (self.connector, ', '.join([str(c) for c in
                self.children]))

    def _as_tree(self, indent=-1):
        """
        Prettyprinter for the whole tree.
        """
        if indent == -1:
            root = self
            while root.parent:
               root = root.parent
            return root._as_tree(indent=0)

        buf = []
        buf.append((" " * indent) + self.connector + ":")
        indent += 2
        if self.negated:
            buf.append(" " * indent + "NOT")
        for child in self.children:
            if isinstance(child, Node):
                buf.append(child._as_tree(indent=indent))
            else:
                buf.append((" " * indent) + str(child))
        return "\n".join(buf)
    as_tree = property(_as_tree)
            
    def __len__(self):
        """
        The size of a node if the number of children it has.
        """
        return len(self.children)

    def __nonzero__(self):
        """
        For truth value testing.
        """
        return bool(self.children)

    def __contains__(self, other):
        """
        Returns True is 'other' is a direct child of this instance.
        """
        return other in self.children

    def add(self, node, conn_type):
        """
        Adds a new node to the tree. If the conn_type is the same as the
        root's current connector type, the node is added to the first level.
        Otherwise, the whole tree is pushed down one level and a new root
        connector is created, connecting the existing tree and the added node.
        """
        if node in self.children and conn_type == self.connector:
            return
        if isinstance(node, Node):
            node.parent = self
        if self.connector == conn_type:
            self.children.append(node)
        else:
            obj = self._new_instance([node], conn_type, False)
            obj2 = self.clone()
            obj.parent = obj2.parent = self
            self.children = [obj, obj2]

    def negate(self):
        """
        Negate the sense of the root connector. This reorganises the children
        so that the current node has a single child: a negated node containing
        all the previous children. This slightly odd construction makes adding
        new children behave more intuitively.

        Interpreting the meaning of this negate is up to client code. This
        method is useful for implementing "not" arrangements.
        """
        self.negated = not self.negated
        """ 
        self.children = [self._new_instance(self.children, self.connector,
                not self.negated)]
        self.connector = self.default
        """

    def subtree(self, conn_type):
        obj = self.empty()
        obj.connector = conn_type
        obj.parent = self
        self.children.append(obj)
        return obj

    def prune_unused_childs(self):
        new_childs = []
        for child in self.children:
            if child:
                new_childs.append(child)
        self.children = new_childs
