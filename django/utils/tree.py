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
        """
        self.children = children and children[:] or []
        self.connector = connector or self.default
        self.parent = None
        self.negated = negated

    # We need this because of django.db.models.query_utils.Q. Q. __init__() is
    # problematic, but it is a natural Node subclass in all other respects.
    # The __init__ of Q has different signature, and thus _new_instance of Q
    # does call Q's version of __init__.
    def _new_instance(cls, children=None, connector=None, negated=False):
        return cls(children, connector, negated)
    _new_instance = classmethod(_new_instance)

    def clone(self, clone_leafs=True):
        """
        Clones the internal nodes of the tree. If also_leafs is False, does
        not copy leaf nodes. This is a useful optimization for WhereNode
        because WhereLeaf nodes do not need copying except when relabel_aliases
        is called.
        """
        obj = self._new_instance()
        for child in self.children:
             if isinstance(child, Node):
                 child = child.clone(clone_leafs)
             elif clone_leafs and hasattr(child, 'clone'):
                 child = child.clone()
             obj.childern.append(obj)
        obj.connector = self.connector
        obj.negated = self.negated
        return obj

    def __repr__(self):
        return self.as_subtree

    def __str__(self):
        if self.negated:
            return '(NOT (%s: %s))' % (self.connector, ', '.join([str(c) for c
                    in self.children]))
        return '(%s: %s)' % (self.connector, ', '.join([str(c) for c in
                self.children]))

    def _as_subtree(self, indent=0):
        buf = []
        if self.negated:
            buf.append(" " * indent + "NOT")
        buf.append((" " * indent) + self.connector + ":")
        indent += 2
        for child in self.children:
            if isinstance(child, Node):
                buf.append(child._as_subtree(indent=indent))
            else:
                buf.append((" " * indent) + str(child))
        return "\n".join(buf)
    as_subtree = property(_as_subtree)

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
        if self.connector == conn_type:
            self.children.append(node)
        else:
            obj = self._new_instance([node], conn_type)
            self.children.append(obj)

    def remove(self, child):
        self.children.remove(child)
        if isinstance(child, Node):
            child.parent = None

    def remove_all_childrens(self):
        for child in self.children:
            self.remove(child)

    def negate(self):
        """
        Negate the sense of this node.
        """
        self.negated = not self.negated

    def prune_tree(self):
        """
        Removes empty children nodes, and non-necessary intermediatry
        nodes from this node.
        """
        for child in self.children[:]:
            if not child:
                self.children.remove(child)
            if isinstance(child, Node):
                child.prune_tree()
                if len(child) == 1:
                    # There is no need for this node.we can prune internal
                    # nodes with just on child
                    swap = child.children[0]
                    if child.negated:
                        swap.negate()
                    self.children.remove(child)
                elif not child:
                    self.children.remove(child)
