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

    def clone(self, memo=None):
        """
        Clones the whole tree, not just the subtree. We have loops in
        the tree due to keeping both parent and child links. Because
        of this, we must keep a memo of objects already copied.
        """
        if memo is None:
            memo = {}
        if self in memo:
            return memo[self]
        obj = self._new_instance()
        memo[self] = obj
        for child in self.children:
             if isinstance(child, Node):
                 child = child.clone(memo=memo)
             obj._add(child)
        if self.parent is not None:
            new_parent = self.parent.clone(memo=memo)
            obj.parent = new_parent
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

    def _as_tree(self):
        root = self
        while root.parent:
            root = root.parent
        return root._as_subtree(indent=0)
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

    def _add(self, *nodes):
        """
        A helper method to keep the parent/child links in valid state.
        """
        for node in nodes:
            self.children.append(node)
            if isinstance(node, Node):
                node.parent = self

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
            self._add(node)
        else:
            obj = self._new_instance([node], conn_type)
            obj2 = self.clone()
            self._add(obj, obj2)

    def remove(self, child):
        assert child in self.children
        self.children.remove(child)
        if isinstance(child, Node):
            child.parent = None

    def negate(self):
        """
        Negate the sense of this node.
        """
        self.negated = not self.negated

    def subtree(self, conn_type):
        obj = self._new_instance()
        obj.connector = conn_type
        obj.parent = self
        self.children.append(obj)
        return obj

    def prune_tree(self, recurse=False):
        """
        Removes empty children nodes, and non-necessary intermediatry
        nodes from this node. If recurse is true, will recurse down
        the tree.
        """
        old_childs = self.children[:]
        self.children = []
        for child in old_childs:
            if not child:
                continue
            if isinstance(child, Node):
                if recurse:
                    child.prune_tree(recurse=True)
                if not child.negated and len(child) == 1:
                    child = child.children[0]
            self._add(child)
