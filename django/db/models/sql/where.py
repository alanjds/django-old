"""
Code to manage the creation and SQL rendering of 'where' constraints.
"""
import datetime
from itertools import repeat

from django.utils import tree
from django.db.models.fields import Field
from django.db.models.sql.aggregates import Aggregate
from datastructures import EmptyResultSet, FullResultSet

# Connection types
AND = 'AND'
OR = 'OR'

class WhereLeaf(object):
    """
    Represents a leaf node in a where tree. Contains single constrain,
    and knows how to turn it into sql and params.
    """
    # We could ofcourse test isinstance, but this is prettier.
    is_leaf = True

    def __init__(self, data, negated=False):
        self.sql = ''
        self.negated = negated
        self.params = []
        self.match_all = False
        self.match_nothing = False
        if not isinstance(data, (list, tuple)):
            self.data = data
        else:
            # Preprocess the data
            obj, lookup_type, value = data

            if hasattr(value, '__iter__') and hasattr(value, 'next'):
                # Consume any generators immediately, so that we can determine
                # emptiness and transform any non-empty values correctly.
                value = list(value)

            # The "annotation" parameter is used to pass auxilliary information
            # about the value(s) to the query construction. Specifically, datetime
            # and empty values need special handling. Other types could be used
            # here in the future (using Python types is suggested for consistency).
            if isinstance(value, datetime.datetime):
                annotation = datetime.datetime
            elif hasattr(value, 'value_annotation'):
                annotation = value.value_annotation
            else:
                annotation = bool(value)

            if hasattr(obj, "prepare"):
                value = obj.prepare(lookup_type, value)
            self.data = (obj, lookup_type, annotation, value) 

    def create_sql(self, qn, connection):
        if hasattr(self.data, 'as_sql'):
            self.sql, self.params = self.data.as_sql(qn, connection)
        else:
            self.sql, self.params = self.make_atom(qn, connection)
        if self.negated and self.sql:
            self.sql = 'NOT ' + self.sql

    def as_sql(self):
        return self.sql, self.params

    def make_atom(self, qn, connection):
        """
        Turn a tuple (table_alias, column_name, db_type, lookup_type,
        value_annot, params) into valid SQL.

        Returns the string for the SQL fragment and the parameters to use for
        it.
        """
        lvalue, lookup_type, value_annot, params_or_value = self.data
        if hasattr(lvalue, 'process'):
            lvalue, params = lvalue.process(lookup_type, params_or_value, connection)
        else:
            params = Field().get_db_prep_lookup(lookup_type, params_or_value,
                connection=connection, prepared=True)
        if isinstance(lvalue, tuple):
            # A direct database column lookup.
            field_sql = self.sql_for_columns(lvalue, qn, connection)
        else:
            # A smart object with an as_sql() method.
            field_sql = lvalue.as_sql(qn, connection)

        if value_annot is datetime.datetime:
            cast_sql = connection.ops.datetime_cast_sql()
        else:
            cast_sql = '%s'

        if hasattr(params, 'as_sql'):
            extra, params = params.as_sql(qn, connection)
            cast_sql = ''
        else:
            extra = ''

        if (len(params) == 1 and params[0] == '' and lookup_type == 'exact'
            and connection.features.interprets_empty_strings_as_nulls):
            lookup_type = 'isnull'
            value_annot = True

        if lookup_type in connection.operators:
            format = "%s %%s %%s" % (connection.ops.lookup_cast(lookup_type),)
            return (format % (field_sql,
                              connection.operators[lookup_type] % cast_sql,
                              extra), params)

        if lookup_type == 'in':
            if not value_annot:
                raise EmptyResultSet
            if extra:
                return ('%s IN %s' % (field_sql, extra), params)
            max_in_list_size = connection.ops.max_in_list_size()
            if max_in_list_size and len(params) > max_in_list_size:
                # Break up the params list into an OR of manageable chunks.
                in_clause_elements = ['(']
                for offset in xrange(0, len(params), max_in_list_size):
                    if offset > 0:
                        in_clause_elements.append(' OR ')
                    in_clause_elements.append('%s IN (' % field_sql)
                    group_size = min(len(params) - offset, max_in_list_size)
                    param_group = ', '.join(repeat('%s', group_size))
                    in_clause_elements.append(param_group)
                    in_clause_elements.append(')')
                in_clause_elements.append(')')
                return ''.join(in_clause_elements), params
            else:
                return ('%s IN (%s)' % (field_sql,
                                        ', '.join(repeat('%s', len(params)))),
                        params)
        elif lookup_type in ('range', 'year'):
            return ('%s BETWEEN %%s and %%s' % field_sql, params)
        elif lookup_type in ('month', 'day', 'week_day'):
            return ('%s = %%s' % connection.ops.date_extract_sql(lookup_type, field_sql),
                    params)
        elif lookup_type == 'isnull':
            return ('%s IS %sNULL' % (field_sql,
                (not value_annot and 'NOT ' or '')), ())
        elif lookup_type == 'search':
            return (connection.ops.fulltext_search_sql(field_sql), params)
        elif lookup_type in ('regex', 'iregex'):
            return connection.ops.regex_lookup(lookup_type) % (field_sql, cast_sql), params

        raise TypeError('Invalid lookup_type: %r' % lookup_type)
        
    def subtree_contains_aggregate(self):
        return (isinstance(self.data[0], Aggregate) or 
               (hasattr(self.data[2], 'contains_aggregate') and
                self.data[2].contains_aggregate))
    
    def sql_for_columns(self, data, qn, connection):
        """
        Returns the SQL fragment used for the left-hand side of a column
        constraint (for example, the "T1.foo" portion in the clause
        "WHERE ... T1.foo = 6").
        """
        table_alias, name, db_type = data
        if table_alias:
            lhs = '%s.%s' % (qn(table_alias), qn(name))
        else:
            lhs = qn(name)
        return connection.ops.field_cast_sql(db_type) % lhs

    def relabel_aliases(self, change_map):
        if isinstance(self.data[0], (list, tuple)):
            elt = list(self.data[0])
            if elt[0] in change_map:
                elt[0] = change_map[elt[0]]
                self.data = (tuple(elt),) + self.data[1:]
        else:
            self.data[0].relabel_aliases(change_map)

            # Check if the query value also requires relabelling
            if hasattr(self.data[3], 'relabel_aliases'):
                self.data[3].relabel_aliases(change_map)
    
    def clone(self):
        clone = self.__class__(None, self.negated)
        if hasattr(self.data[3], 'clone'):
            new_data3 = self.data[3].clone()
        else:
            new_data3 = self.data[3]
        clone.data = (self.data[0].clone(), self.data[1], self.data[2], new_data3)
        return clone

    def negate(self):
        self.negated = not self.negated

    def __str__(self):
        return "%s%s, %s, %s" % (self.negated and 'NOT: ' or '',
                                 self.data[0], self.data[1], self.data[3])

class WhereNode(tree.Node):
    """
    Used to represent the SQL where-clause.

    The class is tied to the Query class that created it (in order to create
    the correct SQL).

    The children in this tree are usually either Q-like objects or lists of
    [table_alias, field_name, db_type, lookup_type, value_annotation,
    params]. However, a child could also be any class with as_sql() and
    relabel_aliases() methods.
    """

    default = AND
    is_leaf = False

    def __init__(self, *args, **kwargs):
        super(WhereNode, self).__init__(*args, **kwargs)
        self.match_all = False
        self.match_nothing = False

    def clone(self, *args, **kwargs):
        c = super(WhereNode, self).clone(*args, **kwargs)
        c.match_all = self.match_all
        c.match_nothing = self.match_nothing
        return c

    def leaf_class(cls):
        # Subclass hook
        return WhereLeaf
    leaf_class = classmethod(leaf_class)

    def final_prune(self, qn, connection):
        """
        This will do the final pruning of the tree, that is, removing parts
        of the tree that must match everything / nothing.

        Due to the fact that the only way to get to know that is calling
        as_sql(), we will at the same time turn the leaf nodes into sql.
        """
        for child in self.children:
            if child.is_leaf:
                child.create_sql(qn, connection)
            else:
                child.final_prune(qn, connection)
            if child.match_all:
                 if self.connector == OR:
                     self.match_all = True
                     self.remove_all_children()
                     break
                 self.remove(child)
            if child.match_nothing:
                 if self.connector == AND:
                     self.match_nothing = True
                     self.remove_all_children()
                     break
                 self.remove(child)
        if not self.children:
            self.match_all = True 
        if self.negated:
            self.match_all, self.match_nothing = self.match_nothing, self.match_all

    def split_aggregates(self, having, parent=None):
        if self.connector == OR:
             if self.subtree_contains_aggregate():
                 having.add(self, connector=OR)
                 parent.remove(self)
        else:
             if self.negated:
                 neg_node = having._new_instance(negated=True)
                 having.add(neg_node, AND)
                 having = neg_node
             for child in self.children[:]:
                 if child.is_leaf:
                     if child.subtree_contains_aggregate():
                         having.add(child, AND)
                         self.remove(child)
                 else:
                     child.split_aggregates(having, self) 

    def subtree_contains_aggregate(self):
        """
        Returns whether or not all elements of this q_object need to be put
        together in the HAVING clause.
        """
        for child in self.children:
            return child.subtree_contains_aggregate()
        return False

    def as_sql(self):
        """
        Turns this tree into SQL and params. It is assumed that leaf nodes are already
        TODO: rename, and have as_sql implement the normal as_sql(qn, connection)
        interface.
        """
        if not self.children:
            return '', []
        sql_snippets, params = [], []
        for child in self.children:
            child_sql, child_params = child.as_sql()
            sql_snippets.append(child_sql); params.extend(child_params)

        # The tree should have been pruned before this. This means that every
        # node except for the root MUST contain sql. Root is handled
        # differently before calling this method.
        assert sql_snippets
        conn = ' %s ' % self.connector
        sql_string = conn.join(sql_snippets)
        if self.negated:
            sql_string = 'NOT (%s)' % sql_string
        elif len(self.children) != 1:
            sql_string = '(%s)' % sql_string
        return sql_string, params

    def get_group_by(self, group_by):
        for child in self.children:
             if isinstance(child, tuple):
                 group_by.add((child[0].alias, child[0].col))
             elif hasattr(child, 'get_group_by'):
                 child.get_group_by(group_by)
        return group_by


    def relabel_aliases(self, change_map, node=None):
        """
        Relabels the alias values of any children. 'change_map' is a dictionary
        mapping old (current) alias values to the new values.
        """
        for child in self.children:
            child.relabel_aliases(change_map)

class ExtraWhere(object):
    def __init__(self, sqls, params):
        self.sqls = sqls
        self.params = params

    def as_sql(self, qn=None, connection=None):
        return " AND ".join(self.sqls), tuple(self.params or ())

    def clone(self):
        return self

class Constraint(object):
    """
    An object that can be passed to WhereNode.add() and knows how to
    pre-process itself prior to including in the WhereNode.
    """
    
    def __init__(self, alias, col, field):
        self.alias, self.col, self.field = alias, col, field
                                   

    def __getstate__(self):
        """Save the state of the Constraint for pickling.

        Fields aren't necessarily pickleable, because they can have
        callable default values. So, instead of pickling the field
        store a reference so we can restore it manually
        """
        obj_dict = self.__dict__.copy()
        if self.field:
            obj_dict['model'] = self.field.model
            obj_dict['field_name'] = self.field.name
        del obj_dict['field']
        return obj_dict

    def __setstate__(self, data):
        """Restore the constraint """
        model = data.pop('model', None)
        field_name = data.pop('field_name', None)
        self.__dict__.update(data)
        if model is not None:
            self.field = model._meta.get_field(field_name)
        else:
            self.field = None

    def prepare(self, lookup_type, value):
        if self.field:
            return self.field.get_prep_lookup(lookup_type, value)
        return value

    def process(self, lookup_type, value, connection):
        """
        Returns a tuple of data suitable for inclusion in a WhereNode
        instance.
        """
        # Because of circular imports, we need to import this here.
        from django.db.models.base import ObjectDoesNotExist
        try:
            if self.field:
                params = self.field.get_db_prep_lookup(lookup_type, value,
                    connection=connection, prepared=True)
                db_type = self.field.db_type(connection=connection)
            else:
                # This branch is used at times when we add a comparison to NULL
                # (we don't really want to waste time looking up the associated
                # field object at the calling location).
                params = Field().get_db_prep_lookup(lookup_type, value,
                    connection=connection, prepared=True)
                db_type = None
        except ObjectDoesNotExist:
            raise EmptyResultSet

        return (self.alias, self.col, db_type), params

    def relabel_aliases(self, change_map):
        if self.alias in change_map:
            self.alias = change_map[self.alias]

    def clone(self):
        return Constraint(self.alias, self.col, self.field)

    def __str__(self):
        return "%s.%s" % (self.alias, self.col)
