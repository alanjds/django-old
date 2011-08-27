import copy
from django.core.exceptions import FieldError
from django.db.models.fields import FieldDoesNotExist
from django.db.models.sql import AND, OR
from django.db.models.sql.constants import LOOKUP_SEP, QUERY_TERMS
from django.db.models.sql.expressions import QueryTreeSQLEvaluator as SQLEvaluator
from django.db.models.sql.where import WhereNode, Constraint 
from django.utils.tree import Node

FOLLOW_REVERSE = 'FOLLOW_REVERSE'
FOLLOW_FORWARD = 'FOLLOW_FORWARD'
SUBQ_NOT_EXISTS = 'NOT EXISTS'

def op_sequence_generator():
    """
    Django differentiates qs.filter(op1).filter(op2) from qs.filter(op1, op2)
    so we keep count of when things have been added - this way we can see which
    operations should apply to the same relation chain and which operations need
    a different relation chain when filtering.
    """
    i = 0
    while True:
        i += 1
        yield i
op_sequence_generator = op_sequence_generator()

# A quick note - if Relation and Join cloning is too expensive,
# we can easily do one of the following:
#  - remove clonable variables from Relation and Join, that
#    is, keep the query structure in QueryTree directly.
#  - implement "inplace" querysets
#  - implement lazy cloning (should not be as hard as it first
#    feels like)
# It is very likely that the Relation cloning isn't a bottleneck
# wherenode cloning is more expensive.

class Relation(object):
    def __init__(self, model, op_num, rel_ident=None):
        self.model = model
        self.op_num = op_num
        self.rel_ident = rel_ident or op_sequence_generator.next()
        self.child_joins = []

    def debug_print(self, ident=0):
        child_pretty = ''.join([child.debug_print(ident+2) + '\n' for child in self.child_joins])
        return " " * ident + "model: %s rel_ident: %s, alias: %s, joins:\n%s" % (self.model, self.rel_ident, getattr(self, 'alias', '-----'), child_pretty)
    
    def clone(self):
        c = self.__class__(self.model, self.op_num, self.rel_ident)
        c.child_joins = [child.clone(c) for child in self.child_joins]
        return c

    def from_clause_sqlish(self):
        return self.model._meta.db_table + ' AS ' + self.alias

    def rel_idents_to_aliases(self, prefix):
        change_map = self.generate_change_map(prefix)
        self.set_aliases(change_map)
        return change_map
    
    def generate_change_map(self, prefix):
        rel_idents = self.collect_rel_idents()
        rel_idents.sort()
        return dict([(rel_ident, prefix + str(i + 1)) for i, rel_ident in enumerate(rel_idents)])
    
    def collect_rel_idents(self):
        rel_idents = [self.rel_ident]
        for child in self.child_joins:
            rel_idents += child.to_rel.collect_rel_idents()
        return rel_idents
        
    def set_aliases(self, change_map):
        self.alias = change_map[self.rel_ident]
        for child in self.child_joins:
            child.to_rel.set_aliases(change_map)

    def add_join(self, join_part, filter_info, parent_negated):
        if join_part[1] == FOLLOW_FORWARD:
            to_model = join_part[0].rel.to
        else:
            to_model = join_part[0].model
        to_rel = Relation(to_model, filter_info[0])
        
        join = Join(self, to_rel, [join_part[0:2]], filter_info[0])
        join.negated = filter_info[3] or parent_negated
        join.has_or = filter_info[2] == OR
        join.multirow = join_part[2]
        self.child_joins.append(join)
        return to_rel

class Join(object):
    def __init__(self, from_rel, to_rel, join_fields, op_num, extra_cond=None):
        self.from_rel = from_rel
        self.to_rel = to_rel
        self.join_fields = join_fields
        self.op_num = op_num
        self.extra_cond = extra_cond
        self.multirow = False
        self.has_or = False
        self.negated = False
        self.nullable = False

    def clone(self, from_rel):
        to_rel = self.to_rel.clone()
        c = self.__class__(from_rel, to_rel, self.join_fields, self.op_num, self.extra_cond)
        c.multirow = self.multirow
        c.has_or = self.has_or
        c.negated = self.negated
        c.nullable = self.nullable
        return c

    def debug_print(self, ident=0):
        return (" " * ident + "join conds, hasor: %s, negated: %s, nullable: %s, multirow: %s\n" + \
               " " * ident + "to rel: " + self.to_rel.debug_print(ident + 2) + '\n') % \
               (self.has_or, self.negated, self.nullable, self.multirow)

    def join_clause_sqlish(self):
        if self.nullable:
            join_type = 'LEFT JOIN '
        else:
            join_type = 'JOIN '
        ret = "" + join_type + self.to_rel.from_clause_sqlish() + " ON " 
        for field, direction in self.join_fields:
            if direction == FOLLOW_FORWARD:
                from_col = field.column
                to_col = field.rel.get_related_field().column 
            else:
                from_col = field.rel.get_related_field().column
                to_col = field.column
            ret += "%s.%s = %s.%s" % (
                self.from_rel.alias, from_col,
                self.to_rel.alias, to_col
            )
        for join in self.to_rel.child_joins:
            ret += "\n  " + join.join_clause_sqlish()
        return ret

    def can_reuse(self, join_part, filter_info):
        # the first condition is that this is a join to
        # the same model and the same field
        join_cond = join_part[0:2]
        if not self.join_fields == [join_cond]:
            return False
        if not self.multirow:
            return True
        # multirow and different operation - the filter can target
        # different row.
        if self.op_num != filter_info[0]:
            return False
        return True

    def update_info(self, connection_info):
        self.has_or = self.has_or or connection_info[2] == OR
 

class QueryTree(object):
    query_terms = QUERY_TERMS
    # Note: all instance state setup is done either in
    # prepare_new, or in clone()
    def start_new_op(self):
        self.op_num = op_sequence_generator.next()

    def prepare_new(self, model):
        self.model = model
        self.start_new_op()
        self.base_rel = Relation(model, self.op_num)
        self.selects = dict([(self.base_rel.rel_ident, f) for f in model._meta.fields])
        self.where = WhereNode()
        self.having = WhereNode()
        self.aggregates = {}
        self.aggregate_select = {}
    
    def get_meta(self):
        return self.model._meta
       
    def clone(self):
        c = self.__class__()
        c.model = self.model
        c.base_rel = self.base_rel.clone()
        c.selects = self.selects.copy()
        # Almost all of time in cloning is used in here
        c.where = copy.deepcopy(self.where)
        c.having = copy.deepcopy(self.having)
        c.start_new_op()
        c.aggregates = self.aggregates.copy()
        c.aggregate_select = self.aggregate_select.copy()
        return c

    def get_field_by_name(self, opts, filter_name, allow_explicit_fk=True):
        """
        A wrapper to opts.get_field_by_name, so that 'foo_id' -> 'foo'
        translation does not clutter the main code. TODO: Move to Options.
        """ 
        try:
            field, model, direct, m2m = opts.get_field_by_name(filter_name)
        except FieldDoesNotExist:
            for f in opts.fields:
                if allow_explicit_fk and filter_name == f.attname:
                    # XXX: A hack to allow foo_id to work in values() for
                    # backwards compatibility purposes. If we dropped that
                    # feature, this could be removed.
                    field, model, direct, m2m = opts.get_field_by_name(f.name)
                    break
            else:
                names = opts.get_all_field_names() + self.aggregate_select.keys()
                raise FieldError("Cannot resolve keyword %r into field. "
                        "Choices are: %s" % (filter_name, ", ".join(names)))
        return field, model, direct, m2m

    
    def need_force_having(self, q_object):
        """
        Returns whether or not all elements of this q_object need to be put
        together in the HAVING clause.
        """
        for child in q_object.children:
            if isinstance(child, Node):
                if self.need_force_having(child):
                    return True
            else:
                if child[0].split(LOOKUP_SEP)[0] in self.aggregates:
                    return True
        return False
    
    def filter_chain_to_join_path(self, filter_chain):
        """
        Turns the passed in filter_chain (for example [related_model, id])
        into relations and final_field.
        
        The relations will be in the format 
            [field, direction, ...]
        Direction is either FOLLOW_FORWARD or FOLLOW_REVERSE.
        The format is slightly inconvenient in that the direct flag informs if
        the field lives in the related model or in the previous model of the 
        join chain. This is because ForeignKey fields live only in one model 
        and there is no ReverseForeignKey (is there?). The fields in relations 
        list will always be ForeignKeys, OneToOneKeys or GenericForeignKeys.

        The final field will be the last field in the chain. This field is a 
        local field of the last model in the chain. 
        
        Note that one filter_chain part can lead to multiple relations due to 
        model inheritance and many_to_many filtering.
        """
        relations, final_field = [], None
        opts = self.model._meta
        final_field = None
        for pos, filter_name in enumerate(filter_chain):
            assert final_field == None
            if filter_name == 'pk':
                filter_name = opts.pk.name 
            field, model, direct, m2m = self.get_field_by_name(opts, filter_name)
            if model:
                proxied_model = get_proxied_model(opts)
                for int_model in opts.get_base_chain(model):
                    if int_model is proxied_model:
                        opts = internal_model._meta
                        continue
                    o2o_field = opts.get_ancestor_link(int_model)
                    relations.append((o2o_field, FOLLOW_FORWARD))
                    opts = int_model._meta
            if m2m:
                if not direct:
                    field = field.field
                through_model = field.rel.through
                through_opts = through_model._meta
                through_field1_name = field.m2m_field_name()
                field1, _, _, _ = through_opts.get_field_by_name(through_field1_name)
                through_field2_name = field.m2m_reverse_field_name()
                field2, _, _, _ = through_opts.get_field_by_name(through_field2_name)
                if direct:
                    relations.append((field1, FOLLOW_REVERSE))
                    relations.append((field2, FOLLOW_FORWARD))
                    opts = field.rel.to._meta
                else:
                    relations.append((field2, FOLLOW_REVERSE))
                    relations.append((field1, FOLLOW_FORWARD))
                    opts = field.opts
            else:
                if direct and field.rel:
                    relations.append((field, FOLLOW_FORWARD))
                    opts = field.rel.to._meta
                elif direct: 
                    final_field = field
                else:
                    field = field.field
                    relations.append((field, FOLLOW_REVERSE))
                    opts = field.opts 
        if final_field is None:
            final_field = relations.pop()[0]
        return relations, final_field

    def append_m2m_points(self, join_parts):
        with_m2m_info = []
        for part in join_parts:
            field, direction = part
            if direction != FOLLOW_REVERSE or field.unique:
                with_m2m_info.append((field, direction, False))
            else:
                with_m2m_info.append((field, direction, True))
        return with_m2m_info


    def trim_join_path(self, join_path, final_field, lookup_type, value):
        """
        Removes non-necessary relations from the end.
        """
        pos = len(join_path) - 1
        while pos >= 0:
            field, direction = join_path[pos]
            if direction == FOLLOW_REVERSE or field.rel.get_related_field() <> final_field:
                # Cannot trim this - the field is in the related model and there is no foreign
                # key to it. Thus, we must have the related model in the query.
                break
            final_field = field
            join_path.pop()
            pos -= 1
        return final_field

    def add_join_path(self, join_path, filter_info, to_rel=None, parent_negated=False):
        if to_rel is None:
            to_rel = self.base_rel
        if len(join_path) == 0:
            return to_rel.rel_ident
        current_join, rest_of_joins = join_path[0], join_path[1:]
        for child in to_rel.child_joins:
            if child.can_reuse(current_join, filter_info):
                return self.add_join_path(rest_of_joins, filter_info, child.to_rel, child.negated)
        # No child was reusable - need to do a new join! 
        new_rel = to_rel.add_join(current_join, filter_info, parent_negated)
        return self.add_join_path(rest_of_joins, filter_info, new_rel)
 

    def add_filter(self, filter_expr, connector=AND, negate=False, where_node=None):
        filter_chain, value = filter_expr
        filter_chain = filter_chain.split(LOOKUP_SEP)
        if not filter_chain:
             raise FieldError("Cannot parse keyword query %r" % arg)

        # Work out the lookup type and remove it from 'parts', if necessary.
        if len(filter_chain) == 1 or filter_chain[-1] not in self.query_terms:
            lookup_type = 'exact'
        else:
            lookup_type = filter_chain.pop()
        
        # Interpret '__exact=None' as the sql 'is NULL'; otherwise, reject all
        # uses of None as a query value.
        if value is None:
            if lookup_type != 'exact':
                raise ValueError("Cannot use None as a query value")
            lookup_type = 'isnull'
            value = True
        elif callable(value):
            value = value()
        elif hasattr(value, 'evaluate'):
            # If value is a query expression, evaluate it
            value = SQLEvaluator(value, self)
            having_clause = value.contains_aggregate

        join_path, final_field = self.filter_chain_to_join_path(filter_chain)
        final_field = self.trim_join_path(join_path, final_field, lookup_type, value)
        # TODO - the m2m check could be done easily already in filter_chaing_to...
        join_path_with_m2m = self.append_m2m_points(join_path)
        filter_info = (self.op_num, lookup_type, connector, negate, value)
        final_rel_ident = self.add_join_path(join_path_with_m2m, filter_info)
        where_node.add(
            (Constraint(final_rel_ident, final_field.column, final_field), 
             lookup_type, value), connector
        )
        

    def add_q(self, q_object, force_having=False, needs_prune=True, parent_negated=False):
        if hasattr(q_object, 'add_to_query'):
            q_object.add_to_query(self, used_aliases)
            return
        if q_object.connector == OR and not force_having:
            force_having = self.need_force_having(q_object)
        if force_having:
            where_node = self.having
        else:
            where_node = self.where
        subtree = False
        if where_node and q_object.connector != AND and len(q_object) > 1:
            where_node.start_subtree(AND)
            subtree = True
        connector = AND
        for child in q_object.children:
            where_node.start_subtree(connector)
            if isinstance(child, Node):
                self.add_q(child, force_having=force_having, needs_prune=False, parent_negated=parent_negated or q_object.negated)
            else:
                self.add_filter(child, connector, q_object.negated or parent_negated, where_node=where_node)
            where_node.end_subtree()
            connector = q_object.connector 
        if q_object.negated:
            where_node.negate() 
        if subtree:
            where_node.end_subtree()
        if needs_prune:
            self.prune_tree()

    def prune_tree(self):
        pass
        # print "prune doing nothing at all..."


    def as_sqlish(self, ident=0, prefix='T'):
        assert ord(prefix) <= ord('Z')
        self.prepare_for_execution(prefix)
        buf = []
        buf.append(" " * ident + "SELECT ... ")
        buf.append("  FROM " + self.base_rel.from_clause_sqlish())
        for child in self.base_rel.child_joins:
            buf.append("  " + child.join_clause_sqlish())
        from django.db import connection
        def qn(val):
            return val
        where_sql, where_params = self.where.as_sql(qn, connection)
        if where_sql or self.subqueries:
            buf.append(" WHERE " + where_sql.strip())
            for subq_join, subq_where, subq_strat in self.subqueries:
                 subq = self.__class__()
                 subq.prepare_new(subq_join.to_rel.model)
                 subq.base_rel = subq_join.to_rel
                 subq.where = subq_where
                 subq.select = {} # denotes "select 1"
                 # The following monstrosity is what happens when coding tired
                 buf.append(
                     " " * (ident + 7) + ((where_sql and "AND") or '') +
                     " NOT EXISTS (\n" 
                     + subq.as_sqlish(ident=ident + 10, prefix=chr(ord(prefix)+1)) + 
                     '\n ' + " " * (ident + 7) + ')'
                 )

        return (u'\n' + u' ' * ident) .join(buf) % tuple(where_params)
    
    def final_prune(self):
        return

    def prepare_for_execution(self, prefix):
        # does various tasks related to preparing the query
        # for execution. Many of these things are SQL specific
        self.final_prune()
        self.subqueries = []
        self.extract_subqueries(self.base_rel)
        change_map = self.base_rel.rel_idents_to_aliases(prefix)
        self.where.relabel_aliases(change_map)
        self.having.relabel_aliases(change_map)
    
    def extract_subqueries(self, rel):
        for child in rel.child_joins:
            if child.has_or and child.negated and child.multirow:
                rel.child_joins.remove(child)         
                subq_base_rel = child.to_rel
                base_rel_idents = subq_base_rel.collect_rel_idents()
                subq_where = self.where.__class__()
                self.extract_subq_params(self.where, base_rel_idents, subq_where)
                subquery = (child, subq_where, SUBQ_NOT_EXISTS)
                self.subqueries.append(subquery)
                continue
            self.extract_subqueries(child.to_rel)
    
    def extract_subq_params(self, extract_from, idents, extract_to):
        connector = extract_from.connector
        extract_to.start_subtree(connector)
        for child in extract_from.children:
            if isinstance(child, Node):
                self.extract_subq_params(child, idents, extract_to)
            else:
                if child[0].alias in idents:
                    extract_from.children.remove(child)
                    super(WhereNode, extract_to).add(child, connector)
        extract_to.end_subtree()
