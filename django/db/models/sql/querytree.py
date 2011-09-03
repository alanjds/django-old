import copy

from django.core.exceptions import FieldError
from django.db import connections, DEFAULT_DB_ALIAS
from django.db.models.fields import FieldDoesNotExist
from django.db.models.sql import aggregates as base_aggregates_module
from django.db.models.sql.constants import *
from django.db.models.sql.datastructures import Empty
from django.db.models.sql.expressions import QueryTreeSQLEvaluator as SQLEvaluator
from django.db.models.sql.where import WhereNode, Constraint, AND, OR
from django.utils.datastructures import SortedDict
from django.utils.encoding import force_unicode
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

class Relation(object):
    def __init__(self, model, op_num, rel_ident=None):
        self.model = model
        self.op_num = op_num
        self.rel_ident = rel_ident or op_sequence_generator.next()
        self.child_joins = []

    def clone(self):
        c = self.__class__(self.model, self.op_num, self.rel_ident)
        c.child_joins = [child.clone(c) for child in self.child_joins]
        return c

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

    def add_join(self, join_part, op_num, parent_negated):
        if join_part[1] == FOLLOW_FORWARD:
            to_model = join_part[0].rel.to
        else:
            to_model = join_part[0].model
        to_rel = Relation(to_model, op_num)
        
        join = Join(self, to_rel, [join_part[0:2]], op_num)
        join.negated = parent_negated
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
        # REFACTOR: these should be calculated in pre_sql_setup
        # REFACTOR: are these even interesting?
        # The answer might be: no, except for nullable
        self.has_or = False
        self.negated = False
        self.nullable = False

    def _join_cols(self):
        """
        REFACTOR: use from_field, to_field format in join_fields
        from the beginning...
        """
        cols = []
        for field, direction in self.join_fields:
            if direction == FOLLOW_FORWARD:
                from_field = field.column
                to_field = field.rel.get_related_field()
            else:
                from_field = field.rel.get_related_field()
                to_field = field
            cols.append((from_field, to_field))
        return cols
    join_cols = property(_join_cols)

    def clone(self, from_rel):
        to_rel = self.to_rel.clone()
        c = self.__class__(
            from_rel, to_rel, self.join_fields,
            self.op_num, self.extra_cond)
        c.multirow = self.multirow
        c.has_or = self.has_or
        c.negated = self.negated
        c.nullable = self.nullable
        return c

    def can_reuse(self, join_part, op_num):
        # the first condition is that this is a join to
        # the same model and the same field
        join_cond = join_part[0:2]
        if not self.join_fields == [join_cond]:
            return False
        if not self.multirow:
            return True
        # multirow and different operation - the filter might target
        # a different row.
        if self.op_num != op_num:
            return False
        return True

    def reverse_negates(self):
        self.negated = not self.negated
        for join in self.to_rel.child_joins:
            join.reverse_negates() 
 

class QueryTree(object):
    aggregates_module = base_aggregates_module
    compiler = 'QueryTreeSQLCompiler'
    query_terms = QUERY_TERMS.copy()
    query_terms.update({'exists': None})

    def __init__(self, model, where=WhereNode):
        self.where_class = where
        self.start_new_op()
        self.model = model
        self.base_rel = Relation(model, self.op_num)
        self.select = []
        self.filter_ops = {}
        self.having_ops = {}
        self.aggregates = {}
        self.aggregate_select_mask = None
        self._aggregate_select_cache = None
        self.select_related = False
        self.max_depth = 5
        self.ordering_aliases = []
        self.select_for_update = False
        self.low_mark = 0
        self.high_mark = None
        self.prefix = 'T'
        self.distinct = False
        self.order_by = []
        self.ordering_aliases = []
        self.standard_ordering = True
        self.default_ordering = True
        self.extra = SortedDict()  # Maps col_alias -> (col_sql, params).
        self.extra_select_mask = None
        self._extra_select_cache = None
        self.included_inherited_models = self.get_default_inherited_models()
        self.extra_tables = ()
        self.extra_order_by = ()
        self.deferred_loading = (set(), True)

    def start_new_op(self):
        self.op_num = op_sequence_generator.next()
        self.reverse_polish = []

    def get_meta(self):
        return self.model._meta
       
    def clone(self, memo=None):
        # Clone only clones!
        c = Empty()
        c.where_class = self.where_class
        c.__class__ = self.__class__
        c.model = self.model
        c.base_rel = self.base_rel.clone()
        c.select = self.select[:]
        # Almost all of time in cloning is used in here
        # There is no need to deep-copy filter_ops
        # This will save a _lot_ of time
        c.filter_ops = self.filter_ops.copy()
        c.having_ops = self.having_ops.copy()
        # this needs still thinking - this definitely is not cloning...
        c.filter_ops[self.op_num] = self.reverse_polish
        c.start_new_op()
        c.ordering_aliases = []
        c.aggregates = copy.deepcopy(self.aggregates, memo=memo)
        if self.aggregate_select_mask is None:
            c.aggregate_select_mask = None
        else:
            c.aggregate_select_mask = self.aggregate_select_mask.copy()
        # _aggregate_select_cache cannot be copied, as doing so breaks the
        # (necessary) state in which both aggregates and
        # _aggregate_select_cache point to the same underlying objects.
        # It will get re-populated in the cloned queryset the next time it's
        # used.
        c._aggregate_select_cache = None
        c.select_related = self.select_related
        c.max_depth = self.max_depth
        c.select_for_update = self.select_for_update
        c.low_mark, c.high_mark = self.low_mark, self.high_mark
        c.distinct = self.distinct
        c.prefix = self.prefix
        c.order_by = self.order_by[:]
        c.default_ordering = self.default_ordering
        c.standard_ordering = self.standard_ordering
        c.ordering_aliases = []
        c.extra = self.extra.copy() 
        c.included_inherited_models = self.included_inherited_models.copy()

        # A tuple that is a set of model field names and either True, if these
        # are the fields to defer, or False if these are the only fields to
        # load.
        self.deferred_loading = (set(), True)
        c.extra = self.extra.copy()
        if self.extra_select_mask is None:
            c.extra_select_mask = None
        else:
            c.extra_select_mask = self.extra_select_mask.copy()
        if self._extra_select_cache is None:
            c._extra_select_cache = None
        else:
            c._extra_select_cache = self._extra_select_cache.copy()
        c.extra_tables = self.extra_tables
        c.extra_order_by = self.extra_order_by
        c.deferred_loading = copy.deepcopy(self.deferred_loading, memo=memo)
        return c

    def get_default_inherited_models(self):
        return {}
    
    def set_aggregate_mask(self, names):
        "Set the mask of aggregates that will actually be returned by the SELECT"
        if names is None:
            self.aggregate_select_mask = None
        else:
            self.aggregate_select_mask = set(names)
        self._aggregate_select_cache = None
    
    def clear_deferred_loading(self):
        """
        Remove any fields from the deferred loading set.
        """
        self.deferred_loading = (set(), True)
    
    def deferred_to_data(self, target, callback):
        """
        Converts the self.deferred_loading data structure to an alternate data
        structure, describing the field that *will* be loaded. This is used to
        compute the columns to select from the database and also by the
        QuerySet class to work out which fields are being initialised on each
        model. Models that have all their fields included aren't mentioned in
        the result, only those that have field restrictions in place.

        The "target" parameter is the instance that is populated (in place).
        The "callback" is a function that is called whenever a (model, field)
        pair need to be added to "target". It accepts three parameters:
        "target", and the model and list of fields being added for that model.
        """
        field_names, defer = self.deferred_loading
        if not field_names:
            return
        columns = set()
        orig_opts = self.model._meta
        seen = {}
        if orig_opts.proxy:
            must_include = {orig_opts.proxy_for_model: set([orig_opts.pk])}
        else:
            must_include = {self.model: set([orig_opts.pk])}
        for field_name in field_names:
            parts = field_name.split(LOOKUP_SEP)
            cur_model = self.model
            opts = orig_opts
            for name in parts[:-1]:
                old_model = cur_model
                source = opts.get_field_by_name(name)[0]
                cur_model = opts.get_field_by_name(name)[0].rel.to
                opts = cur_model._meta
                # Even if we're "just passing through" this model, we must add
                # both the current model's pk and the related reference field
                # to the things we select.
                must_include[old_model].add(source)
                add_to_dict(must_include, cur_model, opts.pk)
            field, model, _, _ = opts.get_field_by_name(parts[-1])
            if model is None:
                model = cur_model
            add_to_dict(seen, model, field)

        if defer:
            # We need to load all fields for each model, except those that
            # appear in "seen" (for all models that appear in "seen"). The only
            # slight complexity here is handling fields that exist on parent
            # models.
            workset = {}
            for model, values in seen.iteritems():
                for field, m in model._meta.get_fields_with_model():
                    if field in values:
                        continue
                    add_to_dict(workset, m or model, field)
            for model, values in must_include.iteritems():
                # If we haven't included a model in workset, we don't add the
                # corresponding must_include fields for that model, since an
                # empty set means "include all fields". That's why there's no
                # "else" branch here.
                if model in workset:
                    workset[model].update(values)
            for model, values in workset.iteritems():
                callback(target, model, values)
        else:
            for model, values in must_include.iteritems():
                if model in seen:
                    seen[model].update(values)
                else:
                    # As we've passed through this model, but not explicitly
                    # included any fields, we have to make sure it's mentioned
                    # so that only the "must include" fields are pulled in.
                    seen[model] = values
            # Now ensure that every model in the inheritance chain is mentioned
            # in the parent list. Again, it must be mentioned to ensure that
            # only "must include" fields are pulled in.
            for model in orig_opts.get_parent_list():
                if model not in seen:
                    seen[model] = set()
            for model, values in seen.iteritems():
                callback(target, model, values)


    def deferred_to_columns_cb(self, target, model, fields):
        """
        Callback used by deferred_to_columns(). The "target" parameter should
        be a set instance.
        """
        table = model._meta.db_table
        if table not in target:
            target[table] = set()
        for field in fields:
            target[table].add(field.column)
    
    def clear_select_fields(self):
        """
        Clears the list of fields to select (but not extra_select columns).
        Some queryset types completely replace any existing list of select
        columns.
        """
        self.select = []
        self.select_fields = []

    def get_loaded_field_names(self):
        return []

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
        has_m2m = False
        for pos, filter_name in enumerate(filter_chain):
            assert final_field == None
            if filter_name == 'pk':
                filter_name = opts.pk.name 
            field, model, direct, m2m = self.get_field_by_name(opts, filter_name)
            if m2m:
                has_m2m = True
            if model:
                proxied_model = get_proxied_model(opts)
                for int_model in opts.get_base_chain(model):
                    if int_model is proxied_model:
                        opts = internal_model._meta
                        continue
                    o2o_field = opts.get_ancestor_link(int_model)
                    relations.append((o2o_field, FOLLOW_FORWARD, False))
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
                    relations.append((field1, FOLLOW_REVERSE, True))
                    relations.append((field2, FOLLOW_FORWARD, False))
                    opts = field.rel.to._meta
                else:
                    # something fishy here...
                    relations.append((field2, FOLLOW_REVERSE, True))
                    relations.append((field1, FOLLOW_FORWARD, False))
                    opts = field.opts
            else:
                if direct and field.rel:
                    relations.append((field, FOLLOW_FORWARD, False))
                    opts = field.rel.to._meta
                elif direct: 
                    final_field = field
                else:
                    field = field.field
                    relations.append((field, FOLLOW_REVERSE, True))
                    opts = field.opts 
        if final_field is None:
            final_field = relations.pop()[0]
        return relations, final_field, has_m2m

    def trim_join_path(self, join_path, final_field, lookup_type, value):
        """
        Removes non-necessary relations from the end.
        """
        pos = len(join_path) - 1
        while pos >= 0:
            field, direction, _ = join_path[pos]
            if (direction == FOLLOW_REVERSE or
                field.rel.get_related_field() <> final_field):
                # Cannot trim this - the field is in the related model, and the field's
                # value is not stored in local model foreign key.
                # Thus, we must have the related model in the query.
                break
            final_field = field
            join_path.pop()
            pos -= 1
        return final_field

    def add_join_path(self, join_path, to_rel, parent_negated=False):
        if len(join_path) == 0:
            return to_rel.rel_ident
        current_join, rest_of_joins = join_path[0], join_path[1:]
        for child in to_rel.child_joins:
            if child.can_reuse(current_join, self.op_num):
                child.negated = child.negated or parent_negated
                return self.add_join_path(rest_of_joins, child.to_rel, child.negated)
        # No child was reusable - need to do a new join! 
        new_rel = to_rel.add_join(current_join, self.op_num, parent_negated)
        return self.add_join_path(rest_of_joins, new_rel, parent_negated)
 

    def add_filter(self, filter_expr, connector=AND, negate=False):
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

        if lookup_type == 'exists':
            # Exists is a very special kind of an lookup - 
            # it doesn't relate to any field in the query.
            filter_ = (Constraint('', '', None), lookup_type, value), connector
            self.reverse_polish.append(filter_)
            return
        join_path, final_field, _ = self.filter_chain_to_join_path(filter_chain)
        final_field = self.trim_join_path(
            join_path, final_field, lookup_type, value)
        final_rel_ident = self.add_join_path(
            join_path, self.base_rel, parent_negated=negate)

        filter_ = (
            (Constraint(final_rel_ident, final_field.column, final_field), 
             lookup_type, value), connector)
        self.reverse_polish.append(filter_)


    def add_q(self, q_object, force_having=False, 
              needs_prune=True, parent_negated=False):
        # This could be cleaned up...
        if hasattr(q_object, 'add_to_query'):
            q_object.add_to_query(self, used_aliases)
            return
        if q_object.connector == OR and not force_having:
            force_having = self.need_force_having(q_object)
           
        connector = AND
        first = True
        for child in q_object.children:
            if isinstance(child, Node):
                self.add_q(child, force_having=force_having, needs_prune=False, parent_negated=parent_negated or q_object.negated)
            else:
                self.add_filter(
                    child, connector=connector, 
                    negate=q_object.negated or parent_negated)
            connector = q_object.connector 
            if not first:
                self.reverse_polish.append(q_object.connector)
            first = False
        if q_object.negated:
            self.reverse_polish.append('NOT')
        if needs_prune:
            self.prune_tree()

    def prune_tree(self):
        pass
        # print "prune doing nothing at all..."

    def final_prune(self):
        return

    def bump_prefix(self):
        self.prefix = chr(ord(self.prefix)+1)
        assert ord(self.prefix) <= ord('Z')

    def prepare_for_execution(self):
        # does various tasks related to preparing the query
        # for execution. Many of these things are SQL specific, so
        # at least some of these might belong to qtcompiler...
        self.final_prune()
        self.subqueries = []
        self.extract_subqueries(self.base_rel)
        change_map = self.base_rel.rel_idents_to_aliases(self.prefix)
        self.aliases = set(change_map.values())
        # We will relabel the aliases in filter_ops, so we need a deepcopy
        filter_ops_values = copy.deepcopy(self.filter_ops.values())
        self.where = self.filter_ops_to_where(filter_ops_values)
        self.where.relabel_aliases(change_map)
        having_ops_values = copy.deepcopy(self.having_ops.values())
        self.having = self.filter_ops_to_where(having_ops_values)
        self.having.relabel_aliases(change_map)
        import ipdb; ipdb.set_trace()
        if not self.select:
            self.select = [(self.base_rel.rel_ident, f.column) for f in self.get_meta().fields]
        self.select_cols = []
        for s in self.select:
            if hasattr(s, "relabel_aliases"):
                s.relabel_aliases(change_map)
                self.select_cols.append(s)
            else:
                self.select_cols.append((change_map[s[0]], s[1]))
    
    def extract_subqueries(self, rel):
        for child in rel.child_joins:
            if child.negated and child.multirow:
                rel.child_joins.remove(child) 
                subq_base_rel = child.to_rel
                cond, joins = self.extract_single_subq(subq_base_rel)
                # collect all the relations in the subtree
                subq = self.__class__(subq_base_rel.model)
                subq.base_rel = subq_base_rel
                if cond[-1] != 'NOT':
                    cond.append('NOT')
                else:
                    cond.pop()
                subq.base_rel.child_joins += joins
                # Add a connection to the outer query
                for from_field, to_field in child.join_cols:
                    cond.append(
                        ((Constraint(child.from_rel.rel_ident, from_field.column, 
                                   from_field),
                        'exact', SimpleExpr(child.to_rel.rel_ident, to_field.column)), 
                        AND)
                    )
                    #subq.where.add(
                    #                        #)
                    pass
                subq.filter_ops = {child.op_num: cond}
                subq.select = [] # -> will result in "select 1"
                self.add_filter(('__exists', subq), negate=True) 
                if self.reverse_polish[-1] != 'NOT':
                    self.reverse_polish.append('NOT')
                else:
                    self.reverse_polish.pop()
                if len(self.reverse_polish) > 2:
                    self.reverse_polish.append('AND')
                self.filter_ops[self.op_num] = self.reverse_polish
                subq.bump_prefix()
                subq.prepare_for_execution()
            else:
                self.extract_subqueries(child.to_rel)
    
    def extract_single_subq(self, subq_base_rel):
        extracted_op_cond = None
        for key, conditions in self.filter_ops.items():
            # TODO: cleanup
            found = False
            for cond in conditions:
                if not isinstance(cond, tuple):
                    continue
                if cond[0][0].alias == subq_base_rel.rel_ident:
                    found = True
                    break
            if found:
                extracted_op_cond = conditions
                del self.filter_ops[key]
                break
        assert extracted_op_cond, "Well, this should never happen..."
        # We need to collect those joins that need to be pushed
        # to the subquery - that is all those joins which are referenced
        # in the extracted_op_cond, and which are multirow.
        # The joins below subq_base_rel are already, well, below it.
        joins = []
        extract_joins_for_idents =[cond[0][0].alias for cond in extracted_op_cond if isinstance(cond, tuple)]  
        self.collect_related_joins(
            base_rel=self.base_rel,
            idents=extract_joins_for_idents,
            collect_to=joins
        )
        #where = self.filter_ops_to_where([extracted_op_cond])
        #where.negate()
        return extracted_op_cond, joins

    def collect_related_joins(self, base_rel, idents, collect_to):
        for join in base_rel.child_joins:
            if join.to_rel.rel_ident in idents and join.multirow:
                join.reverse_negates()
                collect_to.append(join)
                base_rel.child_joins.remove(join)
            else:
                self.collect_related_joins(join.to_rel, idents, collect_to)

    def filter_ops_to_where(self, filter_ops):
        # takes a list of filter operations in reverse polish notation
        # turns the list into where tree. 
        where = self.where_class()
        where.start_subtree(AND)
        for reverse_polish in filter_ops:
            self.reverse_polish_to_where(reversed(reverse_polish), where)
        where.end_subtree()
        return where

    def reverse_polish_to_where(self, reverse_iter, where, parent_connector=AND, ops=0):
        for op in reverse_iter:
            if op == 'NOT':
                self.reverse_polish_to_where(
                    reverse_iter, where, parent_connector, ops=1
                )
                where.negate()
            elif op in (AND, OR):
                if op != parent_connector:
                    where.start_subtree(op)
                self.reverse_polish_to_where(reverse_iter, where, op, ops=2)
                if op != parent_connector:
                    where.end_subtree()
            else:
                ops -= 1
                where.add(*op)
            if ops == 0:
                return

    def __iter__(self):
        # Always execute a new query for a new iterator.
        # This could be optimized with a cache at the expense of RAM.
        self._execute_query()
        if not connections[self.using].features.can_use_chunked_reads:
            # If the database can't use chunked reads we need to make sure we
            # evaluate the entire query up front.
            result = list(self.cursor)
        else:
            result = self.cursor
        return iter(result)

    def clear_ordering(self):
        return

    def set_limits(self, low=0, high=None):
        return

    def get_compiler(self, using=None, connection=None):
        if using is None and connection is None:
            raise ValueError("Need either using or connection")
        if using:
            connection = connections[using]

        # Check that the compiler will be able to execute the query
        for alias, aggregate in self.aggregate_select.items():
            connection.ops.check_aggregate_support(aggregate)
        c = self.clone()
        c.prepare_for_execution()
        return connection.ops.compiler(c.compiler)(c, connection, using)

    def has_results(self, using):
        c = self.clone()
        c.add_extra({'a': 1}, None, None, None, None, None)
        c.select = []
        c.clear_ordering(True)
        c.set_limits(high=1)
        compiler = c.get_compiler(using=using)
        return bool(compiler.execute_sql(SINGLE))

    def get_initial_alias(self):
        return None

    def setup_joins(self, names, opts, alias, dupe_multis, allow_many=True,
            allow_explicit_fk=False, negate=False, trim=False):
        """
        Dummy wrapper for backwards compatibility - does something like old
        setup_joins...
        """
        join_path, field, has_many = self.filter_chain_to_join_path(names)
        if not allow_many and has_many:
            raise MultiJoin(-1)
        if trim:
            final_field = self.trim_join_path(join_path)
        target_rel_ident = self.add_join_path(join_path, self.base_rel, negate)
        return field, None, None, [target_rel_ident], None, None

    def get_columns(self):
        return None

    def relabel_aliases(self, change_map):
        self.where.relabel_aliases(change_map)
        self.having.relabel_aliases(change_map)
        self.select_cols = [(change_map[col[0]], col[1].column) for col in self.select]

    def can_filter(self):
        """
        Returns True if adding filters to this instance is still possible.

        Typically, this means no limits or offsets have been put on the results.
        """
        return not self.low_mark and self.high_mark is None
    
    def __str__(self):
        """
        Returns the query as a string of SQL with the parameter values
        substituted in (use sql_with_params() to see the unsubstituted string).

        Parameter values won't necessarily be quoted correctly, since that is
        done by the database interface at execution time.
        """
        sql, params = self.sql_with_params()
        return sql % params

    def sql_with_params(self):
        """
        Returns the query as an SQL string and the parameters that will be
        subsituted into the query.
        """
        return self.get_compiler(DEFAULT_DB_ALIAS).as_sql()
    
    def add_ordering(self, *ordering):
        """
        Adds items from the 'ordering' sequence to the query's "order by"
        clause. These items are either field names (not column names) --
        possibly with a direction prefix ('-' or '?') -- or ordinals,
        corresponding to column positions in the 'select' list.

        If 'ordering' is empty, all ordering is cleared from the query.
        """
        errors = []
        for item in ordering:
            if not ORDER_PATTERN.match(item):
                errors.append(item)
        if errors:
            raise FieldError('Invalid order_by arguments: %s' % errors)
        if ordering:
            self.order_by.extend(ordering)
        else:
            self.default_ordering = False

    def clear_ordering(self, force_empty=False):
        """
        Removes any ordering settings. If 'force_empty' is True, there will be
        no ordering in the resulting query (not even the model's default).
        """
        self.order_by = []
        self.extra_order_by = ()
        if force_empty:
            self.default_ordering = False
    
    def set_extra_mask(self, names):
        """
        Set the mask of extra select items that will be returned by SELECT,
        we don't actually remove them from the Query since they might be used
        later
        """
        if names is None:
            self.extra_select_mask = None
        else:
            self.extra_select_mask = set(names)
        self._extra_select_cache = None
    
    def add_fields(self, field_names, allow_m2m=True):
        """
        Adds the given (model) fields to the select set. The field names are
        added in the order specified.
        """
        for name in field_names:
             join_path, final_field, has_m2m = self.filter_chain_to_join_path(
                 name.split(LOOKUP_SEP))
             if has_m2m and not allow_m2m:
                 raise FieldError("Invalid field name: '%s'" % name)
             field = self.trim_join_path(
                 join_path, final_field, lookup_type=None, value=None)
             final_rel_ident = self.add_join_path(join_path, self.base_rel)
             col = field.column
             self.select.append((final_rel_ident, col))
             self.select_fields.append(field)
        self.remove_inherited_models()
    
    def remove_inherited_models(self):
        self.included_inherited_models = {}
    
    def get_count(self, using):
        """
        Performs a COUNT() query using the current filter constraints.
        """
        obj = self.clone()
        if len(self.select) > 1 or self.aggregate_select:
            # If a select clause exists, then the query has already started to
            # specify the columns that are to be returned.
            # In this case, we need to use a subquery to evaluate the count.
            from django.db.models.sql.subqueries import AggregateQuery
            subquery = obj
            subquery.clear_ordering(True)
            subquery.clear_limits()

            obj = AggregateQuery(obj.model)
            try:
                obj.add_subquery(subquery, using=using)
            except EmptyResultSet:
                # add_subquery evaluates the query, if it's an EmptyResultSet
                # then there are can be no results, and therefore there the
                # count is obviously 0
                return 0

        obj.add_count_column()
        number = obj.get_aggregation(using=using)[None]

        # Apply offset and limit constraints manually, since using LIMIT/OFFSET
        # in SQL (in variants that provide them) doesn't change the COUNT
        # output.
        number = max(0, number - self.low_mark)
        if self.high_mark is not None:
            number = min(number, self.high_mark - self.low_mark)

        return number

    def clear_limits(self):
        """
        Clears any existing limits.
        """
        self.low_mark, self.high_mark = 0, None
    
    def add_count_column(self):
        """
        Converts the query to do count(...) or count(distinct(pk)) in order to
        get its size.
        """
        if not self.distinct:
            if not self.select:
                count = self.aggregates_module.Count('*', is_summary=True)
            else:
                assert len(self.select) == 1, \
                        "Cannot add count col with multiple cols in 'select': %r" % self.select
                count = self.aggregates_module.Count(self.select[0])
        else:
            opts = self.model._meta
            if not self.select:
                count = self.aggregates_module.Count((self.join((None, opts.db_table, None, None)), opts.pk.column),
                                         is_summary=True, distinct=True)
            else:
                # Because of SQL portability issues, multi-column, distinct
                # counts need a sub-query -- see get_count() for details.
                assert len(self.select) == 1, \
                        "Cannot add count col with multiple cols in 'select'."

                count = self.aggregates_module.Count(self.select[0], distinct=True)
            # Distinct handling is done in Count(), so don't do it at this
            # level.
            self.distinct = False

        # Set only aggregate to be the count column.
        # Clear out the select cache to reflect the new unmasked aggregates.
        self.aggregates = {None: count}
        self.set_aggregate_mask(None)
        self.group_by = None
    
    def _aggregate_select(self):
        """The SortedDict of aggregate columns that are not masked, and should
        be used in the SELECT clause.

        This result is cached for optimization purposes.
        """
        if self._aggregate_select_cache is not None:
            return self._aggregate_select_cache
        elif self.aggregate_select_mask is not None:
            self._aggregate_select_cache = SortedDict([
                (k,v) for k,v in self.aggregates.items()
                if k in self.aggregate_select_mask
            ])
            return self._aggregate_select_cache
        else:
            return self.aggregates
    aggregate_select = property(_aggregate_select)
    
    def _extra_select(self):
        if self._extra_select_cache is not None:
            return self._extra_select_cache
        elif self.extra_select_mask is not None:
            self._extra_select_cache = SortedDict([
                (k,v) for k,v in self.extra.items()
                if k in self.extra_select_mask
            ])
            return self._extra_select_cache
        else:
            return self.extra
    extra_select = property(_extra_select)
    
    def get_aggregation(self, using):
        """
        Returns the dictionary with the values of the existing aggregations.
        """
        if not self.aggregate_select:
            return {}

        # If there is a group by clause, aggregating does not add useful
        # information but retrieves only the first row. Aggregate
        # over the subquery instead.
        if self.group_by is not None:
            from django.db.models.sql.subqueries import AggregateQuery
            query = AggregateQuery(self.model)

            obj = self.clone()

            # Remove any aggregates marked for reduction from the subquery
            # and move them to the outer AggregateQuery.
            for alias, aggregate in self.aggregate_select.items():
                if aggregate.is_summary:
                    query.aggregate_select[alias] = aggregate
                    del obj.aggregate_select[alias]

            try:
                query.add_subquery(obj, using)
            except EmptyResultSet:
                return dict(
                    (alias, None)
                    for alias in query.aggregate_select
                )
        else:
            query = self
            self.select = []
            self.default_cols = False
            self.extra = {}
            self.remove_inherited_models()

        query.clear_ordering(True)
        query.clear_limits()
        query.select_for_update = False
        query.select_related = False
        query.related_select_cols = []
        query.related_select_fields = []

        result = query.get_compiler(using).execute_sql(SINGLE)
        if result is None:
            result = [None for q in query.aggregate_select.items()]

        return dict([
            (alias, self.resolve_aggregate(val, aggregate, connection=connections[using]))
            for (alias, aggregate), val
            in zip(query.aggregate_select.items(), result)
        ])
    
    def resolve_aggregate(self, value, aggregate, connection):
        """Resolve the value of aggregates returned by the database to
        consistent (and reasonable) types.

        This is required because of the predisposition of certain backends
        to return Decimal and long types when they are not needed.
        
        TODO: the aggregate itself could do the resolving...
        """
        if value is None:
            if aggregate.is_ordinal:
                return 0
            # Return None as-is
            return value
        elif aggregate.is_ordinal:
            # Any ordinal aggregate (e.g., count) returns an int
            return int(value)
        elif aggregate.is_computed:
            # Any computed aggregate (e.g., avg) returns a float
            return float(value)
        else:
            # Return value depends on the type of the field being processed.
            return self.convert_values(value, aggregate.field, connection)
    
    def count_active_tables(self):
        return len(self.aliases)
    
    def add_extra(self, select, select_params, where, params, tables, order_by):
        """
        Adds data to the various extra_* attributes for user-created additions
        to the query.
        """
        if select:
            # We need to pair any placeholder markers in the 'select'
            # dictionary with their parameters in 'select_params' so that
            # subsequent updates to the select dictionary also adjust the
            # parameters appropriately.
            select_pairs = SortedDict()
            if select_params:
                param_iter = iter(select_params)
            else:
                param_iter = iter([])
            for name, entry in select.items():
                entry = force_unicode(entry)
                entry_params = []
                pos = entry.find("%s")
                while pos != -1:
                    entry_params.append(param_iter.next())
                    pos = entry.find("%s", pos + 2)
                select_pairs[name] = (entry, entry_params)
            # This is order preserving, since self.extra_select is a SortedDict.
            self.extra.update(select_pairs)
        if where or params:
            self.reverse_polish.append((ExtraWhere(where, params), AND))
            if len(self.reverse_polish) > 1:
                self.reverse_polish.append(AND)
        if tables:
            self.extra_tables += tuple(tables)
        if order_by:
            self.extra_order_by = order_by

class SimpleExpr(object):
    """
    This is here just to allow "tbl1"."col1" in where conditions, see
    extract_subquery. There must be a better way of doing this...
    """
    def __init__(self, alias, col):
	self.alias = alias
	self.col = col
	
    def as_sql(self, qn, connection):
	return "%s.%s" % (qn(self.alias), qn(self.col)), []

    def relabel_aliases(self, change_map):
	if self.alias in change_map:
	    self.alias = change_map[self.alias]
    def prepare(self):
        return self

def get_order_dir(field, default='ASC'):
    """
    Returns the field name and direction for an order specification. For
    example, '-foo' is returned as ('foo', 'DESC').

    The 'default' param is used to indicate which way no prefix (or a '+'
    prefix) should sort. The '-' prefix always sorts the opposite way.
    """
    dirn = ORDER_DIR[default]
    if field[0] == '-':
        return field[1:], dirn[1]
    return field, dirn[0]
