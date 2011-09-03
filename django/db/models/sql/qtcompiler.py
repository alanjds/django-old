from django.db.models.sql.constants import *
from django.db.models.sql.datastructures import EmptyResultSet
from django.db.models.sql.querytree import QueryTree, get_order_dir

def empty_iter():
    """
    Returns an iterator containing no results.
    """
    yield iter([]).next()

class QueryTreeSQLCompiler(object):
    def pre_sql_setup(self):
        """
        Does any necessary class setup immediately prior to producing SQL. This
        is for things that can't necessarily be done in __init__ because we
        might not have all the pieces in place at that time.
        """
        if (not self.query.select and self.query.default_cols and not
                self.query.included_inherited_models):
            self.query.setup_inherited_models()
        if self.query.select_related and not self.query.related_select_cols:
            self.fill_related_selections()

    def __init__(self, query, connection, using):
        self.query = query
        self.connection = connection
        self.using = using
        self.quote_cache = {}

    def quote_name_unless_alias(self, name):
        """
        A wrapper around connection.ops.quote_name that doesn't quote aliases
        for table names. This avoids problems with some SQL dialects that treat
        quoted strings specially (e.g. PostgreSQL).
        """
        if name in self.quote_cache:
            return self.quote_cache[name]
        if name in self.query.aliases or name in self.query.extra_select:
            self.quote_cache[name] = name
            return name
        r = self.connection.ops.quote_name(name)
        self.quote_cache[name] = r
        return r

    def results_iter(self):
        """
        Returns an iterator over the results from executing this query.
        """
        resolve_columns = hasattr(self, 'resolve_columns')
        fields = None
        has_aggregate_select = bool(self.query.aggregates)
        # Set transaction dirty if we're using SELECT FOR UPDATE to ensure
        # a subsequent commit/rollback is executed, so any database locks
        # are released.
        if self.query.select_for_update and transaction.is_managed(self.using):
            transaction.set_dirty(self.using)
        for rows in self.execute_sql(MULTI):
            for row in rows:
                if resolve_columns:
                    if fields is None:
                        # We only set this up here because
                        # related_select_fields isn't populated until
                        # execute_sql() has been called.
                        if self.query.select_fields:
                            fields = self.query.select_fields + self.query.related_select_fields
                        else:
                            fields = self.query.model._meta.fields
                        # If the field was deferred, exclude it from being passed
                        # into `resolve_columns` because it wasn't selected.
                        only_load = self.deferred_to_columns()
                        if only_load:
                            db_table = self.query.model._meta.db_table
                            fields = [f for f in fields if db_table in only_load and
                                      f.column in only_load[db_table]]
                    row = self.resolve_columns(row, fields)

                if has_aggregate_select:
                    aggregate_start = len(self.query.extra_select.keys()) + len(self.query.select)
                    aggregate_end = aggregate_start + len(self.query.aggregate_select)
                    row = tuple(row[:aggregate_start]) + tuple([
                        self.query.resolve_aggregate(value, aggregate, self.connection)
                        for (alias, aggregate), value
                        in zip(self.query.aggregate_select.items(), row[aggregate_start:aggregate_end])
                    ]) + tuple(row[aggregate_end:])

                yield row
    
    def execute_sql(self, result_type=MULTI):
        """
        Run the query against the database and returns the result(s). The
        return value is a single data item if result_type is SINGLE, or an
        iterator over the results if the result_type is MULTI.

        result_type is either MULTI (use fetchmany() to retrieve all rows),
        SINGLE (only retrieve a single row), or None. In this last case, the
        cursor is returned if any query is executed, since it's used by
        subclasses such as InsertQuery). It's possible, however, that no query
        is needed, as the filters describe an empty set. In that case, None is
        returned, to avoid any unnecessary database interaction.
        """
        try:
            sql, params = self.as_sql()
            if not sql:
                raise EmptyResultSet
        except EmptyResultSet:
            if result_type == MULTI:
                return empty_iter()
            else:
                return

        cursor = self.connection.cursor()
        cursor.execute(sql, params)

        if not result_type:
            return cursor
        if result_type == SINGLE:
            if self.query.ordering_aliases:
                return cursor.fetchone()[:-len(self.query.ordering_aliases)]
            return cursor.fetchone()

        # The MULTI case.
        if False and self.query.ordering_aliases:
            result = order_modified_iter(cursor, len(self.query.ordering_aliases),
                    self.connection.features.empty_fetchmany_value)
        else:
            result = iter((lambda: cursor.fetchmany(GET_ITERATOR_CHUNK_SIZE)),
                    self.connection.features.empty_fetchmany_value)
        if not self.connection.features.can_use_chunked_reads:
            # If we are using non-chunked reads, we return the same data
            # structure as normally, but ensure it is all read into memory
            # before going any further.
            return list(result)
        return result
    
    def as_sql(self, with_limits=True, with_col_aliases=False):
        """
        Creates the SQL for this query. Returns the SQL string and list of
        parameters.

        If 'with_limits' is False, any limit/offset information is not included
        in the query.
        """
        if with_limits and self.query.low_mark == self.query.high_mark:
            return '', ()

        out_cols = self.get_columns(with_col_aliases)
        ordering, ordering_group_by = self.get_ordering()

        # This must come after 'select' and 'ordering' -- see docstring of
        # get_from_clause() for details.
        from_, f_params = self.get_from_clause()

        qn = self.quote_name_unless_alias

        where, w_params = self.query.where.as_sql(qn=qn, connection=self.connection)
        having, h_params = self.query.having.as_sql(qn=qn, connection=self.connection)
        params = []
        for val in self.query.extra_select.itervalues():
            params.extend(val[1])

        result = ['SELECT']
        if self.query.distinct:
            result.append('DISTINCT')
        result.append(', '.join(out_cols)) # + self.query.ordering_aliases))

        result.append('FROM')
        result.extend(from_)
        params.extend(f_params)

        if where:
            result.append('WHERE %s' % where)
            params.extend(w_params)

        # grouping, gb_params = self.get_grouping()
        grouping = False
        if grouping:
            if ordering:
                # If the backend can't group by PK (i.e., any database
                # other than MySQL), then any fields mentioned in the
                # ordering clause needs to be in the group by clause.
                if not self.connection.features.allows_group_by_pk:
                    for col, col_params in ordering_group_by:
                        if col not in grouping:
                            grouping.append(str(col))
                            gb_params.extend(col_params)
            else:
                ordering = self.connection.ops.force_no_ordering()
            result.append('GROUP BY %s' % ', '.join(grouping))
            params.extend(gb_params)

        if having:
            result.append('HAVING %s' % having)
            params.extend(h_params)

        if ordering:
            result.append('ORDER BY %s' % ', '.join(ordering))

        if with_limits:
            if self.query.high_mark is not None:
                result.append('LIMIT %d' % (self.query.high_mark - self.query.low_mark))
            if self.query.low_mark:
                if self.query.high_mark is None:
                    val = self.connection.ops.no_limit_value()
                    if val:
                        result.append('LIMIT %d' % val)
                result.append('OFFSET %d' % self.query.low_mark)

        if self.query.select_for_update and self.connection.features.has_select_for_update:
            # If we've been asked for a NOWAIT query but the backend does not support it,
            # raise a DatabaseError otherwise we could get an unexpected deadlock.
            nowait = self.query.select_for_update_nowait
            if nowait and not self.connection.features.has_select_for_update_nowait:
                raise DatabaseError('NOWAIT is not supported on this database backend.')
            result.append(self.connection.ops.for_update_sql(nowait=nowait))
        return ' '.join(result), tuple(params)
    
    def get_ordering(self):
        """
        Returns a tuple containing a list representing the SQL elements in the
        "order by" clause, and the list of SQL elements that need to be added
        to the GROUP BY clause as a result of the ordering.

        Also sets the ordering_aliases attribute on this instance to a list of
        extra aliases needed in the select.

        Determining the ordering SQL can change the tables we need to include,
        so this should be run *before* get_from_clause().
        """
        if self.query.extra_order_by:
            ordering = self.query.extra_order_by
        elif not self.query.default_ordering:
            ordering = self.query.order_by
        else:
            ordering = self.query.order_by or self.query.model._meta.ordering
        qn = self.quote_name_unless_alias
        qn2 = self.connection.ops.quote_name
        distinct = self.query.distinct
        select_aliases = self._select_aliases
        result = []
        group_by = []
        ordering_aliases = []
        if self.query.standard_ordering:
            asc, desc = ORDER_DIR['ASC']
        else:
            asc, desc = ORDER_DIR['DESC']

        # It's possible, due to model inheritance, that normal usage might try
        # to include the same field more than once in the ordering. We track
        # the table/column pairs we use and discard any after the first use.
        processed_pairs = set()

        for field in ordering:
            if field == '?':
                result.append(self.connection.ops.random_function_sql())
                continue
            if isinstance(field, int):
                if field < 0:
                    order = desc
                    field = -field
                else:
                    order = asc
                result.append('%s %s' % (field, order))
                group_by.append((field, []))
                continue
            col, order = get_order_dir(field, asc)
            if col in self.query.aggregate_select:
                result.append('%s %s' % (qn(col), order))
                continue
            if '.' in field:
                # This came in through an extra(order_by=...) addition. Pass it
                # on verbatim.
                table, col = col.split('.', 1)
                if (table, col) not in processed_pairs:
                    elt = '%s.%s' % (qn(table), col)
                    processed_pairs.add((table, col))
                    if not distinct or elt in select_aliases:
                        result.append('%s %s' % (elt, order))
                        group_by.append((elt, []))
            elif get_order_dir(field)[0] not in self.query.extra_select:
                # 'col' is of the form 'field' or 'field1__field2' or
                # '-field1__field2__field', etc.
                for table, col, order in self.find_ordering_name(field,
                        self.query.model._meta, default_order=asc):
                    if (table, col) not in processed_pairs:
                        elt = '%s.%s' % (qn(table), qn2(col))
                        processed_pairs.add((table, col))
                        if distinct and elt not in select_aliases:
                            ordering_aliases.append(elt)
                        result.append('%s %s' % (elt, order))
                        group_by.append((elt, []))
            else:
                elt = qn2(col)
                if distinct and col not in select_aliases:
                    ordering_aliases.append(elt)
                result.append('%s %s' % (elt, order))
                group_by.append(self.query.extra_select[col])
        self.query.ordering_aliases = ordering_aliases
        return result, group_by

    def find_ordering_name(self, name, opts, alias=None, default_order='ASC',
            already_seen=None):
        """
        Returns the table alias (the name might be ambiguous, the alias will
        not be) and column name for ordering by the given 'name' parameter.
        The 'name' is of the form 'field1__field2__...__fieldN'.
        """
        name, order = get_order_dir(name, default_order)
        pieces = name.split(LOOKUP_SEP)
        if not alias:
            alias = self.query.get_initial_alias()
        import ipdb; ipdb.set_trace()
        field, target, opts, joins, last, extra = self.query.setup_joins(pieces,
                opts, alias, False)
        alias = joins[-1]
        col = field.column
        #if not field.rel:
            # To avoid inadvertent trimming of a necessary alias, use the
            # refcount to show that we are referencing a non-relation field on
            # the model.
            #self.query.ref_alias(alias)

        # Must use left outer joins for nullable fields and their relations.
        #self.query.promote_alias_chain(joins,
        #    self.query.alias_map[joins[0]][JOIN_TYPE] == self.query.LOUTER)

        # If we get to this point and the field is a relation to another model,
        # append the default ordering for that model.
        if field.rel and len(joins) > 1 and field.rel.to._meta.ordering:
            # Firstly, avoid infinite loops.
            if not already_seen:
                already_seen = set()
            join_tuple = tuple([self.query.alias_map[j][TABLE_NAME] for j in joins])
            if join_tuple in already_seen:
                raise FieldError('Infinite loop caused by ordering.')
            already_seen.add(join_tuple)

            results = []
            for item in opts.ordering:
                results.extend(self.find_ordering_name(item, opts, alias,
                        order, already_seen))
            return results

        #if alias:
            # We have to do the same "final join" optimisation as in
            # add_filter, since the final column might not otherwise be part of
            # the select set (so we can't order on it).
            # while 1:
            #    join = self.query.alias_map[alias]
            #    if col != join[RHS_JOIN_COL]:
            #        break
            #    self.query.unref_alias(alias)
            #    alias = join[LHS_ALIAS]
            #    col = join[LHS_JOIN_COL]
        return [(alias, col, order)]


    def get_columns(self, with_aliases=False):
        """
        Returns the list of columns to use in the select statement. If no
        columns have been specified, returns all columns relating to fields in
        the model.

        If 'with_aliases' is true, any column names that are duplicated
        (without the table names) are given unique aliases. This is needed in
        some cases to avoid ambiguity with nested queries.
        """
        qn = self.quote_name_unless_alias
        qn2 = self.connection.ops.quote_name
        result = ['(%s) AS %s' % (col[0], qn2(alias)) for alias, col in self.query.extra_select.iteritems()]
        aliases = set(self.query.extra_select.keys())
        if with_aliases:
            col_aliases = aliases.copy()
        else:
            col_aliases = set()
        if self.query.select_cols:
            only_load = self.deferred_to_columns()
            for col in self.query.select_cols:
                if isinstance(col, (list, tuple)):
                    alias, column = col
                    r = '%s.%s' % (qn(alias), qn(column))
                    if with_aliases:
                        if col[1] in col_aliases:
                            c_alias = 'Col%d' % len(col_aliases)
                            result.append('%s AS %s' % (r, c_alias))
                            aliases.add(c_alias)
                            col_aliases.add(c_alias)
                        else:
                            result.append('%s AS %s' % (r, qn2(col[1])))
                            aliases.add(r)
                            col_aliases.add(col[1])
                    else:
                        result.append(r)
                        aliases.add(r)
                        col_aliases.add(col[1])
                else:
                    result.append(col.as_sql(qn, self.connection))

                    if hasattr(col, 'alias'):
                        aliases.add(col.alias)
                        col_aliases.add(col.alias)

        max_name_length = self.connection.ops.max_name_length()
        result.extend([
            '%s%s' % (
                aggregate.as_sql(qn, self.connection),
                alias is not None
                    and ' AS %s' % qn(truncate_name(alias, max_name_length))
                    or ''
            )
            for alias, aggregate in self.query.aggregate_select.items()
        ])
        """
        for table, col in self.query.related_select_cols:
            r = '%s.%s' % (qn(table), qn(col))
            if with_aliases and col in col_aliases:
                c_alias = 'Col%d' % len(col_aliases)
                result.append('%s AS %s' % (r, c_alias))
                aliases.add(c_alias)
                col_aliases.add(c_alias)
            else:
                result.append(r)
                aliases.add(r)
                col_aliases.add(col)
        """
        self._select_aliases = aliases
        if not result:
            return ['1']
        return result

    def join_sql(self, join, buf, qn, qn2):
        join_cond = []
        for from_field, to_field in join.join_cols:
            join_cond.append("%s.%s = %s.%s" % (
                qn2(join.from_rel.alias), qn(from_field.column),
                qn2(join.to_rel.alias), qn(to_field.column)
            ))
        # TODO: handle extra join conditions
        join_cond = ' AND '.join(join_cond)
        buf.append("JOIN %s %s ON %s" % (
            qn(join.to_rel.model._meta.db_table), 
            qn2(join.to_rel.alias), join_cond
        ))
        for join in join.to_rel.child_joins:
            self.join_sql(join, buf, qn, qn2)
    
    def deferred_to_columns(self):
        """
        Converts the self.deferred_loading data structure to mapping of table
        names to sets of column names which are to be loaded. Returns the
        dictionary.
        """
        columns = {}
        self.query.deferred_to_data(columns, self.query.deferred_to_columns_cb)
        return columns

    def get_from_clause(self):
        """
        Returns a list of strings that are joined together to go after the
        "FROM" part of the query, as well as a list any extra parameters that
        need to be included. Sub-classes, can override this to create a
        from-clause via a "select".
        """
        result = []
        qn = self.quote_name_unless_alias
        qn2 = self.connection.ops.quote_name
        base_rel = self.query.base_rel
        result.append(
            "%s %s" % (qn(base_rel.model._meta.db_table), qn(base_rel.alias))
        )
        for join in base_rel.child_joins:
             self.join_sql(join, result, qn, qn2)
        return result, []

class QTUpdateCompiler(QueryTreeSQLCompiler):
    def as_sql(self):
        """
        Creates the SQL for this query. Returns the SQL string and list of
        parameters.
        """
        from django.db.models.base import Model

        self.pre_sql_setup()
        if not self.query.values:
            return '', ()
        table = self.query.get_meta().db_table
        qn = self.quote_name_unless_alias
        self.query.where.relabel_aliases({self.query.base_rel.alias: table})
        result = ['UPDATE %s' % qn(table) ]
        result.append('SET')
        values, update_params = [], []
        for field, model, val in self.query.values:
            if hasattr(val, 'prepare_database_save'):
                val = val.prepare_database_save(field)
            else:
                val = field.get_db_prep_save(val, connection=self.connection)

            # Getting the placeholder for the field.
            if hasattr(field, 'get_placeholder'):
                placeholder = field.get_placeholder(val, self.connection)
            else:
                placeholder = '%s'

            if hasattr(val, 'evaluate'):
                val = SQLEvaluator(val, self.query, allow_joins=False)
            name = field.column
            if hasattr(val, 'as_sql'):
                sql, params = val.as_sql(qn, self.connection)
                values.append('%s = %s' % (qn(name), sql))
                update_params.extend(params)
            elif val is not None:
                values.append('%s = %s' % (qn(name), placeholder))
                update_params.append(val)
            else:
                values.append('%s = NULL' % qn(name))
        if not values:
            return '', ()
        result.append(', '.join(values))
        where, params = self.query.where.as_sql(qn=qn, connection=self.connection)
        if where:
            result.append('WHERE %s' % where)
        return ' '.join(result), tuple(update_params + params)

    def execute_sql(self, result_type):
        """
        Execute the specified update. Returns the number of rows affected by
        the primary update query. The "primary update query" is the first
        non-empty query that is executed. Row counts for any subsequent,
        related queries are not available.
        """
        cursor = super(QTUpdateCompiler, self).execute_sql(result_type)
        rows = cursor and cursor.rowcount or 0
        is_empty = cursor is None
        del cursor
        for query in self.query.get_related_updates():
            aux_rows = query.get_compiler(self.using).execute_sql(result_type)
            if is_empty:
                rows = aux_rows
                is_empty = False
        return rows

    def pre_sql_setup(self):
        """
        If the update depends on results from other tables, we need to do some
        munging of the "where" conditions to match the format required for
        (portable) SQL updates. That is done here.

        Further, if we are going to be running multiple updates, we pull out
        the id values to update at this point so that they don't change as a
        result of the progressive updates.
        """
        self.query.select_related = False
        self.query.clear_ordering(True)
        super(QTUpdateCompiler, self).pre_sql_setup()
        count = self.query.count_active_tables()
        if not self.query.related_updates and count == 1:
            return

        # We need to use a sub-select in the where clause to filter on things
        # from other tables.
        query = self.query.clone()
        query.__class__ = QueryTree
        query.bump_prefix()
        query.extra = {}
        query.select = []
        query.add_fields([query.model._meta.pk.name])
        must_pre_select = count > 1 and not self.connection.features.update_can_self_select

        # Now we adjust the current query: reset the where clause and get rid
        # of all the tables we don't need (since they're in the sub-select).
        self.query.where = self.query.where_class()
        if self.query.related_updates or must_pre_select:
            # Either we're using the idents in multiple update queries (so
            # don't want them to change), or the db backend doesn't support
            # selecting from the updating table (e.g. MySQL).
            idents = []
            for rows in query.get_compiler(self.using).execute_sql(MULTI):
                idents.extend([r[0] for r in rows])
            self.query.add_filter(('pk__in', idents))
            self.query.related_ids = idents
        else:
            # The fast path. Filters and updates in one query.
            self.query.add_filter(('pk__in', query))
        for alias in self.query.tables[1:]:
            self.query.alias_refcount[alias] = 0

class QTDateCompiler(QueryTreeSQLCompiler):
    def results_iter(self):
        """
        Returns an iterator over the results from executing this query.
        """
        resolve_columns = hasattr(self, 'resolve_columns')
        if resolve_columns:
            from django.db.models.fields import DateTimeField
            fields = [DateTimeField()]
        else:
            from django.db.backends.util import typecast_timestamp
            needs_string_cast = self.connection.features.needs_datetime_string_cast

        offset = len(self.query.extra_select)
        for rows in self.execute_sql(MULTI):
            for row in rows:
                date = row[offset]
                if resolve_columns:
                    date = self.resolve_columns(row, fields)[offset]
                elif needs_string_cast:
                    date = typecast_timestamp(str(date))
                yield date
