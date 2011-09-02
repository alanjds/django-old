from django.db.models.sql.constants import *
from django.db.models.sql.datastructures import EmptyResultSet

class QueryTreeSQLCompiler(object):
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
        # ordering, ordering_group_by = self.get_ordering()
        ordering, ordering_group_by = [], []

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
            "%s %s" % (qn(base_rel.model._meta.db_table), qn2(base_rel.alias))
        )
        for join in base_rel.child_joins:
             self.join_sql(join, result, qn, qn2)
        return result, []

