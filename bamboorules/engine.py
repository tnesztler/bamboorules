from collections.abc import Mapping
from functools import reduce

import pandas as pd

SEQUENCE = (tuple, list, range)


class Engine:
    _custom_operations = {}

    @property
    def operations(self):
        """Gather all operations."""
        operations = {}
        operations.update(self._logical_operations)
        operations.update(self._scoped_operations)
        operations.update(self._data_operations)
        operations.update(self._common_operations)
        operations.update(self._unsupported_operations)
        operations.update(self._custom_operations)
        return operations

    @staticmethod
    def _is_dictionary(arg):
        """Check if argument is a Python Mapping."""
        return isinstance(arg, Mapping)

    @staticmethod
    def _is_sequence(arg):
        """Check if argument is sequence (tuple, list, range)."""
        return isinstance(arg, SEQUENCE)

    @staticmethod
    def _is_dataframe(arg):
        """Check if argument is a Pandas DataFrame."""
        return isinstance(arg, pd.DataFrame)

    @staticmethod
    def _is_series(arg):
        """Check if argument is a Pandas Series."""
        return isinstance(arg, pd.Series)

    @staticmethod
    def _is_series_or_scalar(arg):
        """Check if argument is a Pandas Series or scalar."""
        return isinstance(arg, pd.Series) or pd.api.types.is_scalar(arg)

    @staticmethod
    def _is_dataframe_or_series(arg):
        """Check if argument is a Pandas DataFrame or Series."""
        return isinstance(arg, (pd.DataFrame, pd.Series))

    @staticmethod
    def _is_dataframe_series_or_sequence(arg):
        """Check if argument is a Pandas DataFrame, Series or Python sequence."""
        return isinstance(arg, (pd.DataFrame, pd.Series, *SEQUENCE))

    @staticmethod
    def _is_dataframe_series_sequence_or_scalar(arg):
        """
        Check if argument is a Pandas DataFrame, Series, scalar
        or Python sequence.
        """
        return isinstance(
            arg, (pd.DataFrame, pd.Series, *SEQUENCE)
        ) or pd.api.types.is_scalar(arg)

    # Common Operations
    def _equal_to(self, a, b):
        """Check for non-strict equality ('==') with JS-style type coercion."""
        if (self._is_dataframe(a) and self._is_dataframe_series_or_sequence(b)) or (
            self._is_series(a) and self._is_series_or_scalar(b)
        ):
            return a.eq(b)
        elif (self._is_dataframe(b) and self._is_dataframe_series_or_sequence(a)) or (
            self._is_series(b) and self._is_series_or_scalar(a)
        ):
            return b.eq(a)
        else:
            return a == b

    def _strict_equal_to(self, a, b):
        """Check for strict equality ('===') including type equality."""
        if type(a) is type(b):
            return self._equal_to(a, b)
        return False

    def _not_equal_to(self, a, b):
        """Check for non-strict inequality ('==') with JS-style type coercion."""
        return not self._equal_to(a, b)

    def _not_strict_equal_to(self, a, b):
        """Check for strict inequality ('!==') including type inequality."""
        return not self._strict_equal_to(a, b)

    def _less_than(self, a, b):
        """Check that A is less then B (A < B)."""
        if (self._is_dataframe(a) and self._is_dataframe_series_or_sequence(b)) or (
            self._is_series(a) and self._is_series_or_scalar(b)
        ):
            return a.lt(b)
        elif (self._is_dataframe(b) and self._is_dataframe_series_or_sequence(a)) or (
            self._is_series(b) and self._is_series_or_scalar(a)
        ):
            return b.gt(a)
        else:
            return a < b

    def _less_than_or_equal_to(self, a, b):
        """Check that A is less then or equal to B (A <= B)."""
        if (self._is_dataframe(a) and self._is_dataframe_series_or_sequence(b)) or (
            self._is_series(a) and self._is_series_or_scalar(b)
        ):
            return a.le(b)
        elif (self._is_dataframe(b) and self._is_dataframe_series_or_sequence(a)) or (
            self._is_series(b) and self._is_series_or_scalar(a)
        ):
            return b.ge(a)
        else:
            return a <= b

    def _greater_than(self, a, b):
        """Check that A is greater then B (A > B)."""
        return self._less_than(b, a)

    def _greater_than_or_equal_to(self, a, b):
        """Check that A is greater then or equal to B (A >= B)."""
        return self._less_than_or_equal_to(b, a)

    @staticmethod
    def _truthy(a):
        """Check that argument evaluates to True according to core JsonLogic."""
        return bool(a)

    def _falsy(self, a):
        """Check that argument evaluates to False according to core JsonLogic."""
        return not self._truthy(a)

    def _add(self, a, b):
        """Add B to A."""
        if (
            self._is_dataframe(a) and self._is_dataframe_series_sequence_or_scalar(b)
        ) or (self._is_series(a) and self._is_series_or_scalar(b)):
            return a.add(b)
        elif (
            self._is_dataframe(b) and self._is_dataframe_series_sequence_or_scalar(a)
        ) or (self._is_series(b) and self._is_series_or_scalar(a)):
            return b.radd(a)
        else:
            return a + b

    def _sub(self, a, b=None):
        """Subtract B from A. If only A is provided - return its arithmetic negative."""
        if b is None:
            return -a
        else:
            if (
                self._is_dataframe(a)
                and self._is_dataframe_series_sequence_or_scalar(b)
            ) or (self._is_series(a) and self._is_series_or_scalar(b)):
                return a.sub(b)
            elif (
                self._is_dataframe(b)
                and self._is_dataframe_series_sequence_or_scalar(a)
            ) or (self._is_series(b) and self._is_series_or_scalar(a)):
                return b.rsub(a)
            else:
                return a - b

    def _mul(self, a, b):
        """Multiply A by B."""
        if (
            self._is_dataframe(a) and self._is_dataframe_series_sequence_or_scalar(b)
        ) or (self._is_series(a) and self._is_series_or_scalar(b)):
            return a.mul(b)
        elif (
            self._is_dataframe(b) and self._is_dataframe_series_sequence_or_scalar(a)
        ) or (self._is_series(b) and self._is_series_or_scalar(a)):
            return b.rmul(a)
        else:
            return a * b

    def _truediv(self, a, b):
        """Divide A by B (float division)."""
        if (
            self._is_dataframe(a) and self._is_dataframe_series_sequence_or_scalar(b)
        ) or (self._is_series(a) and self._is_series_or_scalar(b)):
            return a.truediv(b)
        elif (
            self._is_dataframe(b) and self._is_dataframe_series_sequence_or_scalar(a)
        ) or (self._is_series(b) and self._is_series_or_scalar(a)):
            return b.rtruediv(a)
        else:
            return a / b

    def _floordiv(self, a, b):
        """Divide A by B (integer division)."""
        if (
            self._is_dataframe(a) and self._is_dataframe_series_sequence_or_scalar(b)
        ) or (self._is_series(a) and self._is_series_or_scalar(b)):
            return a.floordiv(b)
        elif (
            self._is_dataframe(b) and self._is_dataframe_series_sequence_or_scalar(a)
        ) or (self._is_series(b) and self._is_series_or_scalar(a)):
            return b.rfloordiv(a)
        else:
            return a // b

    def _mod(self, a, b):
        """Modulo of A by B."""
        if (
            self._is_dataframe(a) and self._is_dataframe_series_sequence_or_scalar(b)
        ) or (self._is_series(a) and self._is_series_or_scalar(b)):
            return a.mod(b)
        elif (
            self._is_dataframe(b) and self._is_dataframe_series_sequence_or_scalar(a)
        ) or (self._is_series(b) and self._is_series_or_scalar(a)):
            return b.rmod(a)
        else:
            return a % b

    def _pow(self, a, b):
        """A to the power B."""
        if (
            self._is_dataframe(a) and self._is_dataframe_series_sequence_or_scalar(b)
        ) or (self._is_series(a) and self._is_series_or_scalar(b)):
            return a.pow(b)
        elif (
            self._is_dataframe(b) and self._is_dataframe_series_sequence_or_scalar(a)
        ) or (self._is_series(b) and self._is_series_or_scalar(a)):
            return b.rpow(a)
        else:
            return a ** b

    def _abs(self, a):
        """Absolute value of A."""
        if self._is_dataframe_or_series(a):
            return a.abs()
        else:
            return abs(a)

    def _min(self, *args):
        """Minimal value of sequence or unique element."""
        if self._is_sequence(args):
            return min(args)
        else:
            return min((args,))

    def _min_reduce(self, a):
        """Returns the min along each axis"""
        if self._is_dataframe_or_series(a):
            return a.min()

    def _max(self, *args):
        """Maximal value of sequence or unique element."""
        if self._is_sequence(args):
            return max(args)
        else:
            return max((args,))

    def _max_reduce(self, a):
        """Returns the max along each axis."""
        if self._is_dataframe_or_series(a):
            return a.max()

    @staticmethod
    def _method(obj, method, args=[]):
        """
        Call the specified 'method' on an 'obj' object with an array of 'args'
        arguments.

        Example:
        {"method": [{"var": "today"}, "isoformat"]}
        gets datetime.date object from 'today' variable and calls its 'isoformat'
        method returning an ISO-formated date string.
        {"method": ["string value", "split", [" "]]}
        calls split(' ') method on a "string value" string returning an array
        of ["string", "format"].

        Can also be used to get property values instead of calling a method.
        In this case arguments are ignored.

        Example:
        {"method": [{"var": "today"}, "month"]}
        gets datetime.date object from 'today' variable and returns the number of
        the month via 'month' property.
        """
        method = getattr(obj, method)
        if callable(method):
            return method(*args)
        return method

    @property
    def _common_operations(self):
        return {
            "==": self._equal_to,
            "===": self._strict_equal_to,
            "!=": self._not_equal_to,
            "!==": self._not_strict_equal_to,
            ">": self._greater_than,
            ">=": self._greater_than_or_equal_to,
            "<": self._less_than,
            "<=": self._less_than_or_equal_to,
            "!!": self._truthy,
            "!": self._falsy,
            "+": self._add,
            "-": self._sub,
            "*": self._mul,
            "/": self._truediv,
            "//": self._floordiv,
            "%": self._mod,
            "abs": self._abs,
            "min": self._min,
            "min_reduce": self._min_reduce,
            "max": self._max,
            "max_reduce": self._max_reduce,
            "method": self._method,
        }

    # Logical operations

    def _if(self, data, *args):
        """
        Evaluate chainable conditions with multiple 'else if' support and return
        the corresponding evaluated argument based on the following patterns:

        if 0 then 1 else None
        if 0 then 1 else 2
        if 0 then 1 else if 2 then 3 else 4
        if 0 then 1 else if 2 then 3 else if 4 then 5 else 6

        - If no arguments are given then return None.
        - If only one argument is given the evaluate and return it.
        - If two arguments are given then evaluate the first one, if it evaluates
        to True return evaluated second argument, otherwise return None.
        - If three arguments are given then evaluate the first one, if it evaluates
        to True return evaluated second argument, otherwise return evaluated
        third argument.
        - For more then 3 arguments:
            - If the first argument evaluates to True then evaluate and return
            the second argument.
            - If the first argument evaluates to False then jump to the next pair
            (e.g.: from 0,1 to 2,3) and evaluate them.
        """
        for i in range(0, len(args) - 1, 2):
            if self._truthy(self.execute(args[i], data)):
                return self.execute(args[i + 1], data)
        if len(args) % 2:
            return self.execute(args[-1], data)
        else:
            return None

    def _iif(self, data, a, b, c):
        """
        Evaluate ternary expression and return corresponding evaluated
        argument based on the following pattern: if (A) then {B} else {C}
        """
        return self._if(data, a, b, c)

    def _and(self, data, *args):
        """
        Evaluate and logically join arguments using the 'and' operator.
        If all arguments evaluate to True return the last (truthy) one (meaning
        that the whole expression evaluates to True).
        Otherwise return first countered falsy argument (meaning that the whole
        expression evaluates to False).
        """
        current = False
        for current in args:
            current = self.execute(current, data)
            if self._falsy(current):
                return current  # First falsy argument
        return current  # Last argument

    def _or(self, data, *args):
        """
        Evaluate and logically join arguments using the 'or' operator.
        If at least one argument evaluates to True - return it
        (meaning that the whole expression evaluates to True).
        Otherwise return the last (falsy) argument (meaning that the whole
        expression evaluates to False).
        """
        current = False
        for current in args:
            current = self.execute(current, data)
            if self._truthy(current):
                return current  # First truthy argument
        return current  # Last argument

    @property
    def _logical_operations(self):
        return {
            "if": self._if,
            "?:": self._iif,
            "and": self._and,
            "or": self._or,
        }

    # Scoped operations

    def _filter(self, data, scopedData, scopedLogic):
        """
        Filter 'scopedData' using the specified 'scopedLogic' argument.

        'scopedData' argument can be:
        - a manually specified data array;
        - a JsonLogic rule returning a data array;
        - a JsonLogic 'var' operation returning part of the data object
            containing a data array; like: {"var": "a"};
        - a JsonLogic 'var' operation returning the whole data object
            if it is an array itself; like: {"var": ""}.

        'scopedLogic' is a normal JsonLogic rule that uses a 'scopeData'
        element as its data object.

        'scopedLogic' must evaluate to a truthy value in order for the current
        'scopedData' element to be included into the resulting array, or to
        a falsy value to exclude it.

        Example:
        {"filter": [
            [1, 2, 3, 4, 5],
            {"%": [{"var": ""}, 2]}
        ]}
        calculates to: [1, 3, 5]

        If 'scopedData' argument does not evaluate to an array, an empty array
        is returned.
        """
        scopedData = self.execute(scopedData, data)
        if not self._is_sequence(scopedData):
            return []
        return list(
            filter(
                lambda datum: self._truthy(self.execute(scopedLogic, datum)),
                scopedData,
            )
        )

    def _map(self, data, scopedData, scopedLogic):
        """
        Apply 'scopedLogic' argument to each 'scopedData' element.

        'scopedData' argument can be:
        - a manually specified data array;
        - a JsonLogic rule returning a data array;
        - a JsonLogic 'var' operation returning part of the data object
            containing a data array; like: {"var": "a"};
        - a JsonLogic 'var' operation returning the whole data object
            if it is an array itself; like: {"var": ""}.

        'scopedLogic' is a normal JsonLogic rule that uses a 'scopeData'
        element as its data object.

        Result returned by 'scopedLogic' is included into the resulting array.

        Example:
        {"map": [
            [1, 2, 3, 4, 5],
            {"*": [{"var": ""}, 2]}
        ]}
        calculates to: [2, 4, 6, 8, 10]

        If 'scopedData' argument does not evaluate to an array, an empty array
        is returned.
        """
        scopedData = self.execute(scopedData, data)
        if not self._is_sequence(scopedData):
            return []
        return list(map(lambda datum: self.execute(scopedLogic, datum), scopedData))

    def _reduce(self, data, scopedData, scopedLogic, initial=None):
        """
        Apply 'scopedLogic' cumulatively to the elements in 'scopedData' argument,
        from left to right, so as to reduce the sequence it to a single value.
        If 'initial' is provided, it is placed before all 'scopedData' elements in
        the calculation, and serves as a default when 'scopedData' array is empty.

        'scopedData' argument can be:
        - a manually specified data array;
        - a JsonLogic rule returning a data array;
        - a JsonLogic 'var' operation returning part of the data object
            containing a data array; like: {"var": "a"};
        - a JsonLogic 'var' operation returning the whole data object
            if it is an array itself; like: {"var": ""}.

        'scopedLogic' is a normal JsonLogic rule that is applied to the following
        data object: {'accumulator': accumulator, 'current': current}; where
        'accumulator' is the result of all previous iterations (of 'initial' if
        none had occurred so far), and 'current' is the value of the current
        'scopedData' element being analyzed.

        The return value of the final application is returned as the result of
        the 'reduce' operation.

        Example:
        {"reduce": [
            [1, 2, 3, 4, 5],
            {"+": [{"var": "accumulator"}, {"var": "current"}]},
            0
        ]}
        calculates as: ((((1+2)+3)+4)+5) = 15

        If 'scopedData' argument does not evaluate to an array, the 'initial'
        value is returned.
        """
        scopedData = self.execute(scopedData, data)
        if not self._is_sequence(scopedData):
            return initial
        return reduce(
            lambda accumulator, current: self.execute(
                scopedLogic, {"accumulator": accumulator, "current": current}
            ),
            scopedData,
            initial,
        )

    def _all(self, data, scopedData, scopedLogic):
        """
        Check if 'scopedLogic' evaluates to a truthy value for all
        'scopedData' elements.

        'scopedData' argument can be:
        - a manually specified data array;
        - a JsonLogic rule returning a data array;
        - a JsonLogic 'var' operation returning part of the data object
            containing a data array; like: {"var": "a"};
        - a JsonLogic 'var' operation returning the whole data object
            if it is an array itself; like: {"var": ""}.

        'scopedLogic' is a normal JsonLogic rule that uses a 'scopeData'
        element as its data object.

        Return True if 'scopedLogic' evaluates to a truthy value for all
        'scopedData' elements. Return False otherwise.

        Example:
        {"all": [
            [1, 2, 3, 4, 5],
            {">=":[{"var":""}, 1]}
        ]}
        evaluates to: True

        If 'scopedData' argument does not evaluate to an array or if the array
        is empty, False is returned.

        N.B.: According to current core JsonLogic evaluation of 'scopedData'
        elements stops upon encountering first falsy value.
        """
        scopedData = self.execute(scopedData, data)
        if not self._is_sequence(scopedData):
            return False
        if len(scopedData) == 0:
            return False  # "all" of an empty set is false
        for datum in scopedData:
            if self._falsy(self.execute(scopedLogic, datum)):
                return False  # First falsy, short circuit
        return True  # All were truthy

    def _none(self, data, scopedData, scopedLogic):
        """
        Check if 'scopedLogic' evaluates to a truthy value for none of
        'scopedData' elements.

        'scopedData' argument can be:
        - a manually specified data array;
        - a JsonLogic rule returning a data array;
        - a JsonLogic 'var' operation returning part of the data object
            containing a data array; like: {"var": "a"};
        - a JsonLogic 'var' operation returning the whole data object
            if it is an array itself; like: {"var": ""}.

        'scopedLogic' is a normal JsonLogic rule that uses a 'scopeData'
        element as its data object.

        Return True if 'scopedLogic' evaluates to a falsy value for all
        'scopedData' elements. Return False otherwise.

        Example:
        {"none": [
            [1, 2, 3, 4, 5],
            {"==":[{"var":""}, 10]}
        ]}
        evaluates to: True

        If 'scopedData' argument does not evaluate to an array or if the array
        is empty, True is returned.

        N.B.: According to current core JsonLogic all 'scopedData' elements are
        evaluated before returning the result. It does not stop at first truthy
        value.
        """
        return len(self._filter(data, scopedData, scopedLogic)) == 0

    def _some(self, data, scopedData, scopedLogic):
        """
        Check if 'scopedLogic' evaluates to a truthy value for at least
        one 'scopedData' element.

        'scopedData' argument can be:
        - a manually specified data array;
        - a JsonLogic rule returning a data array;
        - a JsonLogic 'var' operation returning part of the data object
            containing a data array; like: {"var": "a"};
        - a JsonLogic 'var' operation returning the whole data object
            if it is an array itself; like: {"var": ""}.

        'scopedLogic' is a normal JsonLogic rule that uses a 'scopeData'
        element as its data object.

        Return True if 'scopedLogic' evaluates to a truthy value for at least
        one 'scopedData' element. Return False otherwise.

        Example:
        {"some": [
            [1, 2, 3, 4, 5],
            {"==":[{"var":""}, 3]}
        ]}
        evaluates to: True

        If 'scopedData' argument does not evaluate to an array or if the array
        is empty, False is returned.

        N.B.: According to current core JsonLogic all 'scopedData' elements are
        evaluated before returning the result. It does not stop at first truthy
        value.
        """
        return len(self._filter(data, scopedData, scopedLogic)) > 0

    @property
    def _scoped_operations(self):
        return {
            "filter": self._filter,
            "map": self._map,
            "reduce": self._reduce,
            "all": self._all,
            "none": self._none,
            "some": self._some,
        }

    # Data operations

    @staticmethod
    def _var(data, var_name=None, default=None):
        """
        Get variable value from the data object.
        Can also access variable properties (to any depth) via dot-notation:
            "variable.property"
            "variable.property.subproperty"
        The same is true for array elements that can be accessed by index:
            "array_variable.5"
            "variable.array_property.0.subproperty"
        Return the specified default value if variable, its property or element
        is not found. Return None if no default value is specified.
        Return the whole data object if variable name is None or an empty string.
        """
        if var_name is None or var_name == "":
            return data  # Return the whole data object
        try:
            for key in str(var_name).split("."):
                try:
                    data = data[key]
                except TypeError:
                    data = data[int(key)]
        except (KeyError, TypeError, ValueError):
            return default
        else:
            return data

    def _missing(self, data, *args):
        """
        Check if one or more variables are missing from data object.
        Take either:
        - multiple arguments (one variable name per argument) like:
            {"missing:["variable_1", "variable_2"]}.
        - a single argument that is an array of variable names like:
            {"missing": [["variable_1", "variable_2"]] (this typically happens
            if this operator is applied to the output of another operator
            (like 'if' or 'merge').
        Return an empty array if all variables are present and non-empty.
        Otherwise return a array of all missing variable names.

        N.B.: Per core JsonLogic, if missing variable name is provided several
        times it will also be represented several times in the resulting array.
        """
        missing_array = []
        var_names = args[0] if args and self._is_sequence(args[0]) else args
        for var_name in var_names:
            if self._var(data, var_name) in (None, ""):
                missing_array.append(var_name)
        return missing_array

    def _missing_some(self, data, need_count, args):
        """
        Check if at least some of the variables are missing from data object.
        Take two arguments:
        - minimum number of variables that are required to be present.
        - array of variable names to check for.
        I.e.: "{"missing_some":[1, ["a", "b", "c"]]}" means that at least one of
        the provided "a", "b" and "c" variables must be present in the data object.
        Return an empty array if minimum number of present variables is met.
        Otherwise return an array of all missing variable names.

        N.B.: Per core JsonLogic, if missing variable name is provided several
        times it will also be represented several times in the resulting array.
        In that case all occurrences are counted towards the minimum number of
        variables to be present and may lead to unexpected results.
        """
        missing_array = self._missing(data, args)
        if len(args) - len(missing_array) >= need_count:
            return []
        return missing_array

    @property
    def _data_operations(self):
        return {
            "var": self._var,
            "missing": self._missing,
            "missing_some": self._missing_some,
        }

    # Unsupported operations

    def _count(self, *args):
        """Execute 'count' operation unsupported by core JsonLogic."""
        if self._is_dataframe_or_series(args):
            return args.count()
        else:
            return sum(1 if a else 0 for a in args)

    def _get(self, a, b):
        """Execute 'get' operation on DataFrame or Dictionary or return None."""
        if self._is_dataframe(a) or self._is_dictionary(a):
            return a.get(b)
        else:
            return None

    def _query(self, a, b):
        """Execute 'query' operation DataFrame or return None."""
        if self._is_dataframe(a):
            return a.query(b)
        else:
            return None

    def _set_index(self, a, b):
        """Execute 'set_index' operation DataFrame or return None."""
        if self._is_dataframe(a):
            return a.set_index(b)
        else:
            return None

    @property
    def _unsupported_operations(self):
        return {
            "count": self._count,
            "get": self._get,
            "query": self._query,
            "set_index": self._set_index,
        }

    # Main Logic

    def _is_logic(self, logic):
        """
        Determine if specified object is a JsonLogic rule or not.
        A JsonLogic rule is a dictionary with exactly one key.
        An array of JsonLogic rules is not considered a rule itself.
        """
        return self._is_dictionary(logic) and len(logic.keys()) == 1

    @staticmethod
    def _get_operator(logic):
        """Return operator name from JsonLogic rule."""
        return next(iter(logic))

    def _get_values(self, logic, operator, normalize: bool = True):
        """Return array of values from JsonLogic rule by operator name."""
        values = logic[operator]
        # Easy syntax for unary operators like {"var": "x"}
        # instead of strict {"var": ["x"]}
        if normalize and not self._is_sequence(values):
            values = [values]
        return values

    def execute(self, logic, data=None):
        """
        Evaluate provided JsonLogic using given data (if any).
        If a single JsonLogic rule is provided - return a single resulting value.
        If an array of JsonLogic rule is provided - return an array of each rule's
        resulting values.
        """

        # Is this an array of JsonLogic rules?
        if self._is_sequence(logic):
            return logic
            # return list(map(lambda sublogic: self.execute(sublogic, data), logic))

        # You've recursed to a primitive, stop!
        if not self._is_logic(logic):
            return logic

        # Get operator
        operator = self._get_operator(logic)

        # Get values
        values = self._get_values(logic, operator)

        # Get data
        data = data or {}

        # Try applying logical operators first as they violate the normal rule of
        # depth-first calculating consequents. Let each manage recursion as needed.
        if operator in self._logical_operations:
            return self._logical_operations[operator](data, *values)

        # Next up, try applying scoped operations that manage their own data scopes
        # for each constituent operation
        if operator in self._scoped_operations:
            return self._scoped_operations[operator](data, *values)

        # Recursion!
        values = [self.execute(val, data) for val in values]

        # Apply data retrieval operations
        if operator in self._data_operations:
            return self._data_operations[operator](data, *values)

        # Apply simple custom operations (if any)
        if operator in self._custom_operations:
            return self._custom_operations[operator](*values)

        # Apply common operations
        if operator in self._common_operations:
            return self._common_operations[operator](*values)

        # Apply unsupported common operations if any
        if operator in self._unsupported_operations:
            return self._unsupported_operations[operator](*values)

        # Apply dot-notated custom operations (if any)
        suboperators = operator.split(".")
        if len(suboperators) > 1 and suboperators[0]:  # Dots in the middle
            current_operation = self._custom_operations
            for idx, suboperator in enumerate(suboperators):
                try:
                    if self._is_dictionary(current_operation):
                        try:
                            current_operation = current_operation[suboperator]
                        except (KeyError, IndexError):
                            current_operation = current_operation[int(suboperator)]
                    elif self._is_sequence(current_operation):
                        current_operation = current_operation[int(suboperator)]
                    else:
                        current_operation = getattr(current_operation, suboperator)
                except (KeyError, IndexError, AttributeError, ValueError):
                    raise ValueError(
                        "Unrecognized operation %r (failed at %r)"
                        % (operator, ".".join(suboperators[: idx + 1]))
                    )
            return current_operation(*values)

        # Report unrecognized operation
        raise ValueError("Unrecognized operation %r" % operator)

    def add_operation(self, name, code):
        """
        Add a custom common JsonLogic operation.

        Operation code must only take positional arguments that are absolutely
        necessary for its execution. JsonLogic will run it using the array of
        provided values, like: code(*values)

        Example:
        {"my_operation": [1, 2, 3]} will be called as code(1, 2, 3)

        Can also be used to add custom classed or even packages to extend JsonLogic
        functionality.

        Example:
        add_operation("datetime", datetime)

        Methods of such classes or packages can later be called using dot-notation.

        Example:
        {"datetime.datetime.now": []}
        can be used to retrieve current datetime value.
        {"datetime.date": [2018, 1, 1]}
        can be used to retrieve January 1, 2018 date.

        N.B.: Custom operations may be used to override common JsonLogic functions,
        but not logical, scoped or data retrieval ones.
        """
        self._custom_operations[str(name)] = code

    def rm_operation(self, name):
        """Remove previously added custom common JsonLogic operation."""
        del self._custom_operations[str(name)]
