from functools import partial, reduce
from itertools import product
import operator


class ParameterGrid:
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.
    The order of the generated parameter combinations is deterministic.

    Parameters
    ----------
    param : dict of str to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Examples
    --------
    >>> from model_selection import ParameterGrid
    >>> param_grid = {'a': {'b': [1, 2], 'c': [True, False]}}
    >>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': {'b': 1, 'c': True}},
            {'a': {'b': 1, 'c': False},
    ...     {'a': {'b': 2, 'c': True},
            {'a': {'b': 2, 'c': False})
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                               {'kernel': 'rbf', 'gamma': 1},
    ...                               {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True
    """

    def __init__(self, param_grid):
        if not isinstance(param_grid, (dict, list)):
            raise TypeError(
                f"Parameter grid should be a dict or a list, got: {param_grid!r} of"
                f" type {type(param_grid).__name__}"
            )

        if isinstance(param_grid, dict):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        self.items = list(map(self.extract_items, param_grid))
        # Product function that can handle iterables.
        self.product = partial(reduce, operator.mul)

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for keys, values in self.items:
            values = product(*values)
            generate_params = partial(self.generate_params, keys)
            for value_tuple in values:
                yield generate_params(value_tuple)

    def __len__(self):
        """Number of points on the grid."""
        return sum(
            self.product(len(value_list) for value_list in values)
            for _, values in self.items
        )

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
            The iteration index

        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """

        if not isinstance(ind, int):
            raise TypeError(
                f"index should be a integer, got: {ind!r} of"
                f" type {type(ind).__name__}"
            )
        total = sum(
            self.product(len(value_tuple) for value_tuple in values)
            for _, values in self.items
        )
        if ind < 0:
            ind += total
        if ind >= total:
            raise IndexError("ParameterGrid index out of range")
        # This is used to make discrete sampling without replacement memory efficient.
        for keys, values in self.items:
            subtotal = self.product(len(value_tuple) for value_tuple in values)
            if ind >= subtotal:
                # Try the next grid
                ind -= subtotal
            else:
                size = len(values)
                ways = [len(value_list) for value_list in values]
                value_list = [None] * size
                for i in reversed(range(size)):
                    value_list[i] = values[i][ind % ways[i]]
                    ind //= ways[i]
                yield self.generate_params(keys, value_list)

    def extract_items(self, grid: dict) -> tuple:
        """Extract parameters and paths to ones.

        Parameters
        ----------
        grid : dict
            Parameter grid

        Returns
        -------
        keys : 2-dimensional list
            List of keys representing paths to parameters
        values: list of any
            List of parameter values
        """

        keys = []
        values = []

        def traverse_dict(current_keys, current_dict):
            if len(current_dict) == 0:
                raise ValueError(
                    f"Parameter grid should be a non-empty dict, got: {current_dict!r}"
                )
            for key, value in current_dict.items():
                if isinstance(value, dict):
                    traverse_dict(current_keys + [key], value)
                elif isinstance(value, list):
                    if len(value) == 0:
                        raise ValueError(
                            f"Parameter grid for parameter {key!r} need "
                            f"to be a non-empty list, got: {value!r}"
                        )
                    keys.append(current_keys + [key])
                    values.append(value)
                else:
                    raise TypeError(
                        f"Parameter grid should be a dict or a list, got: {value!r} of"
                        f" type {type(value).__name__}"
                    )

        traverse_dict([], grid)
        return keys, values

    def generate_params(self, keys: list, values):
        """Generate parameters.

        Parameters
        ----------
        keys : 2-dimensional list
            List of keys representing paths to parameters
        values : list of integers
            List of parameter values

        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """

        params = {}
        for key_list, v in zip(keys, values):
            current_dict = params
            for k in key_list[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            current_dict[key_list[-1]] = v
        return params
