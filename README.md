# ParameterGrid
A utility for generating parameter combinations and facilitating iteration, providing enhanced support for a wide range of grid structures compared to `sklearn.model_selection.ParameterGrid`.
## Usage and comparison with `sklearn.model_selection.ParameterGrid`
This `ParameterGrid` iterator can extract list of parameters from the grid and gerenate all combinations of them, regardless of the nesting level of the values.
### Example
```python
param_grid = {'a': [1,2], 'b': {'c': [True, False]}}

list(ParameterGrid(param_grid)) == (
    [{'a': 1, 'b': {'c': True}}, {'a': 1, 'b': {'c': False}},
    {'a': 2, 'b': {'c': True}}, {'a': 2, 'b': {'c': False}}])
```
```console
True
```
you would encounter an error if you try to perform the same operation using `sklearn.model_selection.ParameterGrid`.
```python
from sklearn.model_selection import ParameterGrid

param_grid = {'a': [1,2], 'b': {'c': [True, False]}}

ParameterGrid(param_grid)
```
```console
TypeError: Parameter grid for parameter 'b' needs to be a list or a numpy array,
but it got {'c': [True, False]} (of type dict) instead. Single values need to be wrapped in a list with one element.
```
