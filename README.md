# SBFL
The Package for SBFL (spectrum-Based Fault Localisation)

## Implemented techniques
- Ochiai: `from sbfl import ochiai`
- Op2: `from sbfl import op2`
- DStar: `from sbfl import dstar`
- Tarantula: `from sbfl import tarantula`

## Package Installation
```bash
git clone https://github.com/agb94/sbfl
cd sbfl
python setup.py install
```

## Usage
```python
from sbfl import ochiai
from sbfl.utils import ranking

X = [
  [3, 5, 0, 1, 0], # test 1
  [1, 0, 5, 4, 0], # test 2
  [0, 1, 0, 3, 1]  # test 3
]
y = [
  1, # pass 
  0, # fail
  1  # pass
]

scores = ochiai(X, y)
# array([0.5       , 0.        , 0.70710678, 0.40824829, 0.        ])
ranking(scores)
# array([2, 5, 1, 3, 5])
```

The fault localisation methods expect two arguments: `X` and `y`.
- `X`: the coverage matrix
  - rows: test cases 
  - columns: program elements
  - `X[i,j]` represents how many times the jth program element is covered by the ith test case.
- `y`: the test result (0: fail, 1: pass).

The helper method `ranking` takes the scores and computes the ranking for each element (descending order of score)

## Create a custom scoring method from a user-defined ranking formula

```python
from sbfl.utils import matrix_to_index

def custom(X, y):
    def my_formula(ep, np, ef, nf):
        # calculating the score from the index
        score = ...
        return score
    return my_formula(*matrix_to_index(X, y))
 ```
### What does each index mean?
For each program element,
- `ep`: the number of passing tests execute the element
- `np`: the number of passing tests do not execute the element
- `ef`: the number of failing tests execute the element
- `nf`: the number of failing tests do not execute the element
