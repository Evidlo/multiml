# MultiML

See associated paper [here](https://github.com/evidlo/ICIP2022).

## Install

    pip install .
    
## Example

``` python
>>> from multiml.observation import test_sequence
>>> clean, noisy = test_sequence(drift=(10, 10))
>>> noisy.shape
(30, 250, 250)
>>> from multiml.multiml import register
>>> register(noisy)
(10, 10)
```

    
    
