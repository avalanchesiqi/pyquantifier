Header1
----------------------------

`pdoc` extracts documentation of:

* modules (including submodules),
* functions (including methods, properties, coroutines ...),
* classes, and
* variables (including globals, class variables, and instance variables).

Documentation is extracted from live objects' [docstrings]
using Python's `__doc__` attribute[^execution]. Documentation for
variables is found by examining objects' abstract syntax trees.

[docstrings]: https://docs.python.org/3/glossary.html#term-docstring

[^execution]:
    Documented modules are _executed_ in order to provide `__doc__`
    attributes. Any [non-fenced] global code in imported modules will
    _affect the current runtime environment_.

[non-fenced]: https://stackoverflow.com/questions/19578308/what-is-the-benefit-of-using-main-method-in-python/19578335#19578335


Header2
----------------------------
[public-private]: #what-objects-are-documented
`pdoc` only extracts _public API_ documentation.[^public]
Code objects (modules, variables, functions, classes, methods) are considered
public in the modules where they are defined (vs. imported from somewhere else)
as long as their _identifiers don't begin with an underscore_ ( \_ ).[^private]
If a module defines [`__all__`][__all__], then only the identifiers contained
in this list are considered public, regardless of where they were defined.

This can be fine-tuned through [`__pdoc__` dict][__pdoc__].

Header3
----------------------------
TKTK