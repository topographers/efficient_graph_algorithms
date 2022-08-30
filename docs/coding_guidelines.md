# Topographers coding standards
(This document is inspired by the [Open Knowledge Foundation Coding Standards](https://github.com/okfn/coding-standards))

This document outlines coding standards for use by all contributors to Topographers projects. It is a living document, 
and we encourage pull request and issues to improve on or contest ideas as expressed.

## TL;DR

* We use Python and C++. If you plan to develop in another language please flag this and discuss.
* Tests are required. Unit tests, as well as functional and integration tests. Aiming for test coverage of 80% and above is desirable.
  * Tests must be automated via a continuous integration platform that is triggered when code is pushed to the canonical repository. 
* Documentation is required for all code. Documentation is just as important as tests.
  * Document functions, classes, modules and packages with docstrings.
  * Provide a great `README.md` file with examples of how to use the code in the docs folder.
  * Only use documentation builders like Sphinx for large projects; prefer `README.md` files for brevity and accessibility of documentation.
* Use spaces and never tabs.
  * Python & C++: 4 space indentation.
* Strictly enforce a 120 character line limit.
* Lint Python using `pylint`.
* Use common language conventions for naming functions, classes and variables.
* Code should be submitted via pull requests, which another person should merge.
* Write small, reusable libraries where possible. There are many opportunities for reuse across our different products.

---

## Python Style and linting

1. Follow the Python Style Guide (PSG) as formulated in PEP-8: http://www.python.org/dev/peps/pep-0008/
2. Use `pylint` to lint code.

The critical points are:

* Use spaces; never use tabs
* 4 space indentation
* 120 character line limit
* Variables, functions and methods should be `lower_case_with_underscores` or snake_case
* Classes are `TitleCase`

And other preferences:

* Use ' and not " as the quote character by default
* When writing a method, consider if it is really a method (needs `self`) or if it would be better as a utility function
* When writing a `@classmethod`, consider if it really needs the class (needs `cls`) or it would be better as a utility function or factory class

As a rule, all Python code should be written to support Python 3.

## Python Testing

1. Use `tox` with `py.test` to test code.

## Python Documentation

### Docstrings

Use Sphinx-style or Google-style documentation conventions.

* http://packages.python.org/an_example_pypi_project/sphinx.html#function-definitions
* https://google.github.io/styleguide/pyguide.html#Comments

### User documentation

Prefer to make really good `README.md` files, rather than implementing a full documentation framework.

## Python Frameworks

We prefer the following frameworks and libraries. If you want to use an *alternative to one of these please flag this before starting any work.

* Torch

## Version control

We use Git for all projects.

### Branch management

We generally follow Git Flow, with some modifications, and some flexibility per project. The following should hold true for pretty much all projects:

* Have a `main` branch
* Never commit directly to `main`
* Always work from a `feature/{}` or a `fix/{}` branch that is checked out from `main`
* Always reference issues from Git messages using `#{issue_id}`, and the various other related conventions used by most Git hosts.
* Properly describe changes in commit messages: "Fixes database migration script failure on Python 2.7", not "Fix."

## Further reading

* http://docs.python-guide.org/en/latest/
