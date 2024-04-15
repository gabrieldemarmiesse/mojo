# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements Python interoperability.

You can import these APIs from the `python` package. For example:

```mojo
from python import Python
```
"""

from sys import external_call, sizeof
from sys.ffi import _get_global

from memory import Pointer

from utils import StringRef

from ._cpython import CPython, Py_eval_input
from .object import PythonObject


fn _init_global(ignored: Pointer[NoneType]) -> Pointer[NoneType]:
    var ptr = Pointer[CPython].alloc(1)
    ptr[] = CPython()
    return ptr.bitcast[NoneType]()


fn _destroy_global(python: Pointer[NoneType]):
    var p = python.bitcast[CPython]()
    CPython.destroy(p[])
    python.free()


@always_inline
fn _get_global_python_itf() -> _PythonInterfaceImpl:
    var ptr = _get_global["Python", _init_global, _destroy_global]()
    return ptr.bitcast[CPython]()


struct _PythonInterfaceImpl:
    var _cpython: Pointer[CPython]

    fn __init__(inout self, cpython: Pointer[CPython]):
        self._cpython = cpython

    fn __copyinit__(inout self, existing: Self):
        self._cpython = existing._cpython

    fn cpython(self) -> CPython:
        return self._cpython[]


struct Python:
    """Provides methods that help you use Python code in Mojo."""

    var impl: _PythonInterfaceImpl
    """The underlying implementation of Mojo's Python interface."""

    fn __init__(inout self):
        """Default constructor."""
        self.impl = _get_global_python_itf()

    fn __copyinit__(inout self, existing: Self):
        """Copy constructor.

        Args:
            existing: The existing instance to copy from.
        """
        self.impl = existing.impl

    fn eval(inout self, code: StringRef) -> Bool:
        """Executes the given Python code.

        Args:
            code: The python code to execute.

        Returns:
            `True` if the code executed successfully or `False` if the code
            raised an exception.
        """
        var cpython = self.impl.cpython()
        return cpython.PyRun_SimpleString(code)

    @staticmethod
    fn evaluate(expr: StringRef) raises -> PythonObject:
        """Executes the given Python code.

        Args:
            expr: The Python expression to evaluate.

        Returns:
            `PythonObject` containing the result of the evaluation.
        """
        var cpython = _get_global_python_itf().cpython()
        var module = PythonObject(cpython.PyImport_AddModule("__main__"))
        # PyImport_AddModule returns a borrowed reference - IncRef it to keep it alive.
        cpython.Py_IncRef(module.py_object)
        var dict_obj = PythonObject(cpython.PyModule_GetDict(module.py_object))
        # PyModule_GetDict returns a borrowed reference - IncRef it to keep it alive.
        cpython.Py_IncRef(dict_obj.py_object)
        var result = cpython.PyRun_String(
            expr, dict_obj.py_object, dict_obj.py_object, Py_eval_input
        )
        # We no longer need module and dictionary, release them.
        Python.throw_python_exception_if_error_state(cpython)
        return PythonObject(result)

    @staticmethod
    fn add_to_path(dir_path: String) raises:
        """Adds a directory to the Python path.

        This might be necessary to import a Python module via `import_module()`.
        For example:

        ```mojo
        from python import Python

        # Specify path to `mypython.py` module
        Python.add_to_path("path/to/module")
        var mypython = Python.import_module("mypython")

        var c = mypython.my_algorithm(2, 3)
        ```

        Args:
            dir_path: The path to a Python module you want to import.
        """
        var cpython = _get_global_python_itf().cpython()
        var sys = Python.import_module("sys")
        var directory: PythonObject = dir_path
        _ = sys.path.append(directory)

    @staticmethod
    fn import_module(module: StringRef) raises -> PythonObject:
        """Imports a Python module.

        This provides you with a module object you can use just like you would
        in Python. For example:

        ```mojo
        from python import Python

        # This is equivalent to Python's `import numpy as np`
        var np = Python.import_module("numpy")
        a = np.array([1, 2, 3])
        ```

        Args:
            module: The Python module name. This module must be visible from the
                list of available Python paths (you might need to add the
                module's path with `add_to_path()`).

        Returns:
            The Python module.
        """
        var cpython = _get_global_python_itf().cpython()
        var module_maybe = cpython.PyImport_ImportModule(module)
        Python.throw_python_exception_if_error_state(cpython)
        return PythonObject(module_maybe)

    @staticmethod
    fn dict() -> PythonObject:
        """Construct an empty Python dictionary.

        Returns:
            The constructed empty Python dictionary.
        """
        return PythonObject(Dict[PythonObject, PythonObject]())

    @staticmethod
    fn list() -> PythonObject:
        """Construct an empty Python list.

        Returns:
            The constructed empty Python list.
        """
        return PythonObject([])

    fn __str__(inout self, str_obj: PythonObject) -> StringRef:
        """Return a string representing the given Python object.

        This function allows to convert Python objects to Mojo string type.

        Returns:
            Mojo string representing the given Python object.
        """
        var cpython = self.impl.cpython()
        return cpython.PyUnicode_AsUTF8AndSize(str_obj.py_object)

    @staticmethod
    fn throw_python_exception_if_error_state(inout cpython: CPython) raises:
        """Raise an exception if CPython interpreter is in an error state.

        Args:
            cpython: The cpython instance we wish to error check.
        """
        if cpython.PyErr_Occurred():
            var error = PythonObject(cpython.PyErr_Fetch()).__getattr__(
                "__str__"
            )()
            var err: Error = cpython.PyUnicode_AsUTF8AndSize(error.py_object)
            cpython.PyErr_Clear()
            raise err

    @staticmethod
    fn is_type(x: PythonObject, y: PythonObject) -> Bool:
        """Test if the `x` object is the `y` object, the same as `x is y` in
        Python.

        Args:
            x: The left-hand-side value in the comparison.
            y: The right-hand-side type value in the comparison.

        Returns:
            True if `x` and `y` are the same object and False otherwise.
        """
        var cpython = _get_global_python_itf().cpython()
        return cpython.Py_Is(x.py_object, y.py_object)

    @staticmethod
    fn type(obj: PythonObject) -> PythonObject:
        """Return Type of this PythonObject.

        Args:
            obj: PythonObject we want the type of.

        Returns:
            A PythonObject that holds the type object.
        """
        var cpython = _get_global_python_itf().cpython()
        return cpython.PyObject_Type(obj.py_object)

    @staticmethod
    fn none() -> PythonObject:
        """Get a `PythonObject` representing `None`.

        Returns:
            `PythonObject` representing `None`.
        """
        return PythonObject(None)
