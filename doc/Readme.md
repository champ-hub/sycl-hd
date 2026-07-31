# Documentation
The documentation for this library is provided in the source code in Doxygen format, that can be optionally built locally in the default mode for *html* and *LaTeX* formats.


## Build documentation
To build the documentation make sure the `-DBUILD_WITH_DOCS=ON` option is set, either when running CMake or in the `CMakeCache.txt` file inside your build directory.

Next, simply run:
```bash
make docs
```
This will create two new directories under `doc/doxygen/` which contain the *html* and *LaTeX* built documentation.

