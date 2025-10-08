# C/C++ project template

This is a template C/C++ project managed with CMake.

## Usage
CMake presets for every major compiler are available
- clang
- gcc
- msvc (cl)

After running the cmake configuration with the chosen preset, a Makefile is generated with the following targets:
- `run <exec> <args>` (default target): Build and run the specified executable, or the source root main executable if not specified.
- `test <args>`: Build and run all tests.
- `clean`: Delete generated cmake files (except the Makefile itself)
- `build`: Build the project.

## Configuration
Inside the [cmake](./cmake/) dir you can find the config files you need
- [setup.cmake](./cmake/setup.cmake): Included at the top of the cmake file. Used to define/override variables.
- [config.cmake](./cmake/config.cmake): Included at the end of the cmake file. Used to add extra configuration.
- [templates](./templates): Config files to configure with cmake and put in the project root. The path relative to the templates directory is the same relative to the project root.
