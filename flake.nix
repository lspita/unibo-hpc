{
  description = "Generic nix devshell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShell =
          with pkgs;
          mkShell {
            buildInputs = [
              # nix
              nixd
              nil
              nixfmt
              # c/c++
              gnumake
              cmake
              clang
              gcc
              clang-tools
              lldb
              gdb
              glibc
              libcxx
              # OpenCL
              ocl-icd
              opencl-headers
              # OpenMP (clang)
              llvmPackages.openmp
              # X11
              libx11
              # mpi
              mpi
            ];
          };
      }
    );
}
