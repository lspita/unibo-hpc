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
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
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
              gcc
              clang-tools # must be before clang to have the correct clangd in PATH
              clang
              lldb
              gdb
              glibc
              libcxx
              # cuda
              cudaPackages.cudatoolkit
              # opencl
              ocl-icd
              opencl-headers
              # openmp (clang)
              llvmPackages.openmp
              # x11
              libx11
              # mpi
              mpi
            ];
          };
      }
    );
}
