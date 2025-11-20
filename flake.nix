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
              # cmake
              gnumake
              cmake
              neocmakelsp
              # c/c++
              gcc
              gdb
              clang-tools # must be before clang to have the correct clangd in PATH
              clang
              lldb
              # opencl
              ocl-icd
              opencl-headers
              # openmp (clang)
              llvmPackages.openmp
              # x11
              libx11
              # mpi
              mpi
              # cuda
              cudaPackages.cudatoolkit
              cudaPackages.cuda_cudart
            ];
            shellHook = ''
              set -a
              source .env 2> /dev/null
              MANPATH=${mpi.man}/share/man:$MANPATH
              PATH=$(realpath ./scripts):$PATH
              set +a
            '';
          };
      }
    );
}
