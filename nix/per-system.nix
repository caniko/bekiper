{inputs}: {
  lib,
  system,
  ...
}: let
  pkgs' = import inputs.nixpkgs {
    inherit system;
    overlays = [(import inputs.rust-overlay)];
  };

  pyHarbor = inputs.harbor-py.lib;
  clangEnv = {
    LIBCLANG_PATH = "${pkgs'.llvmPackages.libclang.lib}/lib";
    BINDGEN_EXTRA_CLANG_ARGS = toString [
      "-isystem"
      "${pkgs'.llvmPackages.libclang.lib}/lib/clang/${lib.getVersion pkgs'.llvmPackages.libclang}/include"
      "-isystem"
      "${lib.getDev pkgs'.glibc}/include"
    ];
  };
  gpuEnv = {
    HSA_OVERRIDE_GFX_VERSION = "11.0.0";
    PYTORCH_ROCM_ARCH = "gfx1100";
  };

  toolchain = inputs.harbor-rs.lib.mkToolchain {
    pkgs = pkgs';
    channel = "stable";
    crossTargets = ["x86_64-unknown-linux-gnu"];
    extensions = ["rust-src" "rustfmt"];
  };

  fmtToolchain = inputs.harbor-rs.lib.mkToolchain {
    pkgs = pkgs';
    toolchainProfile = "nightly";
  };
  treefmtEval = inputs.treefmt-nix.lib.evalModule pkgs' (import ./treefmt.nix {
    inherit (inputs) harbor-rs;
    rustfmtPackage = fmtToolchain.rustToolchain;
  });

  cross = inputs.harbor-rs.lib.mkCross {
    pkgs = pkgs';
    inherit system;
    enableOsxcross = false;
  };

  cargoConfig = inputs.harbor-rs.lib.mkCargoConfig {
    pkgs = pkgs';
    channel = "stable";
    inherit (toolchain) crossTargets;
    enableCranelift = false;
    enableShareGenerics = false;
    enableParallelFrontend = false;
  };

  rocmRuntime = import ./rocm-runtime.nix {
    pkgs = pkgs';
    inherit lib;
    rocmPkgs = inputs.nixpkgs-rocm.legacyPackages.${system}.pkgsRocm or null;
    mesaPkgs = inputs.nixpkgs-rocm.legacyPackages.${system};
  };

  inherit
    (rocmRuntime)
    rocmPkgs
    torchvision-rocm
    onnxruntime-rocm
    mesa-radv
    linuxWindowingLibs
    guiRuntimeLibs
    guiRuntimeLibraryPath
    guiRuntimeRpathFlags
    ;

  packages = import ./packages.nix {
    inherit inputs lib rocmPkgs linuxWindowingLibs guiRuntimeLibraryPath;
    inherit python pyHarbor torchvision-rocm guiRuntimeLibs gpuEnv;
    pkgs = pkgs';
    inherit (toolchain) craneLib;
  };

  python = pkgs'.python313;
  venvRelDir = "python/.venv";

  uvExcludedPackages = [
    "torch"
    "torchvision"
    "pytorch-triton-rocm"
  ];

  uvSyncArgs = lib.cli.toCommandLineGNU {} {
    no-install-package = uvExcludedPackages;
    quiet = true;
  };
in {
  formatter = treefmtEval.config.build.wrapper;

  packages = {
    cargo-artifacts = packages.cargoArtifacts;
    inherit
      (packages)
      bekiper-cli
      bekiper-post-analysis
      bekiper-post-analysis-renderer
      bekiper-post-analysis-tex
      bekiper-inspect
      bekiper-python-cli
      bekiper
      bekiper-rs
      ;
    default = packages.bekiper;
    inherit (packages) bekiper-auto-detect;
  };

  devShells.default = import ./devshell.nix {
    inherit
      lib
      rocmPkgs
      torchvision-rocm
      onnxruntime-rocm
      python
      venvRelDir
      uvSyncArgs
      guiRuntimeLibs
      guiRuntimeRpathFlags
      mesa-radv
      clangEnv
      gpuEnv
      pyHarbor
      ;
    pkgs = pkgs';
    inherit (toolchain) craneLib;
    harbor-rs = inputs.harbor-rs.lib;
    metaDevShell = inputs.harbor-meta.lib.devShell;
    opencodeLsp = inputs.harbor-rs.inputs.nix-opencode-lsp.lib;
    inherit cross;
    inherit cargoConfig;
    treefmtWrapper = treefmtEval.config.build.wrapper;
  };
}
