{
  pkgs,
  craneLib,
  harbor-rs,
  metaDevShell,
  pyHarbor,
  opencodeLsp,
  cross,
  cargoConfig,
  treefmtWrapper,
  lib,
  rocmPkgs,
  torchvision-rocm,
  onnxruntime-rocm,
  python,
  venvRelDir,
  uvSyncArgs,
  guiRuntimeLibs,
  guiRuntimeRpathFlags,
  clangEnv,
  gpuEnv,
  mesa-radv,
}: let
  runtime = metaDevShell.mkRuntime {
    inherit pkgs;
    name = "bekiper-runtime";
    libraryPackages = guiRuntimeLibs;
    defaultEnv = lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
      VK_DRIVER_FILES = "${mesa-radv}/share/vulkan/icd.d/radeon_icd.${pkgs.stdenv.hostPlatform.parsed.cpu.name}.json";
    };
  };

  baseShell = harbor-rs.mkDevShell {
    inherit pkgs craneLib cross cargoConfig runtime;
    enableWindowsEnv = false;
    enableOsxcrossEnv = false;

    pkgConfigDeps = [
      pkgs.ffmpeg_8-headless
      pkgs.dav1d
      pkgs.libwebp
      pkgs.hdf5
      pkgs.openssl
      pkgs.libxkbcommon
      pkgs.libx11
      pkgs.libxcursor
      pkgs.libxi
      pkgs.libxrandr
      pkgs.wayland
    ];

    packages = [
      treefmtWrapper
      # The annotation server's interaction tests generate short fixture videos
      # with ffmpeg. Keep the executable available in the development shell;
      # pkgConfigDeps alone only exposes headers and linker metadata.
      pkgs.ffmpeg_8-headless
      pkgs.llvmPackages.libclang.lib
      pkgs.mdbook
      pkgs.sccache
      pkgs.vulkan-loader
    ];

    extraShellHook = "";
  };

  pythonShell = pyHarbor.mkUvDevShell {
    inherit pkgs python runtime;
    # GPU Python packages are supplied by Nix system-site-packages.
    uvExtra = null;
    devGroup = "dev";
    venvDir = venvRelDir;
    systemSitePackages = true;
    syncArgs = uvSyncArgs;
    extraPackages = lib.optionals (rocmPkgs != null) [
      rocmPkgs.python313Packages.torchWithRocm
      torchvision-rocm
      rocmPkgs.rocmPackages.clr
      onnxruntime-rocm
    ];
    pythonPathEntries = [];
    opencodeLsp = {enable = false;};
    shellHookSuffix = ''
      export PATH="$VIRTUAL_ENV/bin:$PWD/target/debug:$PATH"
      export PYTHONPATH="$PWD/python/bekiper-inspect:$PWD/python/bekiper-auto-detect:$PWD/python/bekiper-cli:''${PYTHONPATH:-}"

      MIGRAPHX_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/bekiper/migraphx"
      mkdir -p "$MIGRAPHX_CACHE_DIR/model"
      export ORT_MIGRAPHX_CACHE_PATH="$MIGRAPHX_CACHE_DIR"
      export ORT_MIGRAPHX_MODEL_CACHE_PATH="$MIGRAPHX_CACHE_DIR/model"
      export ORT_MIGRAPHX_SAVE_COMPILED_MODEL=0

      echo "Documentation: cd docs && mdbook serve" >&2
    '';
  };

  lspShell = opencodeLsp.mkShell {
    inherit pkgs;
    profiles = ["python"];
    rustAnalyzer = pkgs.rust-analyzer;
  };
in
  metaDevShell.mkShell {
    inherit pkgs;
    shells = [lspShell pythonShell baseShell];
    # git-lfs: design sources under design/review-package are LFS objects.
    packages = [runtime pkgs.git-lfs];
    mkShellArgs.passthru = {
      inherit runtime;
      inherit (pythonShell) venvPython;
    };
    env =
      clangEnv
      // gpuEnv
      // {
        RUSTC_WRAPPER = lib.getExe pkgs.sccache;
        RUSTFLAGS = lib.optionalString pkgs.stdenv.hostPlatform.isLinux (
          guiRuntimeRpathFlags + " -C link-arg=-fuse-ld=mold"
        );
        ROCM_PATH = lib.optionalString (rocmPkgs != null) "${rocmPkgs.rocmPackages.clr}";
        ORT_DYLIB_PATH = lib.optionalString (onnxruntime-rocm != null) "${onnxruntime-rocm}/lib/libonnxruntime.so";
      };
  }
