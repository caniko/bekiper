{
  inputs,
  pkgs,
  lib,
  craneLib,
  rocmPkgs,
  linuxWindowingLibs,
  guiRuntimeLibraryPath,
  python,
  pyHarbor,
  torchvision-rocm,
  guiRuntimeLibs,
  gpuEnv,
}: let
  src = lib.cleanSourceWith {
    src = ../.;
    filter = path: type: let
      pathString = toString path;
    in
      (!lib.hasPrefix (toString ../.cargo) pathString)
      && (
        (craneLib.filterCargoSources path type)
        || (
          type
          == "regular"
          && (
            lib.hasSuffix "/rust/bekiper-gpu/kernels/preprocess.hip" pathString
            || (
              lib.hasInfix "/rust/bekiper-behaviour/src/radial_arm/tests/golden/" pathString
              && lib.hasSuffix ".json" pathString
            )
            || (
              lib.hasInfix "/rust/bekiper-post-analysis/tests/fixtures/" pathString
              && lib.hasSuffix ".json" pathString
            )
            || lib.hasSuffix "/behavioral_post_analysis_harness/reference/BEHAVIORAL_RECIPE.md" pathString
            || lib.hasSuffix "/behavioral_post_analysis_harness/reference/YMAZE_AMENDMENT.json" pathString
            || lib.hasSuffix "/examples/neo/imen/post_analysis_plan.toml" pathString
          )
        )
      );
  };

  ortGitSource = "git+https://github.com/pykeio/ort.git#5e669f6944911bb3fc34014680ec91f74cee30a0";
  craneCargoOutputHashes = {
    "${ortGitSource}" = "sha256-FGZO/Q740yOZEZMMUSNFgKMZiW89W8Y6D1jRpIjuJKk=";
  };

  # Custom vendoring for the ort git workspace (has backends with
  # default-members = [] that make `cargo package -l` fail). Copies the
  # source tree directly instead of running cargo metadata/package.
  ortOverrideGitCheckout = packages: drv:
    if builtins.any (package: builtins.elem package.name ["ort" "ort-sys"] && package.version == "2.0.0-rc.12") packages
    then
      drv.overrideAttrs (_old: {
        installPhase = ''
          runHook preInstall

          mkdir -p "$out/ort-2.0.0-rc.12" "$out/ort-sys-2.0.0-rc.12"
          cp -R --no-preserve=mode,ownership "$src"/. "$out/ort-2.0.0-rc.12/"
          cp -R --no-preserve=mode,ownership "$src/ort-sys"/. "$out/ort-sys-2.0.0-rc.12/"
          echo '{"files":{}, "package":null}' > "$out/ort-2.0.0-rc.12/.cargo-checksum.json"
          echo '{"files":{}, "package":null}' > "$out/ort-sys-2.0.0-rc.12/.cargo-checksum.json"

          runHook postInstall
        '';
      })
    else drv;

  cargoVendorDir = craneLib.vendorCargoDeps {
    inherit src;
    cargoLock = ../Cargo.lock;
    outputHashes = craneCargoOutputHashes;
    overrideVendorGitCheckout = ortOverrideGitCheckout;
  };

  cargoDeps = pkgs.runCommand "bekiper-cargo-deps" {} ''
    install -Dm644 ${cargoVendorDir}/config.toml "$out/.cargo/config.toml"
    install -Dm644 ${../Cargo.lock} "$out/Cargo.lock"
  '';

  commonArgs = {
    inherit src;
    strictDeps = true;
    inherit cargoVendorDir;

    nativeBuildInputs =
      [
        pkgs.pkg-config
        pkgs.llvmPackages.libclang.lib
      ]
      ++ lib.optionals (rocmPkgs != null) [
        rocmPkgs.rocmPackages.clr
        rocmPkgs.rocmPackages.hipcc
        # HIP_CLANG_PATH selects the GPU compiler directly. Adding the toolchain
        # here injects its older libc into native RUNPATH, mismatching the native
        # loader and causing ordinary Rust build scripts to abort at startup.
        rocmPkgs.rocmPackages.rocm-device-libs
      ];
    env =
      {
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        PYTORCH_ROCM_ARCH = "gfx1100";
        BEKIPER_CODE_REVISION = inputs.self.rev or (inputs.self.dirtyRev or "unknown");
        # The export regenerates Cargo.lock; engine identity binds to the original.
        BEKIPER_SOURCE_CARGO_LOCK = "${../Cargo.lock.source}";
        BEKIPER_CODE_DIRTY =
          if inputs.self ? rev
          then "false"
          else if inputs.self ? dirtyRev
          then "true"
          else "unknown";
        BINDGEN_EXTRA_CLANG_ARGS = toString [
          "-isystem"
          "${pkgs.llvmPackages.libclang.lib}/lib/clang/${lib.getVersion pkgs.llvmPackages.libclang}/include"
          "-isystem"
          "${pkgs.glibc.dev}/include"
        ];
      }
      // lib.optionalAttrs (rocmPkgs != null) {
        ROCM_PATH = "${rocmPkgs.rocmPackages.clr}";
        HIP_PATH = "${rocmPkgs.rocmPackages.clr}";
        HIP_CLANG_PATH = "${rocmPkgs.rocmPackages.rocm-toolchain}/bin";
        DEVICE_LIB_PATH = "${rocmPkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode";
      };

    buildInputs =
      [
        pkgs.ffmpeg_8-headless.dev
        pkgs.dav1d.dev
        pkgs.libwebp
        pkgs.openssl.dev
        pkgs.vulkan-loader
        # libhdf5 for bekiper-pose's DeepLabCut .h5 reader (hdf5-metno-sys
        # uses pkg-config to find it).
        pkgs.hdf5
      ]
      ++ linuxWindowingLibs;
  };

  cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  individualCrateArgs =
    commonArgs
    // {
      inherit cargoArtifacts;
      inherit (craneLib.crateNameFromCargoToml {inherit src;}) version;
      doCheck = false;
    };

  commonCrateFileSet = lib.fileset.unions [
    ../Cargo.toml
    ../Cargo.lock
    (craneLib.fileset.commonCargoSources ../rust/bekiper-core)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-math)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-perimeter)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-feature)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-behaviour)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-detect)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-annotation)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-annotate-protocol)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-gpu)
    ../rust/bekiper-gpu/kernels/preprocess.hip
    (craneLib.fileset.commonCargoSources ../rust/bekiper-ingress)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-post-analysis)
    ../behavioral_post_analysis_harness/reference/BEHAVIORAL_RECIPE.md
    ../behavioral_post_analysis_harness/reference/YMAZE_AMENDMENT.json
    (craneLib.fileset.commonCargoSources ../rust/bekiper-pose)
    (craneLib.fileset.commonCargoSources ../rust/bekiper-inspect-py)
    (craneLib.fileset.commonCargoSources ../rust/themekit)
  ];

  fileSetForCrates = crates:
    lib.fileset.toSource {
      root = ../.;
      fileset = lib.fileset.unions ([commonCrateFileSet] ++ map craneLib.fileset.commonCargoSources crates);
    };

  # Build the CLI as the top-level binary derivation (no GUI, no annotate).
  bekiper-cli = craneLib.buildPackage (
    individualCrateArgs
    // {
      pname = "bekiper-cli";
      cargoExtraArgs = "-p bekiper-cli --features full";
      src = fileSetForCrates [../rust/bekiper-cli];
      nativeBuildInputs =
        individualCrateArgs.nativeBuildInputs
        ++ [pkgs.makeWrapper];
      postInstall = ''
        wrapProgram "$out/bin/bekiper" --prefix PATH : "${cliRuntimePath}" --prefix LD_LIBRARY_PATH : "${guiRuntimeLibraryPath}"
      '';
    }
  );

  # Rust-backed Python types (PyO3 extension built with maturin).
  bekiper-rs = pkgs.python313.pkgs.buildPythonPackage {
    pname = "bekiper-rs";
    version = "0.1.0";
    format = "pyproject";
    src = lib.fileset.toSource {
      root = ../.;
      fileset = lib.fileset.unions [
        (craneLib.fileset.commonCargoSources ../.)
        ../rust/bekiper-inspect-py/python
        ../rust/bekiper-inspect-py/pyproject.toml
      ];
    };
    sourceRoot = "source/rust/bekiper-inspect-py";
    cargoRoot = "../..";
    inherit cargoDeps;
    env.CARGO_TARGET_DIR = "target";

    nativeBuildInputs = [pkgs.rustPlatform.cargoSetupHook];
    build-system = [pkgs.rustPlatform.maturinBuildHook];

    maturinBuildFlags = ["--locked"];
  };

  pythonPackages = import ./python-packages.nix {
    inherit lib pkgs python pyHarbor rocmPkgs torchvision-rocm guiRuntimeLibs gpuEnv bekiper-rs;
  };

  inherit (pythonPackages) bekiper-inspect bekiper-python-cli bekiper-auto-detect;

  # The plotting protocol imports only argparse/json, Matplotlib and the
  # bekiper-rs theme palette. Package a dedicated environment so headless
  # analysis does not inherit detection/GPU dependencies.
  postAnalysisRenderer = python.pkgs.buildPythonPackage {
    pname = "bekiper-post-analysis-renderer";
    version = "0.1.0";
    format = "pyproject";
    src = ../python/bekiper-cli;
    nativeBuildInputs = [python.pkgs.hatchling python.pkgs.pythonRelaxDepsHook];
    propagatedBuildInputs = [python.pkgs.matplotlib bekiper-rs];
    pythonRemoveDeps = ["typer" "tomli-w" "bekiper-auto-detect" "bekiper-inspect"];
    pythonImportsCheck = ["bekiper_cli.post_analysis.render_bundle"];
    postInstall = ''
      rm -f "$out/bin/bekiper" "$out/bin/bekiper-detect" "$out/bin/bekiper-inspect"
    '';
  };
  postAnalysisPython = python.withPackages (_: [postAnalysisRenderer]);
  postAnalysisTex = pkgs.texlive.combine {
    inherit (pkgs.texlive) scheme-small latexmk luatex;
  };
  postAnalysisRuntimePath = lib.makeBinPath [postAnalysisPython postAnalysisTex pkgs.poppler-utils];
  # The full CLI also delegates inspection and HDF5 conversion to Python.
  cliRuntimePath = lib.makeBinPath [pythonPackages.pythonEnvironment postAnalysisTex pkgs.poppler-utils pkgs.ffmpeg_8-headless pkgs.gnutar pkgs.zstd];

  # The standalone statistical engine has no FFmpeg, HDF5 or GPU dependency.
  # Keep its dependency artifacts separate from the full CLI workspace build.
  postAnalysisArgs = {
    pname = "bekiper-post-analysis";
    inherit (individualCrateArgs) version;
    src = fileSetForCrates [../rust/bekiper-post-analysis];
    inherit cargoVendorDir;
    strictDeps = true;
    cargoExtraArgs = "-p bekiper-post-analysis --bin bekiper-post-analysis";
    nativeBuildInputs = [pkgs.makeWrapper];
    env = {
      inherit (commonArgs.env) BEKIPER_CODE_REVISION BEKIPER_CODE_DIRTY BEKIPER_SOURCE_CARGO_LOCK;
    };
  };
  postAnalysisCargoArtifacts = craneLib.buildDepsOnly postAnalysisArgs;
  bekiper-post-analysis = craneLib.buildPackage (postAnalysisArgs
    // {
      cargoArtifacts = postAnalysisCargoArtifacts;
      postInstall = ''
        wrapProgram "$out/bin/bekiper-post-analysis" \
          --prefix PATH : "${postAnalysisRuntimePath}"
      '';
    });
  bekiper-post-analysis-renderer = postAnalysisPython;
  bekiper-post-analysis-tex = postAnalysisTex;

  # Combined CLI: native commands can find the Python entry points even when
  # invoked through `nix run`, which does not add sibling scripts to PATH.
  bekiper = pkgs.symlinkJoin {
    name = "bekiper";
    paths = [bekiper-cli bekiper-python-cli bekiper-post-analysis];
    nativeBuildInputs = [pkgs.makeWrapper];
    postBuild = ''
      rm "$out/bin/bekiper"
      makeWrapper ${bekiper-cli}/bin/bekiper "$out/bin/bekiper" \
        --prefix PATH : "${bekiper-python-cli}/bin"
    '';
  };
in {
  inherit
    src
    cargoArtifacts
    commonArgs
    bekiper-cli
    bekiper-post-analysis
    bekiper-post-analysis-renderer
    bekiper-post-analysis-tex
    bekiper-inspect
    bekiper-python-cli
    bekiper-auto-detect
    bekiper
    bekiper-rs
    ;
}
