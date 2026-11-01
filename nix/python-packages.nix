{
  lib,
  pkgs,
  python,
  pyHarbor,
  rocmPkgs,
  torchvision-rocm,
  guiRuntimeLibs,
  gpuEnv,
  bekiper-rs,
}: let
  pythonPkgs = python.pkgs;
  isRocm = rocmPkgs != null;

  # uv2nix keeps dependency metadata on the generated package. Preserve it
  # while replacing binary-heavy wheels with the Nix builds used by Bekiper.
  mkUvExternal = name: package: prev:
    pkgs.symlinkJoin {
      name = "${name}-nix";
      paths = [package];
      passthru = prev.${name}.passthru or {};
    };

  torch =
    if isRocm
    then rocmPkgs.python313Packages.torchWithRocm
    else pythonPkgs.torch;
  torchvision =
    if isRocm && torchvision-rocm != null
    then torchvision-rocm
    else pythonPkgs.torchvision;

  pyprojectOverrides = final: prev: {
    bekiper-rs = mkUvExternal "bekiper-rs" bekiper-rs prev;
    torch = mkUvExternal "torch" torch prev;
    torchvision = mkUvExternal "torchvision" torchvision prev;
    opencv-python = mkUvExternal "opencv-python" pythonPkgs.opencv4 prev;

    bekiper-auto-detect = prev.bekiper-auto-detect.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.hatchling];
    });
    bekiper-inspect = prev.bekiper-inspect.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.hatchling];
    });
    bekiper-cli = prev.bekiper-cli.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.hatchling];
    });
  };

  uvArgs = {
    inherit pkgs python pyprojectOverrides;
    workspaceRoot = ../.;
    sourcePreference = "wheel";
    dependencies = {
      bekiper-cli = [];
    };
  };

  pythonCheckEnvironment = pyHarbor.mkUvCheckEnv (uvArgs
    // {
      name = "bekiper-python-check-env";
      dependencies = {
        bekiper-auto-detect = ["dev"];
        bekiper-cli = ["dev"];
        bekiper-inspect = ["dev"];
        mypy = [];
      };
    });

  wrapperEnv =
    gpuEnv
    // lib.optionalAttrs isRocm {
      ROCM_PATH = "${rocmPkgs.rocmPackages.clr}";
    };

  commonAppArgs = {
    inherit pkgs wrapperEnv;
    includeEnvironment = true;
    runtimePathPackages = [pkgs.ffmpeg_8-headless];
    runtimeLibraryPackages = guiRuntimeLibs;
  };

  bekiper-python-cli = pyHarbor.mkUvAppPackage (uvArgs
    // commonAppArgs
    // {
      name = "bekiper-python-cli";
      envName = "bekiper-python-env";
      scripts = ["bekiper" "bekiper-detect" "bekiper-inspect" "bekiper-post-analysis-render"];
    });

  inherit (bekiper-python-cli) pythonEnvironment;
in {
  inherit pythonEnvironment pythonCheckEnvironment bekiper-python-cli;
  bekiper-inspect = bekiper-python-cli;
  bekiper-auto-detect = bekiper-python-cli;
}
