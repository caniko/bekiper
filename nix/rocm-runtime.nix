{
  pkgs,
  lib,
  rocmPkgs,
  mesaPkgs,
}: let
  # Hydra-cached (cache-pin `pins.rocm`) torchvision. Its ops are CPU-only;
  # bekiper_auto_detect.detect registers a pure-torch GPU `torchvision::nms`.
  torchvision-rocm =
    if rocmPkgs != null
    then rocmPkgs.python313Packages.torchvision
    else null;

  # ONNX Runtime with ROCm support for GPU inference.
  # nixpkgs builds with MIGraphX EP (not ROCm/HIP EP).
  onnxruntime-rocm =
    if rocmPkgs != null
    then rocmPkgs.onnxruntime.override {rocmSupport = true;}
    else null;

  # RADV-only Mesa without LLVM — prevents the LLVM symbol conflict
  # between MIGraphX's bundled LLVM and Mesa's system LLVM.
  # RADV uses ACO (not LLVM) for shader compilation, so this is safe.
  # Built from the ROCm pin, not `nixpkgs`: no cache carries this override, so
  # tracking unstable would recompile Mesa on every nixpkgs bump.
  mesa-radv =
    (mesaPkgs.mesa.override {
      vulkanDrivers = ["amd"];
      galliumDrivers = [];
      eglPlatforms = [];
      vulkanLayers = [];
    }).overrideAttrs (old: {
      pname = "mesa-radv";
      outputs = ["out"];
      # Switch to auto_features=disabled so we only get what we explicitly enable
      mesonAutoFeatures = "disabled";
      # Skip OpenCL/spirv2dxil post-install hooks (we only build RADV)
      postInstall = "";
      postFixup = "";
      mesonFlags = let
        # Flags to strip from the base Mesa build (conflict with RADV-only + no-LLVM)
        strip = [
          "-Dglvnd"
          "-Dgbm"
          "-Dlibgbm-external"
          "-Dteflon"
          "-Dclang-libdir"
          "-Dgallium-rusticl"
          "-Dgallium-extra-hud"
          "-Dvalgrind"
          "-Dintel-rt"
          "-Dvideo-codecs"
          "-Dtools"
          "-Dinstall-mesa-clc"
          "-Dinstall-precomp-compiler"
          "-Dfreedreno-kmds"
          "-Damdgpu-virtio"
          "-Dgallium-rusticl-enable-drivers"
        ];
        keep = f: !builtins.any (s: lib.hasPrefix s f) strip;
      in
        builtins.filter keep (old.mesonFlags or [])
        ++ [
          "-Dllvm=disabled"
          "-Dglvnd=disabled"
          "-Degl=disabled"
          "-Dglx=disabled"
          "-Dgbm=disabled"
          "-Dinstall-mesa-clc=false"
          "-Dinstall-precomp-compiler=false"
        ];
      # Remove LLVM/clang from build inputs since we disabled LLVM
      buildInputs = builtins.filter (
        p: let
          name = p.pname or (builtins.parseDrvName p.name).name;
        in
          !builtins.elem name ["libllvm" "clang" "clang-unwrapped" "libclc" "spirv-llvm-translator"]
      ) (old.buildInputs or []);
      nativeBuildInputs = builtins.filter (
        p: let
          name = p.pname or (builtins.parseDrvName p.name).name;
        in
          !builtins.elem name ["clang" "clang-unwrapped"]
      ) (old.nativeBuildInputs or []);
    });

  linuxWindowingLibs = lib.optionals pkgs.stdenv.hostPlatform.isLinux [
    pkgs.libxkbcommon
    pkgs.wayland
    pkgs.libx11
    pkgs.libxcb
    pkgs.libxcursor
    pkgs.libxi
    pkgs.libxrandr
  ];

  guiRuntimeLibs = lib.optionals pkgs.stdenv.hostPlatform.isLinux (
    [
      pkgs.stdenv.cc.cc.lib
      pkgs.vulkan-loader
      pkgs.libGL
      pkgs.glib
      mesa-radv
      pkgs.zlib
      pkgs.zstd
      pkgs.ffmpeg_8-headless
      pkgs.dav1d
      pkgs.libwebp
      pkgs.hdf5
    ]
    ++ linuxWindowingLibs
    ++ lib.optionals (rocmPkgs != null) [
      rocmPkgs.rocmPackages.clr
      onnxruntime-rocm
    ]
  );
in {
  inherit rocmPkgs torchvision-rocm onnxruntime-rocm mesa-radv linuxWindowingLibs guiRuntimeLibs;

  guiRuntimeLibraryPath = lib.makeLibraryPath guiRuntimeLibs;
  guiRuntimeRpathFlags =
    lib.concatMapStringsSep " " (
      package: "-C link-arg=-Wl,-rpath,${lib.getLib package}/lib"
    )
    guiRuntimeLibs;
}
