{
  description = "Bekiper — kinematic data analysis workspace (public analysis subset)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    harbor-meta = {
      url = "git+https://github.com/caniko/harbor-meta.git?ref=feat/scoped-dev-runtime&rev=780f01e218d159826e47ae125525f3280954b3e7";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    harbor-rs = {
      url = "git+https://github.com/caniko/harbor-rs.git?ref=feat/scoped-dev-runtime&rev=ad028e0278441392a104dbd0f1126dcc9958c638";
      inputs.harbor-meta.follows = "harbor-meta";
    };

    harbor-py = {
      url = "git+https://github.com/caniko/harbor-py.git?ref=feat/scoped-dev-runtime&rev=685f22e60df4fd9faf3d806f8ded1a8e401823fa";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.harbor-meta.follows = "harbor-meta";
    };

    crane.follows = "harbor-rs/crane";
    treefmt-nix.follows = "harbor-rs/treefmt-nix";

    rust-overlay.follows = "harbor-rs/rust-overlay";

    flake-parts.url = "github:hercules-ci/flake-parts";

    # Pinned to a Hydra-evaluated rev for pkgsRocm cache hits; also sources the
    # LLVM-free mesa-radv so nixpkgs bumps don't rebuild it.
    nixpkgs-rocm.url = "github:NixOS/nixpkgs/289c8a834fbf87af1f01cb78f479537ef5219a0b";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      # The ROCm analysis stack is validated on x86_64 Linux only.
      systems = ["x86_64-linux"];

      perSystem = import ./nix/per-system.nix {inherit inputs;};
    };
}
