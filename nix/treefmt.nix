{
  harbor-rs,
  rustfmtPackage,
}: {...}: {
  imports = [
    harbor-rs.inputs.harbor-meta.treefmtModules.nix
    harbor-rs.inputs.harbor-meta.treefmtModules.toml
    harbor-rs.treefmtModules.rust
  ];
  projectRootFile = "flake.nix";
  programs.rustfmt = {
    edition = "2024";
    package = rustfmtPackage;
  };
  # Pin the project's dependency-ordering policy for both local and fleet runs.
  programs.taplo.settings = builtins.fromTOML (builtins.readFile ../taplo.toml);
  # The pinned harbor-py predates its Python language module. Ruff reads this
  # project's existing pyproject.toml; no Python fixer is part of formatting.
  programs.ruff-format.enable = true;
}
