use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=kernels/preprocess.hip");

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let kernel_src = PathBuf::from("kernels/preprocess.hip");
    let kernel_obj = out_dir.join("preprocess.co");

    // Find hipcc — check ROCM_PATH first, then PATH
    let hipcc = env::var("ROCM_PATH")
        .ok()
        .map(|p| PathBuf::from(p).join("bin/hipcc"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("hipcc"));

    let target_arch = env::var("PYTORCH_ROCM_ARCH").unwrap_or_else(|_| "gfx1100".into());

    // Compile .hip → .co (code object) for loading via hipModuleLoadData
    let status = Command::new(&hipcc)
        .args([
            "--genco",
            &format!("--offload-arch={target_arch}"),
            "-O3",
            "-o",
        ])
        .arg(&kernel_obj)
        .arg(&kernel_src)
        .status()
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("failed to run hipcc at {}: {e}", hipcc.display()),
            )
        })?;

    if !status.success() {
        return Err(std::io::Error::other(format!(
            "hipcc compilation failed with exit code: {status}"
        ))
        .into());
    }

    // Embed the compiled code object as a byte array
    let embed_path = out_dir.join("preprocess_kernel.rs");
    std::fs::write(
        &embed_path,
        format!(
            "/// Compiled HIP code object for the preprocessing kernel.\n\
             pub static PREPROCESS_KERNEL_CO: &[u8] = include_bytes!(\"{}\");\n",
            kernel_obj.display().to_string().replace('\\', "/")
        ),
    )
    .map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to write {}: {e}", embed_path.display()),
        )
    })?;

    // Emit rpath for vulkan-loader so dlopen("libvulkan.so.1") works in test
    // binaries without LD_LIBRARY_PATH. Vulkan is loaded at runtime via ash,
    // not linked at build time.
    #[cfg(feature = "vulkan")]
    emit_vulkan_rpath();

    // Link against libva (VA-API) only when the vaapi feature is enabled.
    // HIP and Vulkan are loaded at runtime via libloading/ash respectively.
    #[cfg(feature = "vaapi")]
    link_libva();

    Ok(())
}

/// Emit rpath for vulkan-loader so test binaries can dlopen libvulkan.so.1
/// without requiring LD_LIBRARY_PATH.
#[cfg(feature = "vulkan")]
fn emit_vulkan_rpath() {
    // Try pkg-config first
    if let Ok(output) = Command::new("pkg-config")
        .args(["--libs-only-L", "vulkan"])
        .output()
        && output.status.success()
    {
        let text = String::from_utf8_lossy(&output.stdout);
        for flag in text.split_whitespace() {
            if let Some(dir) = flag.strip_prefix("-L") {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
                return;
            }
        }
    }

    // Search LD_LIBRARY_PATH for libvulkan.so.1
    if let Ok(ld_path) = env::var("LD_LIBRARY_PATH") {
        for dir in ld_path.split(':') {
            if PathBuf::from(dir).join("libvulkan.so.1").exists() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
                return;
            }
        }
    }

    // NixOS: check /run/opengl-driver/lib
    let nixos_path = "/run/opengl-driver/lib";
    if PathBuf::from(nixos_path).join("libvulkan.so.1").exists() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{nixos_path}");
    }
}

#[cfg(feature = "vaapi")]
fn link_libva() {
    // Try pkg-config first, then fall back to searching LD_LIBRARY_PATH and
    // common nix store paths.
    let mut found_va = false;

    if let Ok(output) = std::process::Command::new("pkg-config")
        .args(["--libs-only-L", "libva"])
        .output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout);
        for flag in path.split_whitespace() {
            if let Some(dir) = flag.strip_prefix("-L") {
                println!("cargo:rustc-link-search=native={dir}");
                found_va = true;
            }
        }
    }

    if !found_va {
        // Search LD_LIBRARY_PATH for libva.so
        if let Ok(ld_path) = env::var("LD_LIBRARY_PATH") {
            for dir in ld_path.split(':') {
                if PathBuf::from(dir).join("libva.so").exists() {
                    println!("cargo:rustc-link-search=native={dir}");
                    found_va = true;
                    break;
                }
            }
        }
    }

    if !found_va {
        // Search LIBRARY_PATH and NIX_LDFLAGS for the libva directory
        for var in ["LIBRARY_PATH", "NIX_LDFLAGS"] {
            if let Ok(val) = env::var(var) {
                for token in val.split_whitespace() {
                    let dir = token.strip_prefix("-L").unwrap_or(token);
                    if PathBuf::from(dir).join("libva.so").exists() {
                        println!("cargo:rustc-link-search=native={dir}");
                        found_va = true;
                        break;
                    }
                }
                if found_va {
                    break;
                }
            }
        }
    }

    if !found_va {
        // Last resort: search nix store for libva
        for e in std::fs::read_dir("/nix/store")
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.contains("libva") && !name.contains(".drv") {
                let lib_dir = e.path().join("lib");
                if lib_dir.join("libva.so").exists() {
                    println!("cargo:rustc-link-search=native={}", lib_dir.display());
                    break;
                }
            }
        }
    }

    println!("cargo:rustc-link-lib=dylib=va");
}
