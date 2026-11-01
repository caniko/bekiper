//! Compiler invocation and retained subprocess evidence.
use super::*;

pub(super) fn compile(
    directory: &Path,
    compiler: &str,
    runner: &mut dyn FnMut(Command) -> std::io::Result<Output>,
    build: &mut Value,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
) -> Result<()> {
    let tools: &[&str] = if compiler == "latexmk" {
        &["latexmk", "lualatex"]
    } else {
        &["lualatex"]
    };
    for tool in tools {
        let flag = if *tool == "latexmk" {
            "-v"
        } else {
            "--version"
        };
        let output = run_command(directory, tool, &[flag], runner, build, stdout, stderr)?;
        ensure!(output.status.success(), "{tool} version check failed");
        let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        ensure!(!version.is_empty(), "{tool} returned no version identity");
        build["versions"][*tool] = version.into();
    }
    let args: &[&str] = if compiler == "latexmk" {
        &[
            "-norc",
            "-lualatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-no-shell-escape",
            "-outdir=_internal/report",
            "_internal/report/report.tex",
        ]
    } else {
        &[
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-no-shell-escape",
            "-output-directory=_internal/report",
            "_internal/report/report.tex",
        ]
    };
    // latexmk drives all necessary passes. The direct LuaLaTeX route needs a
    // second pass for longtable widths, page references and PDF outlines.
    for _ in 0..if compiler == "latexmk" { 1 } else { 2 } {
        let output = run_command(directory, compiler, args, runner, build, stdout, stderr)?;
        ensure!(
            output.status.success(),
            "{compiler} report compilation failed with status {}",
            output.status
        );
    }
    Ok(())
}

pub(super) fn run_command(
    directory: &Path,
    program: &str,
    args: &[&str],
    runner: &mut dyn FnMut(Command) -> std::io::Result<Output>,
    build: &mut Value,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
) -> Result<Output> {
    build["commands"]
        .as_array_mut()
        .context("report build commands")?
        .push(json!({"program":program,"args":args}));
    let mut command = Command::new(program);
    command
        .current_dir(directory)
        .args(args)
        // ROCm/FFmpeg development paths can inject an incompatible glibc into
        // TeX. Packaged compilers resolve their own runtime via their wrappers.
        .env_remove("LD_LIBRARY_PATH")
        .env("SOURCE_DATE_EPOCH", "0")
        .env("FORCE_SOURCE_DATE", "1")
        // LuaLaTeX reads Unicode/font support using absolute kpathsea paths.
        // TeX's paranoid input policy blocks those legitimate runtime reads.
        // Restricted input still blocks hidden files; all document text is
        // escaped, figure paths are fixed local basenames, and shell escape is
        // disabled independently of the input-file policy.
        .env("openin_any", "r")
        .env("openout_any", "p");
    let result = runner(command);
    match result {
        Ok(output) => {
            stdout.extend_from_slice(&output.stdout);
            stdout.push(b'\n');
            stderr.extend_from_slice(&output.stderr);
            stderr.push(b'\n');
            Ok(output)
        }
        Err(error) => {
            let message = format!("could not execute {program}: {error}");
            stderr.extend_from_slice(message.as_bytes());
            stderr.push(b'\n');
            Err(error).with_context(|| format!("could not execute {program}"))
        }
    }
}
