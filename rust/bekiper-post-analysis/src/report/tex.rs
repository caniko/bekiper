//! TeX rendering and literal text escaping.
use super::*;

/// The report follows the figures' colour mode: `BEKIPER_PLOT_COLOR_MODE`
/// (inherited by the figure renderer too), else dark.
pub(super) fn report_mode(lookup: impl Fn(&str) -> Option<String>) -> Result<themekit::Mode> {
    Ok(themekit::Mode::resolve(
        None,
        lookup("BEKIPER_PLOT_COLOR_MODE").as_deref(),
        None,
    )?)
}

pub(super) fn render_tex(
    title: &str,
    sections: &[Section],
    figures: &[Figure],
    palette: &themekit::Palette,
) -> String {
    let mut output = String::from(
        r"\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{array,longtable,booktabs,graphicx,xcolor,hyperref}
\hypersetup{hidelinks}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\emergencystretch=3em
\pdfvariable suppressoptionalinfo 767\relax
\begin{document}
",
    );
    // Page and text colours match the figures' palette (dark pages in night mode).
    let hex = |color: &str| color.trim_start_matches('#').to_ascii_uppercase();
    output.push_str(&format!(
        "\\definecolor{{pagebg}}{{HTML}}{{{}}}\\definecolor{{pageink}}{{HTML}}{{{}}}\\pagecolor{{pagebg}}\\color{{pageink}}\n",
        hex(palette.background),
        hex(palette.ink)
    ));
    output.push_str(&format!(
        "\\begin{{center}}\\LARGE\\bfseries {}\\end{{center}}\n",
        tex_escape(title)
    ));
    for section in sections {
        output.push_str(&format!("\\section{{{}}}\n", tex_escape(&section.heading)));
        for paragraph in &section.paragraphs {
            output.push_str(&format!("{}\n\n", tex_escape(paragraph)));
        }
        for table in &section.tables {
            let width = 0.90 / table.headers.len() as f64;
            let columns =
                vec![
                    format!(">{{\\raggedright\\arraybackslash}}p{{{width:.4}\\linewidth}}");
                    table.headers.len()
                ]
                .join("");
            output.push_str(&format!(
                "{{\\footnotesize\n\\begin{{longtable}}{{@{{}}{columns}@{{}}}}\n\\toprule\n"
            ));
            let header = table
                .headers
                .iter()
                .map(|value| format!("\\textbf{{{}}}", tex_escape(value)))
                .collect::<Vec<_>>()
                .join(" & ");
            output.push_str(&format!("{header} \\\\\n\\midrule\n\\endfirsthead\n\\toprule\n{header} \\\\\n\\midrule\n\\endhead\n"));
            for row in &table.rows {
                output.push_str(&format!(
                    "{} \\\\\n",
                    row.iter()
                        .map(|value| tex_escape(value))
                        .collect::<Vec<_>>()
                        .join(" & ")
                ));
            }
            output.push_str("\\bottomrule\n\\end{longtable}\n}\n");
        }
    }
    if !figures.is_empty() {
        output.push_str("\\clearpage\n\\section{Selected figures}\n");
    }
    for figure in figures {
        output.push_str(&format!("\\subsection{{{}}}\nStatus: {}.\n\\begin{{center}}\\includegraphics[width=\\linewidth,height=0.72\\textheight,keepaspectratio]{{{}}}\\end{{center}}\n",tex_escape(&figure.name),tex_escape(&figure.status),figure.path));
        for caption in &figure.captions {
            output.push_str(&format!("{}\\par\n", tex_escape(caption)));
        }
        output.push_str("\\clearpage\n");
    }
    output.push_str("\\end{document}\n");
    output
}

pub(super) fn tex_escape(value: &str) -> String {
    let mut output = String::new();
    let mut run = 0;
    let mut previous: Option<char> = None;
    let mut characters = value.chars().peekable();
    while let Some(character) = characters.next() {
        let escaped = match character {
            '\\' => "\\textbackslash{}".into(),
            '{' => "\\{".into(),
            '}' => "\\}".into(),
            '$' => "\\$".into(),
            '&' => "\\&".into(),
            '#' => "\\#".into(),
            '%' => "\\%".into(),
            '_' => "\\_".into(),
            '~' => "\\textasciitilde{}".into(),
            '^' => "\\textasciicircum{}".into(),
            '\n' | '\r' | '\t' => " ".into(),
            value if value.is_control() => " ".into(),
            value => value.to_string(),
        };
        output.push_str(&escaped);
        run = if character.is_alphanumeric() {
            run + 1
        } else {
            0
        };
        if run >= 16 {
            output.push_str("\\allowbreak{}");
            run = 0;
        }
        let next_is_digit = characters.peek().is_some_and(char::is_ascii_digit);
        let numeric_separator = next_is_digit
            && ((character == '.' && previous.is_some_and(|value| value.is_ascii_digit()))
                || (character == '-'
                    && previous.is_none_or(|value| value == 'e' || !value.is_alphanumeric())));
        if !numeric_separator
            && matches!(
                character,
                '_' | '.' | '/' | '-' | ':' | '[' | ']' | ',' | ';' | '=' | '*'
            )
        {
            output.push_str("\\allowbreak{}");
        }
        previous = Some(character);
    }
    output
}

pub(super) fn markdown_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace(['\n', '\r'], " ")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
