//! Markdown rendering of the shared report document.
use super::*;

pub(super) fn render_markdown(title: &str, sections: &[Section], figures: &[Record]) -> String {
    let mut output = format!("# {}\n\n", markdown_escape(title));
    for section in sections {
        output.push_str(&format!("## {}\n\n", markdown_escape(&section.heading)));
        for paragraph in &section.paragraphs {
            output.push_str(&format!("{}\n\n", markdown_escape(paragraph)));
        }
        for table in &section.tables {
            output.push_str(&format!(
                "| {} |\n",
                table
                    .headers
                    .iter()
                    .map(|value| markdown_escape(value))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ));
            output.push_str(&format!(
                "| {} |\n",
                vec!["---"; table.headers.len()].join(" | ")
            ));
            for row in &table.rows {
                output.push_str(&format!(
                    "| {} |\n",
                    row.iter()
                        .map(|value| markdown_escape(value))
                        .collect::<Vec<_>>()
                        .join(" | ")
                ));
            }
            output.push('\n');
        }
    }
    output.push_str("## Selected figures\n\n");
    for figure in figures {
        output.push_str(&format!(
            "- {}: {}\n",
            markdown_escape(&text(&figure["name"])),
            markdown_escape(&text(&figure["status"]))
        ));
        for format in figure["formats"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if matches!(format, "svg" | "png" | "pdf") {
                let path =
                    crate::export::relative_path(&format!("{}.{}", text(&figure["name"]), format));
                output.push_str(&format!("  - [{format}]({path})\n"));
            }
        }
    }
    output
}
