//! The templates a picker offers: which ones show, how they group by locale,
//! and the preview text. Both front ends draw from it.

use std::collections::{BTreeMap, BTreeSet};

use tracing::warn;

use crate::adapters::templates::YamlTemplates;
use crate::domain::template::SessionTemplate;

/// The picker's text when no template shows.
pub const NO_TEMPLATES: &str = "**No templates yet.**\n\nPress n to create one, or turn on Show built-in templates in Settings > Sessions.";

/// One list row: a locale heading, or a template by its key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Row {
    Locale(String),
    Template(String),
}

pub struct TemplateCatalog {
    pub templates: YamlTemplates,
    pub user_names: BTreeSet<String>,
    pub builtin_names: BTreeSet<String>,
    /// The templates that show, by key. A built-in one hides unless
    /// `show_builtins`, or a user template has its name.
    pub loaded: BTreeMap<String, SessionTemplate>,
}

impl TemplateCatalog {
    pub fn new(templates: YamlTemplates, show_builtins: bool) -> Self {
        let user_names = templates.user_names();
        let builtin_names = templates.builtin_names();
        let mut loaded = BTreeMap::new();
        for name in templates.all_names() {
            let hidden =
                !show_builtins && builtin_names.contains(&name) && !user_names.contains(&name);
            if hidden {
                continue;
            }
            match templates.load_template(&name) {
                Ok(template) => {
                    loaded.insert(name, template);
                }
                Err(e) => warn!(target: "ltn.app", "Template {name} does not load: {e}"),
            }
        }
        Self {
            templates,
            user_names,
            builtin_names,
            loaded,
        }
    }

    /// The rows for a lower-case filter: each locale, then its templates
    /// whose key or description holds the filter.
    pub fn rows(&self, query: &str) -> Vec<Row> {
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, template) in &self.loaded {
            let matches = query.is_empty()
                || name.to_lowercase().contains(query)
                || template.metadata.description.to_lowercase().contains(query);
            if matches {
                let locale = template.metadata.locale.clone();
                groups.entry(locale).or_default().push(name.clone());
            }
        }
        let mut rows = Vec::new();
        for (locale, names) in groups {
            rows.push(Row::Locale(locale));
            rows.extend(names.into_iter().map(Row::Template));
        }
        rows
    }

    /// The name a row shows: the template's own name, or its key.
    pub fn label(&self, name: &str) -> String {
        match self.loaded.get(name) {
            Some(template) if !template.metadata.name.is_empty() => template.metadata.name.clone(),
            _ => name.to_string(),
        }
    }

    pub fn preview_markdown(&self, name: &str) -> Option<String> {
        let template = self.loaded.get(name)?;
        let meta = &template.metadata;
        let source = if self.user_names.contains(name) {
            "\\[user]"
        } else {
            "\\[built-in]"
        };
        let mut lines = vec![
            format!("## {}  {source}", meta.name),
            String::new(),
            match meta.description.is_empty() {
                true => String::new(),
                false => format!("> {}", meta.description),
            },
            String::new(),
            format!("**Locale:** `{}`", meta.locale),
        ];
        if !template.quick_actions.is_empty() {
            lines.extend([String::new(), "### Quick Actions".into()]);
            for (index, action) in template.quick_actions.iter().enumerate() {
                lines.push(format!(
                    "- **`{}`** {} \u{2014} {}",
                    index + 1,
                    action.label,
                    action.description
                ));
            }
        }
        if !template.recognition_hints.is_empty() {
            lines.extend([
                String::new(),
                format!(
                    "**Recognition hints:** {}",
                    template.recognition_hints.join(", ")
                ),
            ]);
        }
        lines.extend([
            "".into(),
            "---".into(),
            "".into(),
            "### System Prompt".into(),
            "".into(),
        ]);
        lines.push(template.system_prompt.clone());
        Some(lines.join("\n"))
    }

    /// Read a template again after an edit. Err keeps the old copy: the file does not load.
    pub fn reload(&mut self, name: &str) -> Result<(), String> {
        self.user_names = self.templates.user_names();
        let template = self
            .templates
            .load_template(name)
            .map_err(|e| e.to_string())?;
        self.loaded.insert(name.to_string(), template);
        Ok(())
    }

    /// Delete a user template. A built-in one of the same name shows again.
    pub fn delete(&mut self, name: &str) -> Result<(), String> {
        self.templates
            .delete_user_template(name)
            .map_err(|e| e.to_string())?;
        self.user_names = self.templates.user_names();
        let builtin = self
            .builtin_names
            .contains(name)
            .then(|| self.templates.load_template(name).ok())
            .flatten();
        match builtin {
            Some(template) => {
                self.loaded.insert(name.to_string(), template);
            }
            None => {
                self.loaded.remove(name);
            }
        }
        Ok(())
    }
}
