//! Loads, saves and deletes session templates: built-in ones compiled in, user ones on disk.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use yaml_serde::Value;

use super::paths::Paths;
use super::yaml_io::to_yaml;
use crate::domain::template::{SessionTemplate, TemplateMetadata};
use crate::ports::template_loader::TemplateLoader;

macro_rules! builtins {
    ($($name:literal),* $(,)?) => {
        /// (key, YAML) of every built-in template, shared with the Python version's files.
        pub const BUILTIN_TEMPLATES: &[(&str, &str)] = &[
            $(($name, include_str!(concat!("../../../src/lazy_take_notes/templates/", $name, ".yaml"))),)*
        ];
    };
}

builtins!(
    "daily_standup_en",
    "daily_standup_zh_tw",
    "default_en",
    "default_zh_min_nan",
    "default_zh_tw",
    "lecture_notes_en",
    "lecture_notes_zh_tw",
    "podcast_shownotes_en",
    "podcast_shownotes_zh_tw",
    "sprint_retro_en",
    "sprint_retro_zh_tw",
);

#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("Template not found: '{name}'. Available templates: {available}")]
    NotFound { name: String, available: String },
    #[error("'{0}' is not a user template")]
    NotUserTemplate(String),
    #[error("Could not read template {path}: {message}")]
    Invalid { path: String, message: String },
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Parse template YAML. An empty file is an empty template.
pub fn parse_template(text: &str, origin: &str) -> Result<SessionTemplate, TemplateError> {
    let invalid = |message: String| TemplateError::Invalid {
        path: origin.into(),
        message,
    };
    let value: Value = yaml_serde::from_str(text).map_err(|e| invalid(e.to_string()))?;
    if value.is_null() {
        return Ok(SessionTemplate::default());
    }
    let template: SessionTemplate =
        yaml_serde::from_value(value).map_err(|e| invalid(e.to_string()))?;
    template.check().map_err(|e| invalid(e.to_string()))
}

pub struct YamlTemplates {
    user_dir: PathBuf,
}

impl YamlTemplates {
    pub fn new(paths: &Paths) -> Self {
        Self {
            user_dir: paths.user_templates_dir(),
        }
    }

    pub fn builtin_names(&self) -> BTreeSet<String> {
        BUILTIN_TEMPLATES
            .iter()
            .map(|(name, _)| name.to_string())
            .collect()
    }

    pub fn user_names(&self) -> BTreeSet<String> {
        let Ok(entries) = fs::read_dir(&self.user_dir) else {
            return BTreeSet::new();
        };
        entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                entry
                    .file_name()
                    .to_str()?
                    .strip_suffix(".yaml")
                    .map(str::to_string)
            })
            .collect()
    }

    pub fn all_names(&self) -> BTreeSet<String> {
        self.builtin_names()
            .union(&self.user_names())
            .cloned()
            .collect()
    }

    fn user_path(&self, name: &str) -> PathBuf {
        self.user_dir.join(format!("{name}.yaml"))
    }

    fn load_builtin(&self, name: &str) -> Option<Result<SessionTemplate, TemplateError>> {
        let (_, text) = BUILTIN_TEMPLATES.iter().find(|(key, _)| *key == name)?;
        Some(parse_template(text, name).map(|template| with_key(template, name)))
    }

    fn load_user(&self, name: &str) -> Result<SessionTemplate, TemplateError> {
        let path = self.user_path(name);
        let text = fs::read_to_string(&path)?;
        parse_template(&text, &path.display().to_string()).map(|template| with_key(template, name))
    }

    /// A user template overrides the built-in one with the same key.
    fn load_key(
        &self,
        name: &str,
        user_names: &BTreeSet<String>,
    ) -> Option<Result<SessionTemplate, TemplateError>> {
        if user_names.contains(name) {
            return Some(self.load_user(name));
        }
        self.load_builtin(name)
    }

    /// Load by file path, then key (user before built-in), then display name.
    pub fn load_template(&self, template_ref: &str) -> Result<SessionTemplate, TemplateError> {
        let path = Path::new(template_ref);
        if path.is_file() {
            return parse_template(&fs::read_to_string(path)?, template_ref);
        }
        let user_names = self.user_names();
        if let Some(loaded) = self.load_key(template_ref, &user_names) {
            return loaded;
        }
        for key in self.all_names() {
            if let Some(Ok(template)) = self.load_key(&key, &user_names)
                && template.metadata.name == template_ref
            {
                return Ok(template);
            }
        }
        let available: Vec<String> = self.all_names().into_iter().collect();
        Err(TemplateError::NotFound {
            name: template_ref.into(),
            available: available.join(", "),
        })
    }

    /// The path of the user copy of `name`, copying the built-in first if there is none.
    pub fn ensure_user_copy(&self, name: &str) -> Result<PathBuf, TemplateError> {
        let path = self.user_path(name);
        if self.user_names().contains(name) {
            return Ok(path);
        }
        let Some((_, text)) = BUILTIN_TEMPLATES.iter().find(|(key, _)| *key == name) else {
            return Err(TemplateError::NotFound {
                name: name.into(),
                available: String::new(),
            });
        };
        fs::create_dir_all(&self.user_dir)?;
        fs::write(&path, text)?;
        Ok(path)
    }

    pub fn delete_user_template(&self, name: &str) -> Result<(), TemplateError> {
        if !self.user_names().contains(name) {
            return Err(TemplateError::NotUserTemplate(name.into()));
        }
        Ok(fs::remove_file(self.user_path(name))?)
    }

    /// Write `template` as a user template. The key is the file name, so it is not stored.
    pub fn save_user_template(
        &self,
        template: &SessionTemplate,
        name: &str,
    ) -> io::Result<PathBuf> {
        fs::create_dir_all(&self.user_dir)?;
        let mut value = yaml_serde::to_value(template).map_err(io::Error::other)?;
        if let Some(metadata) = value.get_mut("metadata").and_then(Value::as_mapping_mut) {
            metadata.shift_remove("key");
        }
        let path = self.user_path(name);
        fs::write(&path, to_yaml(&value))?;
        Ok(path)
    }
}

fn with_key(mut template: SessionTemplate, key: &str) -> SessionTemplate {
    template.metadata.key = key.into();
    template
}

impl TemplateLoader for YamlTemplates {
    fn load(&self, template_ref: &str) -> Result<SessionTemplate, String> {
        self.load_template(template_ref).map_err(|e| e.to_string())
    }

    /// Built-ins first, then user templates over them, sorted by key. A file that
    /// fails to parse is left out of the list; loading it reports the error.
    fn list_templates(&self) -> Vec<TemplateMetadata> {
        let user_names = self.user_names();
        self.all_names()
            .iter()
            .filter_map(|key| self.load_key(key, &user_names)?.ok())
            .map(|template| template.metadata)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (tempfile::TempDir, YamlTemplates) {
        let dir = tempfile::tempdir().unwrap();
        let templates = YamlTemplates::new(&Paths::under(dir.path()));
        (dir, templates)
    }

    #[test]
    fn every_built_in_file_is_compiled_in_and_parses() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../src/lazy_take_notes/templates"
        );
        let on_disk: BTreeSet<String> = fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| {
                e.unwrap()
                    .file_name()
                    .to_str()?
                    .strip_suffix(".yaml")
                    .map(str::to_string)
            })
            .collect();
        let (_dir, templates) = setup();
        assert_eq!(templates.builtin_names(), on_disk);
        for name in on_disk {
            let template = templates.load_template(&name).unwrap();
            assert_eq!(template.metadata.key, name);
            assert!(!template.system_prompt.is_empty());
        }
    }

    #[test]
    fn a_user_template_overrides_the_built_in_and_loads_by_display_name() {
        let (_dir, templates) = setup();
        let path = templates.ensure_user_copy("default_en").unwrap();
        fs::write(&path, "metadata:\n  name: Mine\nsystem_prompt: S\n").unwrap();
        assert_eq!(
            templates.load_template("default_en").unwrap().metadata.name,
            "Mine"
        );
        assert_eq!(
            templates.load_template("Mine").unwrap().metadata.key,
            "default_en"
        );
        let listed = templates.list_templates();
        assert_eq!(listed.len(), 11);
        assert!(listed.iter().any(|m| m.name == "Mine"));
    }

    #[test]
    fn an_unknown_template_names_the_available_ones() {
        let (_dir, templates) = setup();
        let error = templates.load_template("nope").unwrap_err().to_string();
        assert!(
            error
                .starts_with("Template not found: 'nope'. Available templates: daily_standup_en, ")
        );
    }

    #[test]
    fn save_then_load_round_trips_and_drops_the_key() {
        let (_dir, templates) = setup();
        let mut template = templates.load_template("default_zh_tw").unwrap();
        template.recognition_hints = vec!["No".into(), "on".into(), "貨代".into()];
        let path = templates.save_user_template(&template, "copy").unwrap();
        assert!(!fs::read_to_string(&path).unwrap().contains("key:"));
        let back = templates.load_template("copy").unwrap();
        assert_eq!(
            back,
            SessionTemplate {
                metadata: TemplateMetadata {
                    key: "copy".into(),
                    ..template.metadata.clone()
                },
                ..template
            }
        );
        templates.delete_user_template("copy").unwrap();
        assert!(matches!(
            templates.delete_user_template("copy"),
            Err(TemplateError::NotUserTemplate(_))
        ));
    }

    #[test]
    fn an_empty_template_file_is_an_empty_template() {
        assert_eq!(parse_template("", "x").unwrap(), SessionTemplate::default());
    }
}
