//! The plugin list: `plugins.yaml` (the record) and `plugins.txt` (one spec per
//! line, what the host launch reads). Both are Python's files and formats.

use std::path::{Path, PathBuf};

/// The package name in a pip or uvx spec, as Python's `parse_spec_name`:
/// `ltn-youtube @ git+https://...` and `ltn-youtube>=1.0` are both `ltn-youtube`.
pub fn parse_spec_name(spec: &str) -> String {
    let spec = spec.trim();
    for separator in [" @", ">=", "<=", "==", "!=", "~=", "<", ">"] {
        if let Some((name, _)) = spec.split_once(separator) {
            return name.trim().to_string();
        }
    }
    spec.to_string()
}

pub struct Manifest {
    yaml: PathBuf,
    txt: PathBuf,
}

impl Manifest {
    pub fn in_dir(config_dir: &Path) -> Self {
        Self {
            yaml: config_dir.join("plugins.yaml"),
            txt: config_dir.join("plugins.txt"),
        }
    }

    /// The specs in `plugins.yaml`. A missing file is no plugins.
    pub fn load(&self) -> Result<Vec<String>, String> {
        let text = match std::fs::read_to_string(&self.yaml) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("Cannot read {}: {e}", self.yaml.display())),
        };
        let value: yaml_serde::Value = yaml_serde::from_str(&text)
            .map_err(|e| format!("Cannot parse {}: {e}", self.yaml.display()))?;
        let Some(list) = value.get("plugins") else {
            return Ok(Vec::new());
        };
        let Some(list) = list.as_sequence() else {
            return Err(format!(
                "{}: `plugins` must be a list of specs",
                self.yaml.display()
            ));
        };
        Ok(list
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect())
    }

    /// The lines the host launch passes to `uvx --with`. A missing file is none.
    pub fn launch_specs(&self) -> Vec<String> {
        std::fs::read_to_string(&self.txt)
            .map(|text| {
                text.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Write both files, `plugins.txt` first, as Python's `save_plugins`.
    pub fn save(&self, specs: &[String]) -> Result<(), String> {
        if let Some(dir) = self.yaml.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("Cannot create {}: {e}", dir.display()))?;
        }
        let txt = match specs.is_empty() {
            true => String::new(),
            false => format!("{}\n", specs.join("\n")),
        };
        std::fs::write(&self.txt, txt)
            .map_err(|e| format!("Cannot write {}: {e}", self.txt.display()))?;
        let mut yaml = String::from("plugins:");
        if specs.is_empty() {
            yaml.push_str(" []\n");
        } else {
            yaml.push('\n');
            for spec in specs {
                let quoted = yaml_serde::to_string(spec).map_err(|e| e.to_string())?;
                yaml.push_str(&format!("- {}\n", quoted.trim_end()));
            }
        }
        std::fs::write(&self.yaml, yaml)
            .map_err(|e| format!("Cannot write {}: {e}", self.yaml.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_name_drops_the_source_and_the_version() {
        assert_eq!(parse_spec_name("ltn-youtube"), "ltn-youtube");
        assert_eq!(
            parse_spec_name("ltn-youtube @ git+https://github.com/x/ltn-youtube.git"),
            "ltn-youtube"
        );
        assert_eq!(parse_spec_name("ltn-youtube>=1.0"), "ltn-youtube");
        assert_eq!(parse_spec_name("  ltn-youtube==2 "), "ltn-youtube");
    }

    #[test]
    fn saved_specs_load_back_and_feed_the_launch() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = Manifest::in_dir(dir.path());
        assert!(manifest.load().unwrap().is_empty());
        assert!(manifest.launch_specs().is_empty());
        let specs = vec![
            "ltn-claude-code @ git+https://github.com/CJHwong/ltn-claude-code.git".to_string(),
            "ltn-youtube>=1.0".to_string(),
        ];
        manifest.save(&specs).unwrap();
        assert_eq!(manifest.load().unwrap(), specs);
        assert_eq!(manifest.launch_specs(), specs);
        manifest.save(&[]).unwrap();
        assert!(manifest.load().unwrap().is_empty());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("plugins.txt")).unwrap(),
            ""
        );
    }

    #[test]
    fn a_broken_plugins_yaml_says_which_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("plugins.yaml"), "plugins: {a: 1}\n").unwrap();
        let error = Manifest::in_dir(dir.path()).load().unwrap_err();
        assert!(error.contains("plugins.yaml"), "{error}");
    }
}
