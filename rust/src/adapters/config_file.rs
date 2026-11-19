//! Reads and writes `config.yaml` as raw YAML, so keys this version does not know survive.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use yaml_serde::{Mapping, Value};

use super::paths::Paths;
use super::yaml_io::to_yaml;

#[derive(Debug, thiserror::Error)]
pub enum ConfigFileError {
    #[error("Config file not found: {0}")]
    NotFound(PathBuf),
    #[error("Could not read {path}: {source}")]
    Read { path: PathBuf, source: io::Error },
    #[error("Could not parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: yaml_serde::Error,
    },
}

/// The raw config: the `--config` file, else the first default file that exists, else empty.
/// `overrides` merge on top.
pub fn load_raw(
    paths: &Paths,
    config_path: Option<&Path>,
    overrides: Option<&Value>,
) -> Result<Value, ConfigFileError> {
    let mut data = match config_path {
        Some(path) if !path.exists() => return Err(ConfigFileError::NotFound(path.to_path_buf())),
        Some(path) => read_mapping(path)?,
        None => match paths
            .default_config_paths()
            .into_iter()
            .find(|path| path.exists())
        {
            Some(path) => read_mapping(&path)?,
            None => Value::Mapping(Mapping::new()),
        },
    };
    if let Some(overrides) = overrides {
        deep_merge(&mut data, overrides);
    }
    Ok(data)
}

/// A missing, empty, or non-mapping file reads as an empty mapping, as in the Python version.
fn read_mapping(path: &Path) -> Result<Value, ConfigFileError> {
    let text = fs::read_to_string(path).map_err(|source| ConfigFileError::Read {
        path: path.into(),
        source,
    })?;
    let value: Value = yaml_serde::from_str(&text).map_err(|source| ConfigFileError::Parse {
        path: path.into(),
        source,
    })?;
    Ok(if value.is_mapping() {
        value
    } else {
        Value::Mapping(Mapping::new())
    })
}

/// Merge `overrides` into `base`. Nested mappings merge; anything else replaces.
pub fn deep_merge(base: &mut Value, overrides: &Value) {
    let (Some(base_map), Some(override_map)) = (base.as_mapping_mut(), overrides.as_mapping())
    else {
        *base = overrides.clone();
        return;
    };
    for (key, value) in override_map {
        match base_map.get_mut(key) {
            Some(existing) if existing.is_mapping() && value.is_mapping() => {
                deep_merge(existing, value)
            }
            _ => {
                base_map.insert(key.clone(), value.clone());
            }
        }
    }
}

/// The file Save writes: the first default file that exists, else `config.yaml`.
pub fn config_file_path(paths: &Paths) -> PathBuf {
    let [yaml, yml] = paths.default_config_paths();
    if !yaml.exists() && yml.exists() {
        yml
    } else {
        yaml
    }
}

/// Merge `data` into the config file (or create it) and write it back with mode 0600.
///
/// `path` is the `--config` file; without it the default location is used. Each key
/// path in `owned` belongs to the settings form: its value in `data` replaces the
/// file's value, and a key missing from `data` is deleted. A merge alone would keep
/// a language override or a limit the user removed. Every other key, such as a
/// plugin's settings, merges as before.
pub fn write_config(
    paths: &Paths,
    data: &Value,
    path: Option<&Path>,
    owned: &[&[&str]],
) -> io::Result<PathBuf> {
    let path = match path {
        Some(path) => path.to_path_buf(),
        None => {
            fs::create_dir_all(&paths.config_dir)?;
            config_file_path(paths)
        }
    };
    // A file that does not parse is an error, never an empty config: Save must not wipe it.
    let mut existing = if path.exists() {
        read_mapping(&path)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?
    } else {
        Value::Mapping(Mapping::new())
    };
    deep_merge(&mut existing, data);
    for keys in owned {
        replace_owned(&mut existing, data, keys);
    }
    // The file can hold an API key, so only the user may read it. Writing in place
    // keeps a symlinked config.yaml (dotfiles) a symlink.
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&path)?;
    file.write_all(to_yaml(&existing).as_bytes())?;
    // A file from an older version was created readable by everyone.
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    Ok(path)
}

fn replace_owned(existing: &mut Value, data: &Value, keys: &[&str]) {
    let Some((last, parents)) = keys.split_last() else {
        return;
    };
    let mut target = existing;
    let mut source = Some(data);
    for key in parents {
        source = source.and_then(|s| s.get(*key));
        match target.get_mut(*key) {
            Some(next) if next.is_mapping() => target = next,
            _ => return,
        }
    }
    let Some(target) = target.as_mapping_mut() else {
        return;
    };
    match source.and_then(|s| s.get(*last)) {
        Some(value) => {
            target.insert(Value::String((*last).into()), value.clone());
        }
        None => {
            target.shift_remove(*last);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(text: &str) -> Value {
        yaml_serde::from_str(text).unwrap()
    }

    fn setup() -> (tempfile::TempDir, Paths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::under(dir.path());
        (dir, paths)
    }

    #[test]
    fn deep_merge_merges_mappings_and_replaces_the_rest() {
        let mut base = yaml("a:\n  b: 1\n  c: 2\nlist: [1, 2]\n");
        deep_merge(&mut base, &yaml("a:\n  c: 3\n  d: 4\nlist: [9]\n"));
        assert_eq!(base, yaml("a:\n  b: 1\n  c: 3\n  d: 4\nlist: [9]\n"));
    }

    #[test]
    fn load_reads_yaml_then_yml_and_applies_overrides() {
        let (_dir, paths) = setup();
        assert_eq!(load_raw(&paths, None, None).unwrap(), yaml("{}"));
        fs::create_dir_all(&paths.config_dir).unwrap();
        fs::write(paths.config_dir.join("config.yml"), "digest:\n  model: a\n").unwrap();
        let loaded = load_raw(&paths, None, Some(&yaml("digest:\n  min_lines: 3\n"))).unwrap();
        assert_eq!(loaded, yaml("digest:\n  model: a\n  min_lines: 3\n"));
        fs::write(paths.config_dir.join("config.yaml"), "").unwrap();
        assert_eq!(load_raw(&paths, None, None).unwrap(), yaml("{}"));
    }

    #[test]
    fn an_explicit_missing_file_is_an_error() {
        let (dir, paths) = setup();
        let missing = dir.path().join("nope.yaml");
        assert!(matches!(
            load_raw(&paths, Some(&missing), None),
            Err(ConfigFileError::NotFound(_))
        ));
    }

    #[test]
    fn write_merges_keeps_plugin_keys_and_replaces_owned_keys() {
        let (_dir, paths) = setup();
        fs::create_dir_all(&paths.config_dir).unwrap();
        let file = paths.config_dir.join("config.yaml");
        fs::write(&file, "my_plugin:\n  key: v\ntranscription:\n  models:\n    ja: x\n    zh: y\ndigest:\n  max_lines: 9\n").unwrap();
        let data = yaml("transcription:\n  models:\n    zh: z\ndigest:\n  model: m\n");
        let owned: &[&[&str]] = &[&["transcription", "models"], &["digest", "max_lines"]];
        let written = write_config(&paths, &data, None, owned).unwrap();
        let back = load_raw(&paths, None, None).unwrap();
        assert_eq!(written, file);
        assert_eq!(
            back,
            yaml(
                "my_plugin:\n  key: v\ntranscription:\n  models:\n    zh: z\ndigest:\n  model: m\n"
            )
        );
        assert_eq!(
            fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn write_refuses_to_replace_a_file_it_cannot_parse() {
        let (_dir, paths) = setup();
        fs::create_dir_all(&paths.config_dir).unwrap();
        let file = paths.config_dir.join("config.yaml");
        fs::write(&file, "a: [unclosed\n").unwrap();
        assert!(write_config(&paths, &yaml("theme: y\n"), None, &[]).is_err());
        assert_eq!(fs::read_to_string(file).unwrap(), "a: [unclosed\n");
    }

    #[test]
    fn write_keeps_a_symlink() {
        let (dir, paths) = setup();
        fs::create_dir_all(&paths.config_dir).unwrap();
        let target = dir.path().join("dotfiles.yaml");
        fs::write(&target, "theme: x\n").unwrap();
        std::os::unix::fs::symlink(&target, paths.config_dir.join("config.yaml")).unwrap();
        write_config(&paths, &yaml("theme: dark\n"), None, &[]).unwrap();
        assert!(
            fs::symlink_metadata(paths.config_dir.join("config.yaml"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_to_string(target).unwrap(), "theme: dark\n");
    }
}
