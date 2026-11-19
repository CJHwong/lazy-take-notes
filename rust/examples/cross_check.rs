//! Reads and writes real files for `scripts/cross_check.py`, which compares the results with Python.

use std::path::{Path, PathBuf};

use lazy_take_notes::adapters::config_file::{load_raw, write_config};
use lazy_take_notes::adapters::paths::Paths;
use lazy_take_notes::adapters::session_reader::list_sessions;
use lazy_take_notes::adapters::templates::{YamlTemplates, parse_template};
use lazy_take_notes::config::{build_app_config, build_infra_config};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| PathBuf::from(&args[i]);
    let paths = Paths::for_user().expect("home folder");
    let out = match args[0].as_str() {
        "config" => {
            let raw = load_raw(&paths, Some(&arg(1)), None).unwrap();
            serde_json::json!({"app": build_app_config(&raw).unwrap(), "infra": build_infra_config(&raw).unwrap()})
        }
        // Rewrite a copy of a config file through the Rust writer.
        "rewrite-config" => {
            write_config(
                &paths,
                &yaml_serde::Value::Mapping(Default::default()),
                Some(&arg(1)),
                &[],
            )
            .unwrap();
            serde_json::json!("ok")
        }
        "template" => serde_json::to_value(read_template(&arg(1))).unwrap(),
        // Save a template through the Rust writer into <dir>/config/templates/<name>.yaml.
        "save-template" => {
            let templates = YamlTemplates::new(&Paths::under(&arg(2)));
            let path = templates
                .save_user_template(&read_template(&arg(1)), &args[3])
                .unwrap();
            serde_json::json!(path)
        }
        "sessions" => {
            let names: Vec<String> = list_sessions(&arg(1)).into_iter().map(|s| s.name).collect();
            serde_json::json!(names)
        }
        other => panic!("unknown command {other}"),
    };
    println!("{out}");
}

fn read_template(path: &Path) -> lazy_take_notes::domain::template::SessionTemplate {
    parse_template(
        &std::fs::read_to_string(path).unwrap(),
        &path.display().to_string(),
    )
    .unwrap()
}
