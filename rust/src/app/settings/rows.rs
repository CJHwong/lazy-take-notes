//! What the Models page rows and the settings questions show. Each front end
//! draws them its own way; the words and the rules live here, so they agree.

use crate::adapters::model_catalog::{asset_for_model, expand_model_alias};
use crate::app::download_progress::{describe, megabytes};
use crate::app::settings::ModelRow;
use crate::app::settings::form::model_uri;
use crate::domain::model_asset::{AssetStatus, ModelAsset, RemovalEstimate};

/// A button on a speech model row. Its id is `<action>-<asset id>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowAction {
    Download,
    Remove,
    Cancel,
}

impl RowAction {
    pub fn id(self) -> &'static str {
        match self {
            Self::Download => "download",
            Self::Remove => "remove",
            Self::Cancel => "cancel",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Download => "Download",
            Self::Remove => "Remove",
            Self::Cancel => "Cancel",
        }
    }
}

pub fn status_text(status: AssetStatus) -> &'static str {
    match status {
        AssetStatus::Ready => "Ready",
        AssetStatus::Missing => "Missing",
        AssetStatus::Downloading | AssetStatus::Busy => "Downloading",
        AssetStatus::Corrupt => "Damaged",
        AssetStatus::Unsupported => "Unsupported",
    }
}

/// The row's status word: a cancelled download with bytes kept reads "Paused".
pub fn row_status(row: &ModelRow) -> &'static str {
    match row.paused {
        true => "Paused",
        false => status_text(row.status),
    }
}

/// The theme color of a status, if any.
pub fn status_color(status: AssetStatus) -> Option<&'static str> {
    match status {
        AssetStatus::Ready => Some("success"),
        AssetStatus::Missing => Some("warning"),
        AssetStatus::Corrupt => Some("error"),
        AssetStatus::Downloading | AssetStatus::Busy => Some("accent"),
        AssetStatus::Unsupported => None,
    }
}

/// The buttons a row with `status` shows.
pub fn row_actions(status: AssetStatus) -> Vec<RowAction> {
    let mut actions = Vec::new();
    if matches!(
        status,
        AssetStatus::Missing | AssetStatus::Corrupt | AssetStatus::Busy
    ) {
        actions.push(RowAction::Download);
    }
    if matches!(status, AssetStatus::Ready | AssetStatus::Corrupt) {
        actions.push(RowAction::Remove);
    }
    if status == AssetStatus::Downloading {
        actions.push(RowAction::Cancel);
    }
    actions
}

/// The line under a row: the kept bytes, a download elsewhere, or the progress.
pub fn row_detail(row: &ModelRow) -> String {
    if row.paused {
        format!(
            "{} kept. Download continues from there.",
            megabytes(row.kept)
        )
    } else if row.status == AssetStatus::Busy {
        "In another window. Press Download to follow it.".into()
    } else {
        row.progress.as_ref().map(describe).unwrap_or_default()
    }
}

/// The model choices of a language override row, and the value it starts at.
/// A saved alias reads as its catalog URI; an unknown model joins the choices.
pub fn language_choices(assets: &[ModelAsset], uri: &str) -> (Vec<(String, String)>, String) {
    let uri = match uri.is_empty() {
        true => String::new(),
        false => asset_for_model(&expand_model_alias(uri), assets)
            .map(model_uri)
            .unwrap_or_else(|| uri.to_string()),
    };
    let mut choices: Vec<(String, String)> = assets
        .iter()
        .map(|asset| (asset.label.to_string(), model_uri(asset)))
        .collect();
    if !uri.is_empty() && !choices.iter().any(|(_, value)| *value == uri) {
        choices.push((uri.clone(), uri.clone()));
    }
    let value = match uri.is_empty() {
        true => choices.first().map(|(_, v)| v.clone()).unwrap_or_default(),
        false => uri,
    };
    (choices, value)
}

/// A select row's options: `value` joins them when it is not one of them, so
/// the select can show it and Save writes it back unchanged.
pub fn select_options(options: &[&str], value: &str) -> Vec<String> {
    let mut names: Vec<String> = options.iter().map(|name| name.to_string()).collect();
    if !names.iter().any(|name| name == value) {
        names.push(value.into());
    }
    names
}

/// The question before Settings closes with changes not saved.
pub const DISCARD_TITLE: &str = "Unsaved changes";
pub const DISCARD_BODY: &str = "Your changes to Settings have not been saved. Leave anyway?";
pub const DISCARD_CANCEL: &str = "Keep editing";
pub const DISCARD_CONFIRM: &str = "Discard and exit";

/// The question before a Remove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoveQuestion {
    pub title: String,
    pub body: String,
    /// The delete-legacy choice, when an older copy exists.
    pub checkbox: Option<String>,
}

fn whole_megabytes(size: u64) -> String {
    format!("{:.0} MB", size as f64 / (1024.0 * 1024.0))
}

fn home_relative(path: &str) -> String {
    let Some(home) = dirs::home_dir() else {
        return path.to_string();
    };
    let home = home.to_string_lossy().to_string();
    if path == home || path.starts_with(&format!("{home}/")) {
        return format!("~{}", &path[home.len()..]);
    }
    path.to_string()
}

pub fn remove_question(label: &str, estimate: &RemovalEstimate) -> RemoveQuestion {
    let mut body = format!("This frees {}.", whole_megabytes(estimate.frees));
    let checkbox = estimate.legacy_path.as_ref().map(|legacy| {
        let extra = estimate.frees_with_legacy.saturating_sub(estimate.frees);
        body += &format!(
            " An older copy in {} uses {} more.",
            home_relative(legacy),
            whole_megabytes(extra)
        );
        format!(
            "Also delete it (frees {} in total)",
            whole_megabytes(estimate.frees_with_legacy)
        )
    });
    body += " It downloads again if you need it.";
    RemoveQuestion {
        title: format!("Remove {label}?"),
        body,
        checkbox,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1024 * 1024;

    fn row(status: AssetStatus) -> ModelRow {
        ModelRow {
            status,
            progress: None,
            kept: 0,
            paused: false,
        }
    }

    #[test]
    fn each_status_gets_its_buttons() {
        use RowAction::*;
        assert_eq!(row_actions(AssetStatus::Missing), vec![Download]);
        assert_eq!(row_actions(AssetStatus::Corrupt), vec![Download, Remove]);
        assert_eq!(row_actions(AssetStatus::Busy), vec![Download]);
        assert_eq!(row_actions(AssetStatus::Ready), vec![Remove]);
        assert_eq!(row_actions(AssetStatus::Downloading), vec![Cancel]);
        assert!(row_actions(AssetStatus::Unsupported).is_empty());
    }

    #[test]
    fn a_paused_row_reads_paused_and_names_the_kept_bytes() {
        let mut paused = row(AssetStatus::Missing);
        paused.paused = true;
        paused.kept = 5 * MB;
        assert_eq!(row_status(&paused), "Paused");
        assert!(row_detail(&paused).ends_with("kept. Download continues from there."));
        assert_eq!(row_status(&row(AssetStatus::Corrupt)), "Damaged");
        assert_eq!(row_detail(&row(AssetStatus::Ready)), "");
    }

    #[test]
    fn a_value_outside_the_options_joins_them() {
        assert_eq!(select_options(&["a", "b"], "b"), vec!["a", "b"]);
        assert_eq!(select_options(&["a"], "plugin"), vec!["a", "plugin"]);
    }

    #[test]
    fn an_unknown_language_model_joins_the_choices() {
        let (choices, value) = language_choices(&[], "/models/own.bin");
        assert_eq!(
            choices,
            vec![("/models/own.bin".into(), "/models/own.bin".into())]
        );
        assert_eq!(value, "/models/own.bin");
        assert_eq!(language_choices(&[], ""), (Vec::new(), String::new()));
    }

    #[test]
    fn the_remove_question_offers_the_older_copy_only_when_there_is_one() {
        let alone = RemovalEstimate {
            frees: 10 * MB,
            frees_with_legacy: 10 * MB,
            legacy_path: None,
        };
        let question = remove_question("Model", &alone);
        assert_eq!(question.title, "Remove Model?");
        assert_eq!(
            question.body,
            "This frees 10 MB. It downloads again if you need it."
        );
        assert_eq!(question.checkbox, None);
        let older = RemovalEstimate {
            frees: 10 * MB,
            frees_with_legacy: 30 * MB,
            legacy_path: Some("/old/copy".into()),
        };
        let question = remove_question("Model", &older);
        assert_eq!(
            question.body,
            "This frees 10 MB. An older copy in /old/copy uses 20 MB more. It downloads again if you need it."
        );
        assert_eq!(
            question.checkbox.as_deref(),
            Some("Also delete it (frees 30 MB in total)")
        );
    }
}
