//! The saved sessions as a list of dated groups, for a front end's sidebar.
//! The date comes from the folder name, which the session wrote when it
//! started, so a copied or touched folder keeps its place.

use chrono::{NaiveDate, NaiveDateTime};

use crate::adapters::session_reader::SessionInfo;

/// The stamp at the front of every session folder name.
const STAMP: &str = "%Y-%m-%d_%H%M%S";
const STAMP_LEN: usize = "2026-01-02_150405".len();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Today,
    Yesterday,
    PreviousSevenDays,
    Older,
}

impl Group {
    pub const ORDER: [Group; 4] = [
        Group::Today,
        Group::Yesterday,
        Group::PreviousSevenDays,
        Group::Older,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Group::Today => "Today",
            Group::Yesterday => "Yesterday",
            Group::PreviousSevenDays => "Previous 7 Days",
            Group::Older => "Older",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub info: SessionInfo,
    pub group: Group,
    /// The label the user gave, or "Untitled".
    pub title: String,
    /// When the session started: the time for today and yesterday, else the date too.
    pub when: String,
}

fn started(name: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(name.get(..STAMP_LEN)?, STAMP).ok()
}

fn group(day: NaiveDate, today: NaiveDate) -> Group {
    match (today - day).num_days() {
        0 => Group::Today,
        1 => Group::Yesterday,
        2..=7 => Group::PreviousSevenDays,
        // A future stamp (a wrong clock) is not "today" either.
        _ => Group::Older,
    }
}

/// What a session is called: the label the user gave, "Untitled" without
/// one, or the folder name when the app did not name the folder.
pub fn title(name: &str) -> String {
    if started(name).is_none() {
        return name.to_string();
    }
    let label = name[STAMP_LEN..].trim_start_matches('_').replace('_', " ");
    if label.is_empty() {
        "Untitled".into()
    } else {
        label
    }
}

fn entry(info: SessionInfo, today: NaiveDate) -> Entry {
    let title = title(&info.name);
    let Some(start) = started(&info.name) else {
        return Entry {
            title,
            group: Group::Older,
            when: String::new(),
            info,
        };
    };
    let group = group(start.date(), today);
    let when = match group {
        Group::Today | Group::Yesterday => start.format("%H:%M").to_string(),
        _ => start.format("%Y-%m-%d %H:%M").to_string(),
    };
    Entry {
        title,
        group,
        when,
        info,
    }
}

/// `sessions` in groups, Today first; each group keeps the order it got. Empty groups are left out.
pub fn grouped(sessions: Vec<SessionInfo>, today: NaiveDate) -> Vec<(Group, Vec<Entry>)> {
    let entries: Vec<Entry> = sessions.into_iter().map(|s| entry(s, today)).collect();
    Group::ORDER
        .into_iter()
        .map(|g| {
            let members: Vec<Entry> = entries.iter().filter(|e| e.group == g).cloned().collect();
            (g, members)
        })
        .filter(|(_, members)| !members.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn info(name: &str) -> SessionInfo {
        SessionInfo {
            dir: PathBuf::from("/out").join(name),
            name: name.into(),
            has_notes: false,
            has_transcript: true,
        }
    }

    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, 25).unwrap()
    }

    #[test]
    fn each_session_goes_to_the_group_of_its_start_day() {
        let groups = grouped(
            vec![
                info("2026-11-25_093000_standup"),
                info("2026-11-24_150405"),
                info("2026-11-19_080000_retro"),
                info("2026-11-17_080000_old"),
                info("notes"),
            ],
            today(),
        );
        let titles: Vec<(Group, Vec<&str>)> = groups
            .iter()
            .map(|(g, es)| (*g, es.iter().map(|e| e.title.as_str()).collect()))
            .collect();
        assert_eq!(
            titles,
            vec![
                (Group::Today, vec!["standup"]),
                (Group::Yesterday, vec!["Untitled"]),
                (Group::PreviousSevenDays, vec!["retro"]),
                (Group::Older, vec!["old", "notes"]),
            ]
        );
    }

    #[test]
    fn the_time_shows_alone_for_today_and_with_the_date_before() {
        let groups = grouped(
            vec![info("2026-11-25_093000_a_b"), info("2026-11-01_170000")],
            today(),
        );
        let today_entry = &groups[0].1[0];
        assert_eq!(today_entry.title, "a b");
        assert_eq!(today_entry.when, "09:30");
        assert_eq!(groups[1].1[0].when, "2026-11-01 17:00");
    }

    #[test]
    fn a_future_stamp_and_a_bad_stamp_are_older() {
        let groups = grouped(
            vec![info("2026-12-01_000000"), info("2026-13-01_000000_x")],
            today(),
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, Group::Older);
        assert_eq!(groups[0].1[1].title, "2026-13-01_000000_x");
        assert_eq!(groups[0].1[1].when, "");
    }
}
