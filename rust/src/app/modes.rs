//! The modes the welcome menu of each front end offers.

/// (mode, label, description), in Python's order.
pub const MODES: [(&str, &str, &str); 5] = [
    ("record", "Record", "Live transcription with AI digest"),
    (
        "transcribe",
        "Transcribe",
        "Transcribe an existing audio file",
    ),
    ("view", "View", "Browse previously saved sessions"),
    (
        "create-template",
        "Create Template",
        "Build a custom template with AI",
    ),
    ("config", "Settings", "Edit app configuration"),
];
