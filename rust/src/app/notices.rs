//! The notices each front end shows before a recording, and in the template picker.

/// Python's `ConsentNotice.NOTICE_BODY`.
pub const CONSENT_NOTICE_BODY: &str =
    "This session will **record audio** and generate a transcript.

- **Inform all participants** that recording is active.
- Recording without consent may violate local laws.
- You are responsible for compliance.
";

pub const BUILTIN_NOTICE_BODY: &str =
    "Built-in templates are **example sessions** for common use cases \
(meetings, standups, lectures, podcasts, etc.).

- Press **[n]** in the picker to **create your own** template.
- To hide built-in templates, set `show_builtin_templates: false` \
in settings or config.yaml.
";
