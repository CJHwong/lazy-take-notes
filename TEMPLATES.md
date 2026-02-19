# Templates

Templates are YAML files that control the LLM prompts, display labels, and quick-action keys for a session. Built-in templates ship with the app; you can add your own or override built-ins.

## User templates

Drop a `.yaml` file in `~/.config/lazy-take-notes/templates/`. It's discovered automatically by filename (without the extension). A user template with the same filename as a built-in overrides it.

```
~/.config/lazy-take-notes/templates/
└── ux_interview.yaml      ← discovered as "ux_interview"
```

## YAML schema

```yaml
metadata:
  name: "UX Interview"          # display name shown in picker and header
  description: "..."            # one-line description shown in picker
  locale: "en-US"               # BCP 47 locale — drives whisper model selection

system_prompt: |
  You are … (sets the LLM's role for the whole session)

digest_user_template: |
  # variables: {line_count}, {new_lines}, {user_context}
  New transcript ({line_count} lines):
  {new_lines}
  {user_context}
  Please update the notes.

final_user_template: |
  # variables: {line_count}, {new_lines}, {user_context}, {full_transcript}
  Session ended. Final transcript ({line_count} lines):
  {new_lines}
  {user_context}
  Full transcript:
  {full_transcript}
  Produce the final summary.

whisper_prompt: "optional hint words for the speech recogniser"

quick_actions:
  - key: "1"                    # single character; see reserved keys below
    label: "Pain Points"        # shown in the keybinding bar
    description: "..."          # shown in the help screen
    prompt_template: |
      # variables: {digest_markdown}, {recent_transcript}
      Current notes:
      {digest_markdown}
      Recent transcript:
      {recent_transcript}
      List every pain point the participant expressed.
```

### Template variables

**`digest_user_template` and `final_user_template`**

| Variable           | Content                                              |
| ------------------ | ---------------------------------------------------- |
| `{line_count}`     | Number of new transcript lines in this batch         |
| `{new_lines}`      | The new transcript lines                             |
| `{user_context}`   | User-typed notes (empty string if none)              |
| `{full_transcript}`| Complete transcript (`final_user_template` only)     |

**`quick_actions[].prompt_template`**

| Variable              | Content                                  |
| --------------------- | ---------------------------------------- |
| `{digest_markdown}`   | Latest digest (Markdown)                 |
| `{recent_transcript}` | Last ~30 transcript lines                |

### Reserved quick-action keys

The following keys are reserved by the app and cannot be used for quick actions:

`q` `s` `h` `c` `space` `tab` `escape`

### `whisper_prompt`

An optional hint string prepended to the speech recogniser's context. Use it to bias recognition towards domain vocabulary (names, acronyms, technical terms) that whisper might otherwise mishear.

### `locale`

BCP 47 locale (e.g. `zh-TW`, `en-US`). Controls:
- Which whisper model is selected (via `transcription.models` in config)
- The language hint passed to the recogniser

If omitted or unrecognised, the default transcription model is used.
