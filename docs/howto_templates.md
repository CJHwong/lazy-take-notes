# Create and edit templates

A template sets the AI instructions, the labels, and the quick-action keys for one kind of session. This guide makes your own template, or changes a built-in one.

## Create a template with AI

1. Run `take-note create-template`. In the template picker, `n` does the same.
2. Describe the session, for example "a weekly 1:1 with my manager, in English".
3. Review the template the AI writes. Ask for changes until it fits.
4. Save it. It appears in the template picker.

## Change a template with AI

1. In the template picker, highlight the template and press `a`.
2. Describe the change. The app asks you to confirm it, then lists the fields that changed.
3. Save. The template keeps its name.

Saving a changed built-in template saves your own copy, which then replaces the built-in in the picker. To keep both, change the file name and the display name in the save dialog.

## Edit the file by hand

1. In the template picker, highlight the template and press `e`. It opens in your editor.
2. Change the fields. The [template schema](ref_template_schema.md) lists every field and variable.

## Add a template file yourself

Put a `.yaml` file in the `templates/` folder of your config folder:

| System | Folder |
|---|---|
| macOS | `~/Library/Application Support/lazy-take-notes/templates/` |
| Linux | `~/.config/lazy-take-notes/templates/` |

The file name, without `.yaml`, is the template's key. A file with the same name as a built-in template replaces it.

## Hide the built-in templates

In `take-note config`, open the **Sessions** page and turn off the built-in templates. Only your own templates are then in the picker.
