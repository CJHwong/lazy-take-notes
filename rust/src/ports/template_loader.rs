use crate::domain::template::{SessionTemplate, TemplateMetadata};

pub trait TemplateLoader {
    /// Load a template by name or file path.
    fn load(&self, template_ref: &str) -> Result<SessionTemplate, String>;

    /// Built-in and user templates.
    fn list_templates(&self) -> Vec<TemplateMetadata>;
}
