//! The icons the window draws: the kit's default set, plus the few toolbar
//! icons it leaves out.

use std::borrow::Cow;

use gpui_kit::component::Icon;
use gpui_kit::*;

gpui_kit::assets::icon_assets!(
    ToolbarIcons,
    [
        SquareStop,
        MicOff,
        Sparkles,
        Zap,
        Pencil,
        Trash,
        CircleDashed
    ]
);

/// The kit's assets, with the toolbar icons first.
pub struct AppAssets;

impl AssetSource for AppAssets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        match ToolbarIcons.load(path)? {
            Some(bytes) => Ok(Some(bytes)),
            None => assets::Assets.load(path),
        }
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        let mut paths = ToolbarIcons.list(path)?;
        paths.extend(assets::Assets.list(path)?);
        Ok(paths)
    }
}

/// An icon the kit's `IconName` lacks, from the set above.
pub fn extra(name: &'static str) -> Icon {
    Icon::default().path(format!("icons/{name}.svg"))
}
