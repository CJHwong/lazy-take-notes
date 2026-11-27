//! Page Up, Page Down, Home and End, over a transcript list or a notes
//! column. The keys move the pane the pointer last moved over, as a
//! trackpad scroll does; before the pointer moves, the transcript.

use gpui_kit::{ListOffset, ListState, Pixels, ScrollHandle, point};

use crate::gui::design::space;

/// Where a scroll key moves the pane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Jump {
    PageUp,
    PageDown,
    Top,
    Bottom,
}

/// The two scrolling panes of a session.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Pane {
    #[default]
    Transcript,
    Notes,
}

/// The share of a page that a page key moves, so the last lines of the old
/// page stay in view at the edge of the new one.
const PAGE_SHARE: f32 = 0.9;

/// Move a transcript list.
pub fn jump_list(list: &ListState, jump: Jump) {
    let page = list.viewport_bounds().size.height * PAGE_SHARE;
    match jump {
        Jump::PageUp => list.scroll_by(-page),
        Jump::PageDown => list.scroll_by(page),
        Jump::Top => list.scroll_to(ListOffset {
            item_ix: 0,
            offset_in_item: space::NONE,
        }),
        Jump::Bottom => list.scroll_to(ListOffset {
            item_ix: list.item_count(),
            offset_in_item: space::NONE,
        }),
    }
}

/// Move a scrolling column that tracks `area`.
pub fn jump_area(area: &ScrollHandle, jump: Jump) {
    let page = area.bounds().size.height * PAGE_SHARE;
    let offset = area.offset();
    // The offset grows negative as the column scrolls down.
    let down = area_target(
        -offset.y.as_f32(),
        page.as_f32(),
        area.max_offset().y.as_f32(),
        jump,
    );
    area.set_offset(point(offset.x, Pixels::from(-down)));
}

/// How far down the column should be after `jump`, from how far down it
/// is now, the page step and the furthest it can go.
fn area_target(down: f32, page: f32, furthest: f32, jump: Jump) -> f32 {
    let target = match jump {
        Jump::PageUp => down - page,
        Jump::PageDown => down + page,
        Jump::Top => 0.0,
        Jump::Bottom => furthest,
    };
    target.clamp(0.0, furthest.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::{Jump, area_target};

    #[test]
    fn a_page_key_moves_one_page_and_stops_at_the_ends() {
        assert_eq!(area_target(50.0, 90.0, 200.0, Jump::PageDown), 140.0);
        assert_eq!(area_target(150.0, 90.0, 200.0, Jump::PageDown), 200.0);
        assert_eq!(area_target(50.0, 90.0, 200.0, Jump::PageUp), 0.0);
        assert_eq!(area_target(50.0, 90.0, 200.0, Jump::Top), 0.0);
        assert_eq!(area_target(50.0, 90.0, 200.0, Jump::Bottom), 200.0);
    }

    #[test]
    fn a_column_shorter_than_its_pane_does_not_move() {
        assert_eq!(area_target(0.0, 90.0, 0.0, Jump::PageDown), 0.0);
        assert_eq!(area_target(0.0, 90.0, 0.0, Jump::Bottom), 0.0);
    }
}
