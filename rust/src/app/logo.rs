//! The mascot both front ends draw on the welcome screen: a pig in headphones.

use crate::app::theme::Rgb;

pub const PIG: Rgb = Rgb(0xE8, 0xA8, 0x20);
pub const TEAL: Rgb = Rgb(0x4A, 0x9E, 0x9E);
pub const OUTLINE: Rgb = Rgb(0x2A, 0x18, 0x18);
pub const SNOUT: Rgb = Rgb(0xC4, 0x7A, 0x10);

/// The pixels, one string per row. `.` is the screen background.
pub const PIXELS: [&str; 12] = [
    "..TTTTTTTTTTTTTTTT..",
    ".TT.DDDDDDDDDDDD.TT.",
    "TT.DPPPPPPPPPPPPD.TT",
    "TT.DPDDPPPPPPDDPD.TT",
    "TT.DPPPPPPPPPPPPD.TT",
    "TT.DPPPSSSSSSPPPD.TT",
    ".T.DPPPSDSSDSPPPD.T.",
    "...DPPPSSSSSSPPPD...",
    "...DPPPPPPPPPPPPD...",
    "....DPPPPPPPPPPD....",
    ".....DPPPPPPPPD.....",
    "......DDDDDDDD......",
];

/// The color of one pixel code, or None for the background.
pub fn pixel(code: u8) -> Option<Rgb> {
    match code {
        b'P' => Some(PIG),
        b'T' => Some(TEAL),
        b'D' => Some(OUTLINE),
        b'S' => Some(SNOUT),
        _ => None,
    }
}
