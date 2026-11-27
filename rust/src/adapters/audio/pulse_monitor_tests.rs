use super::pulse_monitor::{monitor_id, pick_monitor};

#[test]
fn a_monitor_id_appends_monitor_to_the_output_name() {
    assert_eq!(
        monitor_id("alsa_output.pci-0000_00_1f.3.analog-stereo"),
        "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
    );
}

#[test]
fn the_pick_takes_the_monitor_of_the_default_output() {
    let ids = [
        "alsa_output.hdmi-stereo.monitor",
        "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
        "alsa_input.pci-0000_00_1f.3.analog-stereo",
    ]
    .map(String::from);
    assert_eq!(
        pick_monitor(&ids, "alsa_output.pci-0000_00_1f.3.analog-stereo"),
        Some(1)
    );
    assert_eq!(pick_monitor(&ids, "alsa_output.hdmi-stereo"), Some(0));
    assert_eq!(pick_monitor(&ids, "alsa_output.usb-headset"), None);
}
