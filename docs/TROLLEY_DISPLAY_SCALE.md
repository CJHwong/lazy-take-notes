# Desktop window reflows when it moves between displays

Status: upstream bug in Trolley's macOS runtime. The report below is a draft. Do not file it without review.

## Symptom

The desktop app runs in a Trolley window (libghostty). Moving the window between a Retina display and a 1x display changes the terminal grid. Textual reflows into a layout that looks zoomed. In some transitions the grid does not return to its original size when the window moves back.

## Reproduction

Machine: built-in Retina display, 1512x982 points at scale 2.0. BenQ EW3270ZL, 2560x1440 at scale 1.0.

The pty size is the Textual grid. Find the app pty with `lsof -p <python-pid> -a -d 0` and read it with `stty -f /dev/ttysXXX size`. It reports rows and columns.

| Step | pty |
| --- | --- |
| Window on the BenQ, 1212x692 | 36 x 151 |
| Move to the Retina display | 19 x 70 |
| Move back to the BenQ | 93 x 98 (stuck) |
| Drag a window edge 12 px | 36 x 151 |

A fullscreen toggle is not a recovery. It changes the frame through the same stale-size path. In one run exiting fullscreen left the grid at 21 x 70.

A second run recovered to 36 x 151 on the move back. The bug is a race, so it is intermittent.

## Cause

Trolley's runtime updates the content scale but not the surface pixel size. From `runtime/macos/Sources/main.swift`:

```swift
override func viewDidChangeBackingProperties() {
    super.viewDidChangeBackingProperties()
    guard let surface = gSurface else { return }
    let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 1.0
    ghostty_surface_set_content_scale(surface, Double(scale), Double(scale))
}
```

Ghostty recomputes the DPI and calls `resize(self.size.screen)` inside `contentScaleCallback()` (`src/Surface.zig`). `self.size.screen` still holds the pixel size from the last `ghostty_surface_set_size()` call, which belongs to the old display. The grid is therefore computed from a stale pixel size.

The same code is in Trolley v0.6.1 and v0.12.0. The matching Ghostty issue is #12925.

## Recovery

Drag a window edge. That fires `setFrameSize()`, which re-sends the pixel size, and the grid snaps back. Do not use fullscreen as a recovery: it can leave the grid wrong.

## Proposed upstream patch

```diff
 override func viewDidChangeBackingProperties() {
     super.viewDidChangeBackingProperties()
     guard let surface = gSurface else { return }
     let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 1.0
+    let backed = convertToBacking(bounds.size)
+    if backed.width > 0 && backed.height > 0 {
+        ghostty_surface_set_size(surface, UInt32(backed.width), UInt32(backed.height))
+    }
     ghostty_surface_set_content_scale(surface, Double(scale), Double(scale))
 }
```

`set_size` must run before `set_content_scale`, because Ghostty resizes the grid inside `contentScaleCallback()` from the stored pixel size.

If a transition still races, keep the same two calls in a `NSWindow.didChangeBackingPropertiesNotification` observer and defer them by one run loop turn, after AppKit settles the frame.

## Draft issue for Trolley

Title: macOS: terminal grid keeps the old pixel size after a display scale change

Body:

> Dragging a Trolley window between a Retina display and a 1x display changes the pty grid. On a 1212x692 window the grid went from 36x151 to 19x70, and one move back left it at 93x98. The grid only recovers after a window resize.
>
> The runtime calls `ghostty_surface_set_content_scale()` in `viewDidChangeBackingProperties()` but never calls `ghostty_surface_set_size()`. Ghostty's `contentScaleCallback()` resizes the grid from the stored pixel size, so the new scale is applied to the old display's size.
>
> Re-send `ghostty_surface_set_size(surface, backed.width, backed.height)` before `set_content_scale()` on every backing change. If AppKit still has the frame in flux, observe `NSWindow.didChangeBackingPropertiesNotification` and defer the same two calls by one run loop turn.
>
> Tested on macOS 26.6.2 with a 2x Retina display and a 1x BenQ EW3270ZL. Trolley v0.6.1 and v0.12.0 both have the bug.

## Draft note for Ghostty

The embedded API lets the host call `ghostty_surface_set_content_scale()` without a matching `ghostty_surface_set_size()`. `contentScaleCallback()` then resizes the grid from the previous pixel size, which is wrong after a display scale change. Ghostty issue #12925 covers the transient pty dimension change. Consider documenting in the embedded API that the host must re-send the surface size whenever the backing scale changes.
