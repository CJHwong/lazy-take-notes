//! Convert the mic's native rate to 16 kHz, in blocks of 512 samples.
//!
//! The device always runs at its own rate: asking Core Audio for another rate
//! changes the device clock for every app on the machine, including the call.
//! An integer ratio (48 or 96 kHz) is decimated by the mean of each group, as
//! the Python version does. Any other ratio (44.1 kHz, 24 kHz call-mode
//! headsets) goes through a windowed-sinc polyphase filter, which Python got
//! from PortAudio. Leftover samples carry over to the next callback, so the
//! output keeps pace with wall time.

/// `MicBoost` frames are 256 samples; a block holds exactly two.
pub const BLOCK: usize = 512;

/// Filter half-width, in samples at the lower of the two rates.
const HALF_WIDTH: usize = 32;
/// Passband edge as a fraction of the lower rate. The Blackman window's transition
/// band ends near the lower rate's Nyquist frequency, so little audio folds back.
const CUTOFF: f64 = 0.45;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

struct Polyphase {
    up: u64,
    down: u64,
    coeffs: Vec<f32>,
    history: Vec<f32>,
    /// Global index of `history[0]`.
    first: u64,
    received: u64,
    next_out: u64,
}

impl Polyphase {
    fn new(native: u32, target: u32) -> Self {
        let divisor = gcd(u64::from(native), u64::from(target));
        let (up, down) = (u64::from(target) / divisor, u64::from(native) / divisor);
        let low = f64::from(native.min(target));
        let high = f64::from(native) * up as f64; // the rate between the up and down steps
        let span = (2 * HALF_WIDTH) as f64 * high / low;
        let taps = span as usize | 1;
        let middle = (taps - 1) as f64 / 2.0;
        let cutoff = CUTOFF * low / high; // cycles per sample at the high rate
        let coeffs = (0..taps)
            .map(|k| {
                let x = k as f64 - middle;
                let sinc = if x == 0.0 {
                    1.0
                } else {
                    (std::f64::consts::TAU * cutoff * x).sin()
                        / (std::f64::consts::PI * 2.0 * cutoff * x)
                };
                let phase = std::f64::consts::TAU * k as f64 / (taps - 1) as f64;
                let window = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
                (up as f64 * 2.0 * cutoff * sinc * window) as f32
            })
            .collect();
        Self {
            up,
            down,
            coeffs,
            history: Vec::new(),
            first: 0,
            received: 0,
            next_out: 0,
        }
    }

    fn oldest_needed(&self, out: u64) -> u64 {
        let reach = self.coeffs.len() as u64 - 1;
        (out * self.down).saturating_sub(reach).div_ceil(self.up)
    }

    fn push(&mut self, input: &[f32], output: &mut Vec<f32>) {
        self.history.extend_from_slice(input);
        self.received += input.len() as u64;
        loop {
            let position = self.next_out * self.down;
            let newest = position / self.up;
            if newest >= self.received {
                break;
            }
            let mut sum = 0.0f32;
            for index in self.oldest_needed(self.next_out).max(self.first)..=newest {
                let tap = (position - index * self.up) as usize;
                sum += self.coeffs[tap] * self.history[(index - self.first) as usize];
            }
            output.push(sum);
            self.next_out += 1;
        }
        let keep_from = self.oldest_needed(self.next_out).min(self.received);
        if keep_from > self.first {
            self.history.drain(..(keep_from - self.first) as usize);
            self.first = keep_from;
        }
    }
}

enum Kind {
    Pass,
    Mean { ratio: usize, carry: Vec<f32> },
    Polyphase(Box<Polyphase>),
}

pub struct RateConverter {
    kind: Kind,
    pending: Vec<f32>,
}

impl RateConverter {
    pub fn new(native: u32, target: u32) -> Self {
        let kind = if native == target {
            Kind::Pass
        } else if native > target && native.is_multiple_of(target) {
            Kind::Mean {
                ratio: (native / target) as usize,
                carry: Vec::new(),
            }
        } else {
            Kind::Polyphase(Box::new(Polyphase::new(native, target)))
        };
        Self {
            kind,
            pending: Vec::new(),
        }
    }

    /// How the native rate is converted, for the log.
    pub fn describe(&self) -> String {
        match &self.kind {
            Kind::Pass => "none".into(),
            Kind::Mean { ratio, .. } => format!("mean of {ratio}"),
            Kind::Polyphase(filter) => format!("polyphase {}/{}", filter.up, filter.down),
        }
    }

    /// Add native-rate samples; return each whole 512-sample block now ready.
    pub fn push(&mut self, input: &[f32]) -> Vec<Vec<f32>> {
        match &mut self.kind {
            Kind::Pass => self.pending.extend_from_slice(input),
            Kind::Mean { ratio, carry } => {
                carry.extend_from_slice(input);
                let groups = carry.chunks_exact(*ratio);
                let used = groups.len() * *ratio;
                self.pending
                    .extend(groups.map(|group| group.iter().sum::<f32>() / *ratio as f32));
                carry.drain(..used);
            }
            Kind::Polyphase(filter) => filter.push(input, &mut self.pending),
        }
        let whole = self.pending.len() / BLOCK * BLOCK;
        let blocks = self.pending[..whole]
            .as_chunks::<BLOCK>()
            .0
            .iter()
            .map(|block| block.to_vec())
            .collect();
        self.pending.drain(..whole);
        blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, hz: f64, seconds: f64) -> Vec<f32> {
        let count = (f64::from(rate) * seconds) as usize;
        (0..count)
            .map(|i| (std::f64::consts::TAU * hz * i as f64 / f64::from(rate)).sin() as f32 * 0.5)
            .collect()
    }

    /// Feed in uneven callback sizes, as Core Audio does.
    fn convert(native: u32, input: &[f32]) -> Vec<f32> {
        let mut converter = RateConverter::new(native, 16_000);
        let mut out = Vec::new();
        for chunk in input.chunks(471) {
            for block in converter.push(chunk) {
                assert_eq!(block.len(), BLOCK);
                out.extend(block);
            }
        }
        out
    }

    fn rms(samples: &[f32]) -> f64 {
        (samples
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt()
    }

    #[test]
    fn every_rate_keeps_pace_with_wall_time() {
        for native in [16_000, 24_000, 44_100, 48_000, 96_000] {
            let out = convert(native, &vec![0.0; native as usize * 10]);
            // At most one partial block and the filter's few samples of delay are pending.
            assert!(
                (160_000 - out.len()) < BLOCK + 64,
                "{native}: {} samples",
                out.len()
            );
        }
    }

    #[test]
    fn mean_decimation_averages_each_group_across_callbacks() {
        let mut converter = RateConverter::new(48_000, 16_000);
        let input: Vec<f32> = (0..BLOCK * 3).map(|i| i as f32).collect();
        assert!(converter.push(&input[..1000]).is_empty());
        let blocks = converter.push(&input[1000..]);
        assert_eq!(blocks.len(), 1);
        assert_eq!(&blocks[0][..3], &[1.0, 4.0, 7.0]);
    }

    #[test]
    fn speech_band_passes_and_alias_bands_are_cut() {
        for native in [24_000, 44_100] {
            let skip = 1_000; // the filter's start-up
            let pass = convert(native, &sine(native, 1_000.0, 2.0));
            let level = rms(&pass[skip..]) / (0.5 / 2f64.sqrt());
            assert!((level - 1.0).abs() < 0.01, "{native}: 1 kHz at {level}");
            // 11 kHz folds to 5 kHz at 16 kHz output; it must be gone.
            let alias = convert(native, &sine(native, 11_000.0, 2.0));
            let leak = 20.0 * (rms(&alias[skip..]) / (0.5 / 2f64.sqrt())).log10();
            assert!(leak < -60.0, "{native}: 11 kHz leaks at {leak:.1} dB");
        }
    }
}
