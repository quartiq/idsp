/// Exponential sweep generator
pub struct SweepIter {
    /// Rate of increase
    pub a: u32,
    /// Current
    pub f: u64,
}

impl Iterator for SweepIter {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        const BIAS: u64 = 1 << (32 - 1);
        self.f += (self.a as u64) * ((self.f + BIAS) >> 32);
        Some(self.f)
    }
}

pub fn sweep(k: u32, f1: u32, f2: u32) -> impl Iterator<Item = u32> {
    let it = SweepIter {
        a: 1,
        f: (f1 as u64) << 32,
    };
    it.take_while(move |f| (f >> 32) as u32 <= f2)
        .scan(0u64, |p, f| {
            *p = p.wrapping_add(f);
            Some((*p >> 32) as _)
        })
}
