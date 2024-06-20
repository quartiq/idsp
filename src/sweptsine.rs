use crate::cossin;

/// Exponential sweep generator
pub struct SweepIter {
    /// Rate of increase
    pub a: i32,
    /// Current
    pub f: i64,
}

impl Iterator for SweepIter {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        self.f = self
            .f
            .wrapping_add((self.a as i64).wrapping_mul(self.f >> 32));
        Some(self.f)
    }
}

impl SweepIter {
    /// Log swept frequency
    pub fn sweep(a: i32, f1: i32, f2: i32) -> impl Iterator<Item = i32> {
        Self {
            a,
            f: (f1 as i64) << 32,
        }
        .map(|f| (f >> 32) as i32)
        .take_while(move |f| *f <= f2)
    }

    /// Log swept sine
    pub fn swept_sine(a: i32, p0: i32, f1: i32, f2: i32) -> impl Iterator<Item = i32> {
        Self::sweep(a, f1, f2).scan(p0, |p, f| {
            *p = p.wrapping_add(f);
            Some(cossin::cossin(*p).1)
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    // use quickcheck_macros::quickcheck;

    #[test]
    fn new() {
        let r = 0x3654;
        let f1 = r;
        let f2: i32 = i32::MAX >> 1;
        let it = SweepIter::swept_sine(r, 0x0, f1, f2);
        println!("{}", it.count());
    }
}
