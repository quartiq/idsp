//! Statically sized linear Kalman filters with scalar measurements.
//!
//! The implementation uses raw values for the estimate, covariance, and noise,
//! and coefficient values for `F`, `H`, and the Kalman gain.
//! Consequently fixed-point operation follows the same widened dot-product
//! pattern as [`crate::iir::Biquad`]: `coefficient * raw` products accumulate
//! wide and quantize once per result.
//!
//! [`RandomWalk`] is the smallest complete filter and an ordinary scalar DSP
//! stage:
//!
//! ```
//! use dsp_process::{Process, Split};
//! use idsp::kalman::{Estimate, RandomWalk};
//!
//! let config = RandomWalk::<f32>::new(0.01, 1.0);
//! let mut filter = Split::new(config, Estimate::new([0.0], [[10.0]]));
//! let filtered = filter.process(2.0);
//! assert!(filtered > 0.0 && filtered < 2.0);
//! ```
//!
//! Use [`ConstantVelocity`] with [`Direct`] for the common position sensor;
//! use [`Transition`] and [`Observation`] only when the matrices are dense.

use core::{
    iter::Sum,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use dsp_fixedpoint::FromRatio;
use dsp_process::{Map, SplitInplace, SplitProcess};
use num_traits::{AsPrimitive, ConstOne, ConstZero};

/// State estimate and its error covariance.
///
/// Fixed-point models must normalize all state components and measurements to
/// compatible numerical ranges. `covariance`, process noise, and measurement
/// noise must use one common raw covariance scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate<T, const N: usize> {
    /// Estimated state vector.
    pub state: [T; N],
    /// Symmetric state-error covariance matrix.
    ///
    /// The transition reads the complete matrix. Prediction and correction
    /// restore exact symmetry by computing the upper triangle and mirroring it.
    pub covariance: [[T; N]; N],
}

impl<T, const N: usize> Estimate<T, N> {
    /// Construct an estimate from a state and covariance.
    #[must_use]
    pub const fn new(state: [T; N], covariance: [[T; N]; N]) -> Self {
        Self { state, covariance }
    }
}

/// Linear state transition model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transition<C, const N: usize, T = C> {
    /// State transition matrix, conventionally `F`.
    pub matrix: [[C; N]; N],
    /// Symmetric process-noise covariance, conventionally `Q`.
    ///
    /// Only the upper triangle is used.
    pub noise: [[T; N]; N],
}

impl<C, const N: usize, T> Transition<C, N, T> {
    /// Construct a transition model.
    #[must_use]
    pub const fn new(matrix: [[C; N]; N], noise: [[T; N]; N]) -> Self {
        Self { matrix, noise }
    }

    /// Advance an estimate through this transition model.
    pub fn predict<A>(&self, estimate: &mut Estimate<T, N>)
    where
        T: 'static + Copy,
        C: Copy + ConstOne + Mul<T, Output = A>,
        A: Add<Output = A> + Sum + AsPrimitive<T>,
    {
        const { assert!(N > 0, "a Kalman filter needs at least one state") }

        let state = estimate.state;
        for (row, output) in self.matrix.iter().zip(&mut estimate.state) {
            let sum: A = row.iter().zip(&state).map(|(&a, &b)| a * b).sum();
            *output = sum.as_();
        }

        // Store F*P. This is the only covariance-sized scratch matrix.
        let mut fp = [[estimate.covariance[0][0]; N]; N];
        for (i, row) in fp.iter_mut().enumerate() {
            for (j, output) in row.iter_mut().enumerate() {
                let sum: A = self.matrix[i]
                    .iter()
                    .zip(&estimate.covariance)
                    .map(|(&a, row)| a * row[j])
                    .sum();
                *output = sum.as_();
            }
        }

        // F*P*F' + Q is symmetric. Compute one triangle and mirror it.
        for (i, fp) in fp.iter().enumerate() {
            for j in i..N {
                let sum: A = self.matrix[j].iter().zip(fp).map(|(&a, &b)| a * b).sum();
                let covariance = (sum + C::ONE * self.noise[i][j]).as_();
                estimate.covariance[i][j] = covariance;
                estimate.covariance[j][i] = covariance;
            }
        }
    }
}

impl<C, const N: usize, T> Transition<C, N, T>
where
    C: Copy + ConstOne + ConstZero,
    T: Copy,
{
    /// Construct an identity transition with the supplied process noise.
    #[must_use]
    pub fn identity(noise: [[T; N]; N]) -> Self {
        let mut matrix = [[C::ZERO; N]; N];
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = C::ONE;
        }
        Self::new(matrix, noise)
    }
}

/// Linear scalar observation model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation<C, const N: usize, T = C> {
    /// Scalar measurement matrix row, conventionally `H`.
    pub matrix: [C; N],
    /// Scalar measurement-noise variance, conventionally `R`.
    pub noise: T,
}

impl<C, const N: usize, T> Observation<C, N, T> {
    /// Construct a scalar observation model.
    #[must_use]
    pub const fn new(matrix: [C; N], noise: T) -> Self {
        Self { matrix, noise }
    }

    /// Project the state estimate into measurement space.
    #[must_use]
    pub fn project<A>(&self, estimate: &Estimate<T, N>) -> T
    where
        T: 'static + Copy,
        C: Copy + Mul<T, Output = A>,
        A: Sum + AsPrimitive<T>,
    {
        const { assert!(N > 0, "a Kalman filter needs at least one state") }
        let sum: A = self
            .matrix
            .iter()
            .zip(&estimate.state)
            .map(|(&a, &b)| a * b)
            .sum();
        sum.as_()
    }

    /// Correct an estimate with one scalar measurement.
    ///
    /// Returns the posterior measurement estimate. The innovation covariance
    /// must be positive and the calculated gain must fit `C`.
    pub fn correct<A>(&self, estimate: &mut Estimate<T, N>, measurement: T) -> T
    where
        T: 'static + Copy,
        C: Copy + ConstOne + FromRatio<T> + Mul<T, Output = A>,
        A: Add<Output = A> + Sub<Output = A> + Sum + AsPrimitive<T>,
    {
        const { assert!(N > 0, "a Kalman filter needs at least one state") }

        let expected: A = self
            .matrix
            .iter()
            .zip(&estimate.state)
            .map(|(&a, &b)| a * b)
            .sum();
        let innovation = (C::ONE * measurement - expected).as_();

        // u = P*H'
        let mut cross = [estimate.covariance[0][0]; N];
        for (row, output) in estimate.covariance.iter().zip(&mut cross) {
            let sum: A = self.matrix.iter().zip(row).map(|(&a, &b)| a * b).sum();
            *output = sum.as_();
        }

        // s = H*u + R
        let variance: A = self.matrix.iter().zip(&cross).map(|(&a, &b)| a * b).sum();
        let variance = (variance + C::ONE * self.noise).as_();
        finish_correction::<C, _, _, _, N>(
            estimate,
            innovation,
            cross,
            variance,
            self.noise,
            |values| {
                let sum: A = self.matrix.iter().zip(values).map(|(&a, &b)| a * b).sum();
                sum.as_()
            },
        )
    }
}

/// Direct observation of state component `I`.
///
/// Unlike a dense [`Observation`], this carries no zero coefficients and lets
/// correction reduce `H*x`, `P*H'`, and `H*P*H'` to indexing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Direct<C, const I: usize, T = C> {
    /// Scalar measurement-noise variance, conventionally `R`.
    pub noise: T,
    coefficient: PhantomData<fn() -> C>,
}

impl<C, const I: usize, T> Direct<C, I, T> {
    /// Construct a direct observation.
    #[must_use]
    pub const fn new(noise: T) -> Self {
        Self {
            noise,
            coefficient: PhantomData,
        }
    }

    /// Project the state estimate into measurement space.
    #[must_use]
    pub fn project<const N: usize>(&self, estimate: &Estimate<T, N>) -> T
    where
        T: Copy,
    {
        const { assert!(I < N, "state index out of range") }
        estimate.state[I]
    }

    /// Correct an estimate with one scalar measurement.
    ///
    /// Returns the posterior measurement estimate. The innovation covariance
    /// must be positive and the calculated gain must fit `C`.
    pub fn correct<A, const N: usize>(&self, estimate: &mut Estimate<T, N>, measurement: T) -> T
    where
        T: 'static + Copy,
        C: Copy + ConstOne + FromRatio<T> + Mul<T, Output = A>,
        A: Add<Output = A> + Sub<Output = A> + AsPrimitive<T>,
    {
        const { assert!(I < N, "state index out of range") }

        let innovation = (C::ONE * measurement - C::ONE * estimate.state[I]).as_();
        let mut cross = [estimate.covariance[0][I]; N];
        for (row, output) in estimate.covariance.iter().zip(&mut cross) {
            *output = row[I];
        }
        let variance = (C::ONE * cross[I] + C::ONE * self.noise).as_();
        finish_correction::<C, _, _, _, N>(
            estimate,
            innovation,
            cross,
            variance,
            self.noise,
            |values| values[I],
        )
    }
}

/// Constant-velocity motion model for `[position, velocity]`.
///
/// Its transition matrix is `[[1, interval], [0, 1]]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantVelocity<C, T = C> {
    /// Sample interval.
    pub interval: C,
    /// Symmetric process-noise covariance. Only the upper triangle is used.
    pub noise: [[T; 2]; 2],
}

impl<C, T> ConstantVelocity<C, T> {
    /// Construct a constant-velocity transition.
    #[must_use]
    pub const fn new(interval: C, noise: [[T; 2]; 2]) -> Self {
        Self { interval, noise }
    }

    /// Advance a position-velocity estimate.
    pub fn predict<A>(&self, estimate: &mut Estimate<T, 2>)
    where
        T: 'static + Copy,
        C: Copy + ConstOne + Mul<T, Output = A>,
        A: Add<Output = A> + AsPrimitive<T>,
    {
        let [[p00, p01], [p10, p11]] = estimate.covariance;
        estimate.state[0] = (C::ONE * estimate.state[0] + self.interval * estimate.state[1]).as_();

        let fp00 = (C::ONE * p00 + self.interval * p10).as_();
        let fp01 = (C::ONE * p01 + self.interval * p11).as_();
        let p00 = (C::ONE * fp00 + self.interval * fp01 + C::ONE * self.noise[0][0]).as_();
        let p01 = (C::ONE * fp01 + C::ONE * self.noise[0][1]).as_();
        let p11 = (C::ONE * p11 + C::ONE * self.noise[1][1]).as_();
        estimate.covariance = [[p00, p01], [p01, p11]];
    }
}

/// One prediction phase followed by one measurement-update phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kalman<P, U> {
    /// Prediction phase.
    pub predict: P,
    /// Measurement-update phase.
    pub update: U,
}

impl<P, U> Kalman<P, U> {
    /// Compose prediction and measurement-update phases over shared state.
    #[must_use]
    pub const fn new(predict: P, update: U) -> Self {
        Self { predict, update }
    }

    /// Run prediction but skip the measurement update when input is absent.
    #[must_use]
    pub fn optional(self) -> Kalman<P, Map<U>> {
        Kalman::new(self.predict, Map(self.update))
    }
}

/// Dense linear transition and scalar measurement model.
pub type DenseKalman<C, const N: usize, T = C> = Kalman<Transition<C, N, T>, Observation<C, N, T>>;

/// Complete scalar random-walk filter with `F = H = 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomWalk<C, T = C> {
    /// Scalar process-noise variance.
    pub process_noise: T,
    /// Scalar measurement-noise variance.
    pub measurement_noise: T,
    coefficient: PhantomData<fn() -> C>,
}

impl<C, T> RandomWalk<C, T> {
    /// Construct a scalar random-walk filter.
    #[must_use]
    pub const fn new(process_noise: T, measurement_noise: T) -> Self {
        Self {
            process_noise,
            measurement_noise,
            coefficient: PhantomData,
        }
    }
}

impl<C, const N: usize, T, A> SplitProcess<(), (), Estimate<T, N>> for Transition<C, N, T>
where
    T: 'static + Copy,
    C: Copy + ConstOne + Mul<T, Output = A>,
    A: Add<Output = A> + Sum + AsPrimitive<T>,
{
    fn process(&self, estimate: &mut Estimate<T, N>, (): ()) {
        self.predict(estimate);
    }
}

impl<C, const N: usize, T, A> SplitProcess<T, T, Estimate<T, N>> for Observation<C, N, T>
where
    T: 'static + Copy,
    C: Copy + ConstOne + FromRatio<T> + Mul<T, Output = A>,
    A: Add<Output = A> + Sub<Output = A> + Sum + AsPrimitive<T>,
{
    fn process(&self, estimate: &mut Estimate<T, N>, measurement: T) -> T {
        self.correct(estimate, measurement)
    }
}

impl<C, const N: usize, T> SplitInplace<T, Estimate<T, N>> for Observation<C, N, T>
where
    T: Copy,
    Self: SplitProcess<T, T, Estimate<T, N>>,
{
}

impl<C, const I: usize, const N: usize, T, A> SplitProcess<T, T, Estimate<T, N>> for Direct<C, I, T>
where
    T: 'static + Copy,
    C: Copy + ConstOne + FromRatio<T> + Mul<T, Output = A>,
    A: Add<Output = A> + Sub<Output = A> + AsPrimitive<T>,
{
    fn process(&self, estimate: &mut Estimate<T, N>, measurement: T) -> T {
        self.correct(estimate, measurement)
    }
}

impl<C, const I: usize, const N: usize, T> SplitInplace<T, Estimate<T, N>> for Direct<C, I, T>
where
    T: Copy,
    Self: SplitProcess<T, T, Estimate<T, N>>,
{
}

impl<C, T, A> SplitProcess<(), (), Estimate<T, 2>> for ConstantVelocity<C, T>
where
    T: 'static + Copy,
    C: Copy + ConstOne + Mul<T, Output = A>,
    A: Add<Output = A> + AsPrimitive<T>,
{
    fn process(&self, estimate: &mut Estimate<T, 2>, (): ()) {
        self.predict(estimate);
    }
}

impl<X: Copy, Y, S, P, U> SplitProcess<X, Y, S> for Kalman<P, U>
where
    P: SplitProcess<(), (), S>,
    U: SplitProcess<X, Y, S>,
{
    fn process(&self, state: &mut S, input: X) -> Y {
        self.predict.process(state, ());
        self.update.process(state, input)
    }
}

impl<X: Copy, S, P, U> SplitInplace<X, S> for Kalman<P, U> where Self: SplitProcess<X, X, S> {}

impl<C, T, A> SplitProcess<T, T, Estimate<T, 1>> for RandomWalk<C, T>
where
    T: 'static + Copy,
    C: Copy + ConstOne + FromRatio<T> + Mul<T, Output = A>,
    A: Add<Output = A> + Sub<Output = A> + AsPrimitive<T>,
{
    fn process(&self, estimate: &mut Estimate<T, 1>, measurement: T) -> T {
        estimate.covariance[0][0] =
            (C::ONE * estimate.covariance[0][0] + C::ONE * self.process_noise).as_();
        Direct::<C, 0, T>::new(self.measurement_noise).correct(estimate, measurement)
    }
}

impl<C, T> SplitInplace<T, Estimate<T, 1>> for RandomWalk<C, T>
where
    T: Copy,
    Self: SplitProcess<T, T, Estimate<T, 1>>,
{
}

fn finish_correction<C, T, A, F, const N: usize>(
    estimate: &mut Estimate<T, N>,
    innovation: T,
    cross: [T; N],
    variance: T,
    noise: T,
    project: F,
) -> T
where
    T: 'static + Copy,
    C: Copy + ConstOne + FromRatio<T> + Mul<T, Output = A>,
    A: Add<Output = A> + Sub<Output = A> + AsPrimitive<T>,
    F: Fn(&[T; N]) -> T,
{
    let gain = C::from_ratios(cross, variance);

    for (state, gain) in estimate.state.iter_mut().zip(gain.iter().copied()) {
        *state = (C::ONE * *state + gain * innovation).as_();
    }

    // Scalar Joseph update. P1 = P - K*(P*H')'.
    for (i, gain) in gain.iter().copied().enumerate() {
        for (j, cross) in cross.iter().copied().enumerate() {
            estimate.covariance[i][j] = (C::ONE * estimate.covariance[i][j] - gain * cross).as_();
        }
    }

    // v = P1*H'.
    let mut projected = [project(&estimate.covariance[0]); N];
    for (row, output) in estimate.covariance.iter().zip(&mut projected) {
        *output = project(row);
    }

    // P+ = P1 - v*K' + (K*R)*K'. Compute one triangle and mirror it.
    for (i, gain_i) in gain.iter().copied().enumerate() {
        let weighted_noise = (gain_i * noise).as_();
        for (j, gain_j) in gain.iter().copied().enumerate().skip(i) {
            let covariance = (C::ONE * estimate.covariance[i][j] - gain_j * projected[i]
                + gain_j * weighted_noise)
                .as_();
            estimate.covariance[i][j] = covariance;
            estimate.covariance[j][i] = covariance;
        }
    }

    project(&estimate.state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dsp_fixedpoint::Q32;
    use dsp_process::{Process, Split};

    const EPS: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < EPS, "{actual} != {expected}");
    }

    #[test]
    fn scalar_reference() {
        let mut filter = Split::new(
            RandomWalk::<f64>::new(0.25, 4.0),
            Estimate::new([2.0], [[9.0]]),
        );

        let output = filter.process(5.0);
        let prior_covariance = 9.25;
        let gain = prior_covariance / (prior_covariance + 4.0);
        let expected = 2.0 + gain * (5.0 - 2.0);
        let expected_covariance = (1.0 - gain) * prior_covariance;

        assert_close(output, expected);
        assert_close(filter.state.state[0], expected);
        assert_close(filter.state.covariance[0][0], expected_covariance);
    }

    #[test]
    fn transition_and_observation_match_combined_process() {
        let transition = ConstantVelocity::new(1.0, [[0.1, 0.0], [0.0, 0.1]]);
        let observation = Direct::<f64, 0>::new(0.5);
        let config = Kalman::new(transition, observation);
        let estimate = Estimate::new([0.0, 1.0], [[2.0, 0.0], [0.0, 1.0]]);
        let mut separate = estimate;
        let mut combined = Split::new(config, estimate);

        transition.predict(&mut separate);
        assert_eq!(separate.state, [1.0, 1.0]);
        assert_eq!(separate.covariance, [[3.1, 1.0], [1.0, 1.1]]);
        let expected = observation.correct(&mut separate, 1.25);
        let actual = combined.process(1.25);

        assert_eq!(actual, expected);
        assert_eq!(combined.state, separate);
        assert_eq!(
            combined.state.covariance[0][1],
            combined.state.covariance[1][0]
        );
    }

    #[test]
    fn direct_matches_dense() {
        let estimate = Estimate::new([2.0, -1.0], [[4.0, 1.5], [1.5, 3.0]]);
        let mut direct = estimate;
        let mut dense = estimate;

        let direct_output = Direct::<f64, 0>::new(0.75).correct(&mut direct, 3.5);
        let dense_output = Observation::new([1.0, 0.0], 0.75).correct(&mut dense, 3.5);

        assert_eq!(direct_output, dense_output);
        assert_eq!(direct, dense);
    }

    #[test]
    fn joseph_matches_closed_form() {
        let observation = Observation::new([1.25, -0.5], 0.75);
        let prior = Estimate::new([2.0, -1.0], [[4.0, 1.5], [1.5, 3.0]]);
        let mut actual = prior;

        let output = observation.correct(&mut actual, 3.5);

        let h = observation.matrix;
        let u = [
            prior.covariance[0][0] * h[0] + prior.covariance[0][1] * h[1],
            prior.covariance[1][0] * h[0] + prior.covariance[1][1] * h[1],
        ];
        let variance = h[0] * u[0] + h[1] * u[1] + observation.noise;
        let gain = [u[0] / variance, u[1] / variance];
        let innovation = 3.5 - h[0] * prior.state[0] - h[1] * prior.state[1];
        let state = [
            prior.state[0] + gain[0] * innovation,
            prior.state[1] + gain[1] * innovation,
        ];
        for (actual, expected) in actual.state.iter().zip(state) {
            assert_close(*actual, expected);
        }
        for (i, (actual, prior)) in actual.covariance.iter().zip(prior.covariance).enumerate() {
            for (j, (actual, prior)) in actual.iter().zip(prior).enumerate() {
                assert_close(*actual, prior - gain[i] * u[j]);
            }
        }
        assert_close(output, h[0] * state[0] + h[1] * state[1]);
    }

    #[test]
    fn optional_measurement_still_predicts() {
        let config = Kalman::new(
            ConstantVelocity::new(1.0, [[0.0; 2]; 2]),
            Direct::<f32, 0>::new(1.0),
        )
        .optional();
        let mut filter = Split::new(config, Estimate::new([2.0, 3.0], [[1.0, 0.0], [0.0, 1.0]]));

        let output = filter.process(None);

        assert_eq!(output, None);
        assert_eq!(filter.state.state, [5.0, 3.0]);
    }

    #[test]
    fn fixed_point_tracks_float() {
        type C = Q32<28>;

        let transition = ConstantVelocity::new(C::from_f32(0.25), [[16, 0], [0, 4]]);
        let observation = Direct::<C, 0, i32>::new(64);
        let mut fixed = Split::new(
            Kalman::new(transition, observation),
            Estimate::new([0, 40], [[256, 0], [0, 64]]),
        );
        let float_transition = ConstantVelocity::new(0.25, [[16.0, 0.0], [0.0, 4.0]]);
        let float_observation = Direct::<f64, 0>::new(64.0);
        let mut float = Split::new(
            Kalman::new(float_transition, float_observation),
            Estimate::new([0.0, 40.0], [[256.0, 0.0], [0.0, 64.0]]),
        );

        for measurement in [12, 24, 37, 49, 63, 74] {
            let fixed_output = fixed.process(measurement);
            let float_output = float.process(measurement as f64);
            assert!((fixed_output as f64 - float_output).abs() < 3.0);
        }

        assert_eq!(fixed.state.covariance[0][1], fixed.state.covariance[1][0]);
    }
}
