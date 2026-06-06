//! Equal temperament calculations and tuning analysis.
//!
//! This module provides tools for working with equal temperaments (ETs) and
//! analyzing how they approximate Just Intonation intervals.
//!
//! # Key Concepts
//!
//! - **Val**: A covector mapping JI intervals to ET steps. In regular temperament
//!   theory, a val represents how each prime maps to steps in an equal temperament.
//! - **Patent val**: The val where each prime maps to its nearest integer step count.
//! - **ED (equal division)**: A division of an equave (usually octave) into equal steps.
//!   "12edo" means 12 equal divisions of the octave.
//! - **Tuning range**: For ternary scales, the set of valid tunings based on
//!   degenerate cases (where step sizes collapse).
//!
//! # Examples
//!
//! ```
//! use tern::equal::{gpval, direct_approx, relative_error};
//! use tern::ji_ratio::RawJiRatio;
//! use tern::monzo;
//!
//! // Get the patent val for 12edo
//! let val_12 = gpval(12.0);
//! // 12edo maps the octave to 12 steps, the fifth (3/2) to 7 steps
//! let fifth = monzo![-1, 1];  // 3/2 = 2^-1 * 3^1
//! assert_eq!(val_12.evaluate(fifth), 7);
//!
//! // Direct approximation of 5/4 in 12edo
//! let major_third = RawJiRatio::try_new(5, 4).unwrap();
//! let steps = direct_approx(major_third, 12.0, RawJiRatio::OCTAVE);
//! assert_eq!(steps, 4);  // 4 steps = 400 cents, approximates 386 cents
//! ```

use std::ops::{Add, AddAssign};

use num_traits::Pow;

use crate::{
    interval::{Dyad, JiRatio},
    ji::odd_limit,
    ji_ratio::RawJiRatio,
    monzo::Monzo,
    primes::{SMALL_PRIMES, SMALL_PRIMES_COUNT},
    vector::RowVector,
};

/// In regular temperament theory, a [*val*](https://en.xen.wiki/w/Val) is a covector,
/// i.e. an element of the [dual module](https://en.wikipedia.org/wiki/Dual_module) of
/// interval vectors
/// (the space of linear maps from the abelian group of intervals to ℤ).
/// Represents the steps in an equal temperament for each prime.
#[derive(Debug, Copy, Clone, Hash, PartialEq)]
pub struct Val(RowVector);

impl Val {
    /// The zero map.
    pub const ZERO: Self = Self(RowVector::new([0i32; SMALL_PRIMES_COUNT]));
    /// Unwrap the RowVector inner representation.
    pub fn into_inner(&self) -> RowVector {
        self.0
    }
    /// Create a Val from a slice. Pads with zeros up to SMALL_PRIMES_COUNT.
    pub fn from_slice(slice: &[i32]) -> Self {
        Self(RowVector::from_slice(slice))
    }
    /// Find how many steps a given val maps a monzo (a prime-factorized JI ratio) to.
    /// Computes the dot product of the val with the monzo.
    pub fn evaluate(&self, monzo: Monzo) -> i32 {
        self.0.dot(&monzo.into_inner())
    }
}

impl Add for Val {
    type Output = Self;
    /// Add two vals (combine their step mappings).
    fn add(self, rhs: Self) -> Self::Output {
        Val(self.0 + rhs.0)
    }
}

impl AddAssign for Val {
    /// In-place addition of vals.
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

/// The [generalized patent val](https://en.xen.wiki/w/Generalized_patent_val) of x-edo where x is any positive real.
/// Computes the step mapping for each prime in x-tone equal division of the octave.
///
/// # Examples
///
/// ```
/// use tern::equal::gpval;
/// use tern::monzo;
///
/// // 12edo patent val: <12 19 28 34|
/// let val_12 = gpval(12.0);
/// assert_eq!(val_12.evaluate(monzo![1]), 12);   // octave = 12 steps
/// assert_eq!(val_12.evaluate(monzo![-1, 1]), 7); // fifth = 7 steps
///
/// // Works with non-integer EDOs too
/// let val_16_9 = gpval(16.9);
/// assert_eq!(val_16_9.evaluate(monzo![1]), 17);  // rounds to nearest
/// ```
pub fn gpval(edo: f64) -> Val {
    let mut arr = [0i32; SMALL_PRIMES_COUNT];
    for (i, &p) in SMALL_PRIMES.iter().enumerate() {
        arr[i] = f64::round(edo * f64::log(p as f64, std::f64::consts::E) / std::f64::consts::LN_2)
            as i32;
    }
    Val(RowVector::new(arr))
}

/// The best approximation of `ratio` in steps of `ed`-ed<`equave`>.
/// Returns the integer number of steps that best approximates the given JI ratio.
pub fn direct_approx<J: JiRatio>(ratio: J, ed: f64, equave: J) -> i32 {
    f64::round(
        ed * (f64::log(ratio.numer() as f64, std::f64::consts::E)
            - f64::log(ratio.denom() as f64, std::f64::consts::E))
            / (f64::log(equave.numer() as f64, std::f64::consts::E)
                - f64::log(equave.denom() as f64, std::f64::consts::E)),
    ) as i32
}

/// `steps` in `ed`-ed<`equave`> converted to cents.
/// Scales the step count to the full size of the equave in cents.
pub fn steps_as_cents(steps: i32, ed: f64, equave: RawJiRatio) -> f64 {
    (steps as f64) / ed * equave.cents()
}

/// Whether `test_value` in cents is in the tuning range of a given interval ax + by + cz in a ternary scale aL bm cs.
/// The tuning range of a given step vector xL + ym + zs is the convex hull of the three degenerate tunings for it:
/// x\\a, (x+y) \\ (a+b), (x+y+z) \\ (a+b+c), i.e. the closed interval
/// [min(x\\a, (x+y) \\ (a+b), (x+y+z) \\ (a+b+c)), max(x\\a, (x+y) \\ (a+b), (x+y+z) \\ (a+b+c))].
/// In short, this follows from observing that the tuning range of a ternary scale is the convex hull of the degenerate 1:0:0, 1:1:0, and 1:1:1 tunings
/// and taking mediants for the tuning of a given interval in the ternary scale.
pub fn is_in_tuning_range(
    test_value: f64,
    step_sig: &[i32],
    steps: &[i32],
    equave: RawJiRatio,
) -> bool {
    let (a, b, c) = (step_sig[0], step_sig[1], step_sig[2]);
    let (x, y, z) = (steps[0], steps[1], steps[2]);
    let value_1_0_0 = equave.cents() * x as f64 / a as f64;
    let value_1_1_0 = equave.cents() * (x + y) as f64 / (a + b) as f64;
    let value_1_1_1 = equave.cents() * (x + y + z) as f64 / (a + b + c) as f64;
    let mut degenerate_tunings = [value_1_0_0, value_1_1_0, value_1_1_1];
    degenerate_tunings.sort_by(|a, b| a.total_cmp(b));
    let min_value = degenerate_tunings[0];
    let max_value = degenerate_tunings[2];
    debug_assert!(min_value <= max_value);
    min_value <= test_value && test_value <= max_value
}

/// All integer ed`equave` tunings for `step_sig` scales below `ed_bound`.
///
/// Returns tunings as `[L_steps, m_steps, s_steps]` where the smallest step
/// size falls within `[aber_lower, aber_upper]` cents.
///
/// # Examples
///
/// ```
/// use tern::equal::ed_tunings_for_ternary;
/// use tern::ji_ratio::RawJiRatio;
///
/// // Find ED tunings for 5L2m3s (blackdye) up to 31edo
/// // with smallest step between 20 and 60 cents
/// let tunings = ed_tunings_for_ternary(&[5, 2, 3], RawJiRatio::OCTAVE, 31, 20.0, 60.0);
/// assert!(!tunings.is_empty());
///
/// // Each tuning is [L, m, s] step counts with L > m > s
/// for t in &tunings {
///     assert!(t[0] > t[1] && t[1] > t[2]);
/// }
/// ```
pub fn ed_tunings_for_ternary(
    step_sig: &[usize],
    equave: RawJiRatio,
    ed_bound: i32,
    aber_lower: f64,
    aber_upper: f64,
) -> Vec<Vec<i32>> {
    (3..ed_bound)
        .flat_map(|l| (2..l).flat_map(move |m| (1..m).map(move |s| vec![l, m, s])))
        .filter(|edostep_counts| {
            let ed: i32 = edostep_counts[0] * step_sig[0] as i32
                + edostep_counts[1] * step_sig[1] as i32
                + edostep_counts[2] * step_sig[2] as i32;
            let aber_size = steps_as_cents(edostep_counts[2], ed as f64, equave);
            edostep_counts[0] * step_sig[0] as i32
                + edostep_counts[1] * step_sig[1] as i32
                + edostep_counts[2] * step_sig[2] as i32
                <= ed_bound
                && aber_lower <= aber_size
                && aber_size <= aber_upper
        })
        .collect()
}

/// Relative error of the patent val mapping for `monzo`.
pub fn relative_error(monzo: Monzo, edo: f64) -> f64 {
    let val = gpval(edo);
    let steps = val.evaluate(monzo);
    (steps_as_cents(steps, edo, RawJiRatio::OCTAVE) - monzo.cents()) * edo / 1200.0
}

/// L^1 error on a specified odd limit.
pub fn odd_limit_l1_error(odd: u32, edo: f64) -> f64 {
    odd_limit(odd)
        .into_iter()
        .filter(|&r| r * r < RawJiRatio::OCTAVE)
        .map(|r| Monzo::try_from_ratio(r).unwrap_or(Monzo::UNISON))
        .map(|monzo| relative_error(monzo, edo).abs()) // get the magnitudes of the relative errors of primes
        .sum()
}

/// L^2 error on a specified odd limit.
pub fn odd_limit_l2_error(odd: u32, edo: f64) -> f64 {
    odd_limit(odd)
        .into_iter()
        .filter(|&r| r * r < RawJiRatio::OCTAVE)
        .map(|r| Monzo::try_from_ratio(r).unwrap_or(Monzo::UNISON))
        .map(|monzo| relative_error(monzo, edo).pow(2)) // square the relative error of each prime
        .sum::<f64>()
        .sqrt()
}

#[macro_export]
/// Creates a [`Val`] from a list of step mappings for each prime.
///
/// # Examples
///
/// ```
/// use tern::{val, monzo};
///
/// // Create 12edo 5-limit val: <12 19 28|
/// let val_12 = val![12, 19, 28];
///
/// // The syntonic comma (81/80) is tempered out in 12edo
/// let syntonic_comma = monzo![-4, 4, -1];
/// assert_eq!(val_12.evaluate(syntonic_comma), 0);
///
/// // The major third (5/4) maps to 4 steps
/// let major_third = monzo![-2, 0, 1];
/// assert_eq!(val_12.evaluate(major_third), 4);
/// ```
macro_rules! val {
    () => (
        $crate::equal::Val::ZERO
    );
    ($elem:expr; $n:expr) => (
        $crate::equal::Val(nalgebra::RowSVector::<i32, {$crate::primes::SMALL_PRIMES_COUNT}>::from_row_slice(&[$elem; $crate::primes::SMALL_PRIMES_COUNT]))
    );
    ($($x:expr),+ $(,)?) => (
        $crate::equal::Val::from_slice(&[$($x),+])
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    #[allow(unused)]
    use super::*;
    use crate::monzo;

    #[test]
    fn test_is_in_tuning_range() {
        let seven_to_six = monzo![-1, -1, 0, 1].cents();
        assert!(is_in_tuning_range(
            seven_to_six,
            &[5, 2, 2],
            &[1, 1, 0],
            RawJiRatio::OCTAVE
        ));

        let eight_to_seven = monzo![3, 0, 0, -1].cents();
        assert!(is_in_tuning_range(
            eight_to_seven,
            &[5, 2, 2],
            &[1, 0, 1],
            RawJiRatio::OCTAVE
        ));

        let seven_to_four = monzo![-2, 0, 0, 1].cents();
        assert!(is_in_tuning_range(
            seven_to_four,
            &[5, 2, 2],
            &[4, 2, 1],
            RawJiRatio::OCTAVE
        ));

        let three_to_two = monzo![-1, 1].cents();
        assert!(is_in_tuning_range(
            three_to_two,
            &[5, 2, 2],
            &[3, 1, 1],
            RawJiRatio::OCTAVE
        ));
        assert!(is_in_tuning_range(
            three_to_two,
            &[5, 2, 3],
            &[3, 1, 2],
            RawJiRatio::OCTAVE
        ));

        let eleven_to_eight = monzo![-3, 0, 0, 0, 1].cents();
        assert!(!is_in_tuning_range(
            eleven_to_eight,
            &[5, 2, 2],
            &[2, 1, 1],
            RawJiRatio::OCTAVE
        ));

        let five_to_four = monzo![-2, 0, 1].cents();
        assert!(is_in_tuning_range(
            five_to_four,
            &[5, 2, 3],
            &[2, 0, 1],
            RawJiRatio::OCTAVE
        ));

        let six_six_six_cents = 666.;
        assert!(!is_in_tuning_range(
            six_six_six_cents,
            &[5, 2, 3],
            &[3, 1, 2],
            RawJiRatio::OCTAVE
        ));
        assert!(!is_in_tuning_range(
            six_six_six_cents,
            &[5, 2, 2],
            &[3, 1, 1],
            RawJiRatio::OCTAVE
        ));
    }
    #[test]
    fn test_rel_error() {
        let five_to_four = monzo![-2, 0, 1];
        let rel_error_in_12edo = relative_error(five_to_four, 12.0);
        assert!(rel_error_in_12edo - 0.1369 < 0.005);
        let rel_error_in_31edo = relative_error(five_to_four, 31.0);
        println!("{rel_error_in_31edo}");
        assert!(rel_error_in_31edo - 0.020 < 0.005);
    }
    /*
    #[test]
    fn test_error_on_odd_limit() {
        println!("unweighted L^1 error:\n");
        for i in 1..=15 { // odd limits 1, ..., 31
            // test 46edo
            println!("{}-odd-limit:\t31\t{:.2}\t34\t{:.2}", 2*i+1, odd_limit_l1_error(2*i+1, 31.0), odd_limit_l1_error(2*i+1, 34.0));
        }
        println!("unweighted L^1 error:\n");
        for i in 1..=15 { // odd limits 1, ..., 31
            // test 46edo
            println!("{}-odd-limit:\t41\t{:.2}\t46\t{:.2}", 2*i+1, odd_limit_l1_error(2*i+1, 41.0), odd_limit_l1_error(2*i+1, 46.0));
        }
        println!();
        for i in 1..=15 { // odd limits 1, ..., 31
            // test 46edo
            println!("{}-odd-limit:\t53\t{:.2}\t58\t{:.2}", 2*i+1, odd_limit_l1_error(2*i+1, 53.0), odd_limit_l1_error(2*i+1, 58.0));
        }
        println!();
    }
    */
    #[test]
    fn test_ternary_ed_tunings() {
        let blackdye_tunings =
            ed_tunings_for_ternary(&[5, 2, 3], RawJiRatio::OCTAVE, 53, 20.0, 60.0);
        assert_eq!(
            BTreeSet::from_iter(blackdye_tunings.into_iter()),
            BTreeSet::from_iter(
                vec![
                    vec![3, 2, 1],
                    vec![4, 2, 1],
                    vec![4, 3, 1],
                    vec![5, 2, 1],
                    vec![5, 3, 1],
                    vec![5, 4, 1],
                    vec![6, 2, 1],
                    vec![6, 3, 1],
                    vec![6, 4, 1],
                    vec![6, 3, 2],
                    vec![7, 2, 1],
                    vec![6, 5, 1],
                    vec![6, 4, 2],
                    vec![7, 3, 1],
                    vec![6, 5, 2],
                    vec![7, 4, 1],
                    vec![7, 3, 2],
                    vec![8, 2, 1],
                    vec![7, 5, 1],
                    vec![7, 4, 2],
                    vec![8, 3, 1],
                    vec![7, 6, 1],
                    vec![7, 5, 2],
                    vec![7, 4, 1],
                    vec![8, 4, 1],
                    vec![8, 3, 2],
                    vec![9, 2, 1],
                    vec![7, 6, 2],
                    vec![8, 5, 1],
                ]
                .into_iter()
            )
        );
    }

    #[test]
    fn test_val_macro() {
        let twelve_edo_5_lim = val![12, 19, 28];
        let syntonic_comma = monzo![-4, 4, -1];
        assert_eq!(twelve_edo_5_lim.evaluate(syntonic_comma), 0);
    }

    #[test]
    fn test_gpval() {
        let val_12 = gpval(12.0);
        assert_eq!(val_12.0[0], 12);
        assert_eq!(val_12.0[1], 19);
        assert_eq!(val_12.0[2], 28);
        assert_eq!(val_12.0[3], 34);
        let val_311 = gpval(311.0);
        assert_eq!(val_311.0[0], 311);
        assert_eq!(val_311.0[1], 493);
        assert_eq!(val_311.0[2], 722);
        assert_eq!(val_311.0[3], 873);
        let val_sixteen_point_nine = gpval(16.9);
        assert_eq!(val_sixteen_point_nine.0[0], 17);
        assert_eq!(val_sixteen_point_nine.0[1], 27);
        assert_eq!(val_sixteen_point_nine.0[2], 39);
    }
}
