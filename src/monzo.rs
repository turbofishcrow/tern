//! Prime-factorized representation of Just Intonation intervals.
//!
//! A **monzo** represents a JI ratio as a vector of prime exponents. For example,
//! the ratio 3/2 = 2^(-1) × 3^1 is represented as `[-1, 1, 0, 0, ...]`.
//!
//! This representation enables efficient arithmetic on JI intervals:
//! - Stacking intervals = vector addition
//! - Inverting intervals = vector negation
//! - Computing cents = dot product with log(primes)
//!
//! # Examples
//!
//! ```
//! use tern::{monzo, monzo::Monzo};
//! use tern::interval::{Dyad, JiRatio};
//!
//! // Create common intervals
//! let octave = Monzo::OCTAVE;           // 2/1
//! let fifth = Monzo::PYTH_5TH;          // 3/2
//!
//! // Create from a ratio
//! let syntonic_comma = Monzo::try_new(81, 80).unwrap();
//! assert_eq!(syntonic_comma.numer(), 81);
//! assert_eq!(syntonic_comma.denom(), 80);
//!
//! // Use the monzo! macro for convenience
//! let same_comma = monzo![-4, 4, -1];   // 2^-4 × 3^4 × 5^-1 = 81/80
//! assert_eq!(syntonic_comma, same_comma);
//!
//! // Arithmetic via the Dyad trait
//! let fourth = octave.unstack(fifth);   // 2/1 ÷ 3/2 = 4/3
//! assert_eq!(fourth, Monzo::PYTH_4TH);
//!
//! // Get size in cents
//! let fifth_cents = fifth.cents();
//! assert!((fifth_cents - 701.96).abs() < 0.01);
//! ```
//!
//! # Prime Limit
//!
//! Monzos are bounded by [`SMALL_PRIMES_COUNT`],
//! currently supporting primes up to 13 (13-prime-limit).

use std::cmp::Ordering;
use std::f64::consts::LOG2_E;
use std::iter::Sum;
use std::ops::Index;

use itertools::Itertools;

use crate::helpers::{bezout, is_sorted_strictly_desc};
use crate::interval::{Dyad, JiRatio};
use crate::ji_ratio::RawJiRatio;
use crate::primes::{SMALL_PRIMES, SMALL_PRIMES_COUNT, factorize, log_primes};
use crate::vector::{Vector, Vectorf64};

/// Function type for weighting monzo components (used in norm calculations).
// TODO: Change `fn` to `Fn`
type Weighting = fn(Monzo) -> Vectorf64;

#[macro_export]
/// Creates a [`Monzo`] from prime exponents.
///
/// # Examples
///
/// ```
/// use tern::monzo;
/// use tern::interval::JiRatio;
///
/// // Empty = unison (1/1)
/// let unison = monzo![];
///
/// // Exponents for primes 2, 3, 5, ...
/// let syntonic_comma = monzo![-4, 4, -1];  // 81/80
/// assert_eq!(syntonic_comma.numer(), 81);
/// assert_eq!(syntonic_comma.denom(), 80);
/// ```
macro_rules! monzo {
    () => (
        $crate::monzo::Monzo::UNISON
    );
    ($elem:expr; $n:expr) => (
        $crate::monzo::Monzo(nalgebra::SVector::<i32, {$crate::primes::SMALL_PRIMES_COUNT}>::from_column_slice(&[$elem; $crate::primes::SMALL_PRIMES_COUNT]))
    );
    ($($x:expr),+ $(,)?) => (
        $crate::monzo::Monzo::from_slice(&[$($x),+])
    );
}

#[macro_export]
/// Creates a const `Monzo`. The array has to be empty or of length exactly `PRIME_LIMIT.len()`.
macro_rules! const_monzo {
    () => (
        $crate::monzo::Monzo::UNISON
    );
    ($($x:expr),+ $(,)?) => (
        $crate::monzo::Monzo::from_array([$($x),+])
    );

}

/// Error type for attempts to construct invalid monzos.
#[derive(Debug, PartialEq)]
pub enum CantMakeMonzo {
    /// The numerator exceeded `SMALL_PRIMES`-prime limit (contains a prime factor > largest SMALL_PRIME).
    NumerExceededPrimeLimit(Vec<u32>),
    /// The denominator exceeded `SMALL_PRIMES`-prime limit (contains a prime factor > largest SMALL_PRIME).
    DenomExceededPrimeLimit(Vec<u32>),
    /// The numerator was 0 (invalid for JI ratios).
    NumerCantBeZero,
    /// The denominator was 0 (invalid for JI ratios).
    DenomCantBeZero,
}

/// A Just Intonation interval represented as a vector of prime exponents.
///
/// Stores exponents of prime factors in order (2, 3, 5, 7, 11, ...).
/// For example, 3/2 = 2^(-1) × 3^1 is stored as `[-1, 1, 0, 0, ...]`.
///
/// # Creating Monzos
///
/// ```
/// use tern::{monzo, monzo::Monzo};
///
/// // From a ratio (fallible)
/// let major_third = Monzo::try_new(5, 4).unwrap();
///
/// // Using the macro (panics if invalid)
/// let minor_third = monzo![-5, 1, 1];  // 6/5
///
/// // Built-in constants
/// let octave = Monzo::OCTAVE;
/// let fifth = Monzo::PYTH_5TH;
/// ```
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Monzo(Vector);

impl Monzo {
    /// 1/1 in monzo form.
    pub const UNISON: Self = Self(Vector::new([0; SMALL_PRIMES_COUNT]));
    /// 2/1 in monzo form.
    pub const OCTAVE: Self = Self(Vector::new([1, 0, 0, 0, 0, 0, 0, 0, 0]));
    /// 3/2 in monzo form.
    pub const PYTH_5TH: Self = Self(Vector::new([-1, 1, 0, 0, 0, 0, 0, 0, 0]));
    /// 4/3 in monzo form.
    pub const PYTH_4TH: Self = Self(Vector::new([2, -1, 0, 0, 0, 0, 0, 0, 0]));
    /// Unwrap the Vector representation.
    pub fn into_inner(&self) -> Vector {
        self.0
    }
    /// Get the monzo for the `n`th prime. Panics if `n >= SMALL_PRIMES_COUNT`.
    /// Returns a monzo with 1 at position n and 0 elsewhere (represents the nth prime).
    pub fn nth_prime(n: usize) -> Self {
        let mut arr = [0i32; SMALL_PRIMES_COUNT];
        arr[n] = 1;
        Self(Vector::new(arr))
    }
    /// Get a const monzo from a const array.
    /// Array length must be exactly SMALL_PRIMES_COUNT.
    pub const fn from_array(arr: [i32; SMALL_PRIMES_COUNT]) -> Self {
        Self(Vector::new(arr))
    }
    /// Get a monzo from a slice. Panics if `slice.len() > SMALL_PRIMES_COUNT`.
    /// Pads with zeros if slice is shorter than SMALL_PRIMES_COUNT.
    pub fn from_slice(slice: &[i32]) -> Self {
        Self(Vector::from_slice(slice))
    }
    /// Whether the monzo represents an interval with positive logarithmic size (ratio > 1/1).
    pub fn is_positive(self) -> bool {
        self.cents() > 0.0
    }

    /// Tries to convert the JI ratio `numer`/`denom` into monzo form.
    ///
    /// Factorizes numerator and denominator, validates they fit within the prime limit.
    ///
    /// # Examples
    ///
    /// ```
    /// use tern::monzo::{Monzo, CantMakeMonzo};
    /// use tern::interval::JiRatio;
    ///
    /// // Valid ratio within prime limit
    /// let fifth = Monzo::try_new(3, 2).unwrap();
    /// assert_eq!(fifth.numer(), 3);
    /// assert_eq!(fifth.denom(), 2);
    ///
    /// // Zero is invalid
    /// assert!(matches!(
    ///     Monzo::try_new(0, 1),
    ///     Err(CantMakeMonzo::NumerCantBeZero)
    /// ));
    /// ```
    pub fn try_new(numer: u32, denom: u32) -> Result<Monzo, CantMakeMonzo> {
        if numer == 0 {
            return Err(CantMakeMonzo::NumerCantBeZero);
        }
        if denom == 0 {
            return Err(CantMakeMonzo::DenomCantBeZero);
        }
        let numer_factors = factorize(numer);
        let numer_primes_too_big: Vec<u32> = numer_factors
            .iter()
            .copied()
            .skip_while(|p| *p <= SMALL_PRIMES[SMALL_PRIMES_COUNT - 1])
            .collect();
        if !numer_primes_too_big.is_empty() {
            Err(CantMakeMonzo::NumerExceededPrimeLimit(numer_primes_too_big))
        } else {
            let denom_factors = factorize(denom);
            let denom_primes_too_big: Vec<u32> = denom_factors
                .iter()
                .copied()
                .skip_while(|p| *p <= SMALL_PRIMES[SMALL_PRIMES_COUNT - 1])
                .collect();
            if !denom_primes_too_big.is_empty() {
                Err(CantMakeMonzo::DenomExceededPrimeLimit(denom_primes_too_big))
            } else {
                let mut result = vec![0; SMALL_PRIMES_COUNT];
                let mut numer_factors_idx: usize = 0;
                let mut denom_factors_idx: usize = 0;
                for (prime_idx, p) in SMALL_PRIMES.into_iter().enumerate() {
                    while numer_factors_idx < numer_factors.len()
                        && numer_factors[numer_factors_idx] < p
                    {
                        numer_factors_idx += 1;
                    }
                    while denom_factors_idx < denom_factors.len()
                        && denom_factors[denom_factors_idx] < p
                    {
                        denom_factors_idx += 1;
                    }
                    if numer_factors_idx >= numer_factors.len()
                        && denom_factors_idx >= denom_factors.len()
                    {
                        break;
                    }
                    if numer_factors_idx < numer_factors.len()
                        && numer_factors[numer_factors_idx] == p
                    {
                        while numer_factors_idx < numer_factors.len()
                            && numer_factors[numer_factors_idx] == p
                        {
                            result[prime_idx] += 1;
                            numer_factors_idx += 1;
                        }
                    }
                    if denom_factors_idx < denom_factors.len()
                        && denom_factors[denom_factors_idx] == p
                    {
                        while denom_factors_idx < denom_factors.len()
                            && denom_factors[denom_factors_idx] == p
                        {
                            result[prime_idx] -= 1;
                            denom_factors_idx += 1;
                        }
                    }
                }
                Ok(Monzo::from_slice(&result))
            }
        }
    }
    /// Attempt to convert a JI ratio into a monzo.
    pub fn try_from_ratio(r: RawJiRatio) -> Result<Monzo, CantMakeMonzo> {
        // A `RawJiRatio` will never have zero numerator or denominator.
        Monzo::try_new(r.numer(), r.denom())
    }
    /// Attempt to convert a monzo into a JI ratio.
    pub fn try_to_ratio(&self) -> Option<RawJiRatio> {
        let numer = self.numer();
        let denom = self.denom();
        RawJiRatio::try_new(numer, denom).ok()
    }
    /// Whether all entries of a monzo are divisible by `rhs`.
    /// Used for checking if a monzo represents an integer power of another.
    pub fn is_divisible_by(&self, rhs: i32) -> bool {
        self.0.iter().all(|ex| ex % rhs == 0)
    }
}

impl Index<usize> for Monzo {
    type Output = i32;
    /// Index into the monzo to get the exponent of the nth prime.
    fn index(&self, idx: usize) -> &Self::Output {
        &self.0[idx]
    }
}

impl std::ops::Mul<i32> for Monzo {
    type Output = Monzo;
    fn mul(self, coeff: i32) -> Self {
        Monzo::from_slice(&self.0.into_iter().map(|ex| ex * coeff).collect::<Vec<_>>())
    }
}

impl std::ops::Div<i32> for Monzo {
    type Output = Monzo;
    fn div(self, coeff: i32) -> Self {
        Monzo::from_slice(&self.0.into_iter().map(|ex| ex / coeff).collect::<Vec<_>>())
    }
}

impl std::ops::Add for Monzo {
    type Output = Monzo;
    fn add(self, other: Self) -> Self {
        Monzo(self.0 + other.0)
    }
}
impl std::ops::AddAssign for Monzo {
    fn add_assign(&mut self, other: Self) {
        self.0 = self.0 + other.0;
    }
}
impl std::ops::Sub for Monzo {
    type Output = Monzo;
    fn sub(self, other: Self) -> Self {
        Monzo(self.0 - other.0)
    }
}
impl std::ops::SubAssign for Monzo {
    fn sub_assign(&mut self, other: Self) {
        self.0 = self.0 - other.0;
    }
}
impl std::ops::Neg for Monzo {
    type Output = Monzo;
    fn neg(self) -> Self {
        Monzo(-self.0)
    }
}

impl std::fmt::Display for Monzo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, &exp) in self.0.iter().enumerate() {
            write!(f, "{exp}")?;
            if i < SMALL_PRIMES_COUNT - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ">")
    }
}

impl Ord for Monzo {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.ln()).total_cmp(&other.ln()) // efficient definition
    }
}

impl PartialOrd for Monzo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Dyad for Monzo {
    fn stack(self, rhs: Self) -> Self {
        Monzo(self.0 + rhs.0)
    }
    fn unstack(self, rhs: Self) -> Self {
        Monzo(self.0 - rhs.0)
    }
    fn unison() -> Self {
        Self::UNISON
    }
    fn log_inv(self) -> Self {
        Monzo(-self.0)
    }
    fn pow(self, n: i32) -> Self {
        Monzo(n * self.0)
    }
    fn cents(self) -> f64 {
        self.0
            .iter()
            .enumerate()
            .map(|(i, exponent)| (*exponent as f64) * log_primes()[i] * LOG2_E * 1200.0)
            .sum()
    }
    fn ln(self) -> f64 {
        self.0
            .iter()
            .enumerate()
            .map(|(i, exponent)| (*exponent as f64) * log_primes()[i])
            .sum()
    }
}

impl JiRatio for Monzo {
    fn numer(&self) -> u32 {
        self.0
            .into_iter()
            .enumerate()
            .filter(|(_, exp)| *exp > 0i32)
            .map(|(i, exp)| SMALL_PRIMES[i].pow(exp as u32))
            .product()
    }
    fn denom(&self) -> u32 {
        self.0
            .into_iter()
            .enumerate()
            .filter(|(_, exp)| *exp < 0i32)
            .map(|(i, exp)| SMALL_PRIMES[i].pow(-exp as u32))
            .product()
    }
}

/// The unweighted L^1 norm of a monzo (sum of absolute exponents).
pub fn l1_norm(v: Monzo) -> f64 {
    v.0.iter().map(|x| (*x as f64).abs()).sum()
}

/// The unweighted L^2 norm of a monzo (Euclidean distance).
pub fn l2_norm(v: Monzo) -> f64 {
    v.0.iter()
        .map(|x| *x as f64)
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// The unweighted L^∞ norm of a monzo (maximum absolute exponent).
pub fn linf_norm(v: Monzo) -> f64 {
    v.0.iter()
        .map(|x| *x as f64)
        .map(|x| x.abs())
        .reduce(f64::max)
        .unwrap_or(0.0)
}

/// The weighted L^1 norm of a monzo.
/// Applies a weighting function to scale the contribution of each prime.
pub fn weighted_l1_norm(weighting: Box<Weighting>, v: Monzo) -> f64 {
    weighting(v).into_iter().map(|x| x.abs()).sum()
}

/// The weighted L^2 norm of a monzo.
/// Applies a weighting function before computing Euclidean distance.
pub fn weighted_l2_norm(weighting: Box<Weighting>, v: Monzo) -> f64 {
    weighting(v).into_iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The weighted L^∞ norm of a monzo.
/// Applies a weighting function before computing maximum absolute value.
pub fn weighted_linf_norm(weighting: Box<Weighting>, v: Monzo) -> f64 {
    weighting(v)
        .into_iter()
        .map(|x| x.abs())
        .reduce(f64::max)
        .unwrap_or(0.0)
}

/// Tenney weighting: weights each prime by its logarithm.
/// Used for complexity-aware distance metrics.
#[allow(unused)]
fn tenney_weighting(v: Monzo) -> Vectorf64 {
    let vec = (0..SMALL_PRIMES_COUNT)
        .map(|i| log_primes()[i] * (v[i] as f64))
        .collect();
    Vectorf64::from_vec(vec)
}

/// No weighting: just converts entries to f64.
#[allow(unused)]
fn unweighting(v: Monzo) -> Vectorf64 {
    Vectorf64::from_vec(v.0.iter().map(|x| *x as f64).collect())
}

/// Weil weighting: weights each prime by its logarithm squared.
/// Used for complexity metrics that penalize large prime factors more heavily.
#[allow(unused)]
fn weil_weighting(v: Monzo) -> Vectorf64 {
    let diagonal = log_primes();
    let mut result = [0.0f64; SMALL_PRIMES_COUNT];
    for i in 0..SMALL_PRIMES_COUNT {
        result[i] = diagonal[i] * diagonal[i] * (v.0[i] as f64);
    }
    Vectorf64::new(result)
}

/// Recursively solve linear Diophantine equation with bounds on exponents.
/// Explores solutions by trying different values for coefficients.
fn solve_linear_diophantine_rec(coeffs: &[i32], constant: i32, bound: i32) -> Vec<Vec<i32>> {
    if coeffs.is_empty() {
        vec![]
    } else if coeffs.len() == 1 {
        let a = coeffs[0];
        if constant % a == 0 {
            (-bound..=bound)
                .map(|x| vec![x * constant / a])
                .collect::<Vec<_>>()
        } else {
            vec![]
        }
    } else {
        let homogeneous_solns = solve_linear_diophantine_homogeneous(coeffs, bound);
        let (d, xs) = bezout(coeffs);
        if constant % d == 0 {
            let particular_soln = xs
                .into_iter()
                .map(|x| x * (constant / d))
                .collect::<Vec<_>>();
            homogeneous_solns
                .into_iter()
                .map(|soln| {
                    // Add `particular_soln` to each `soln`
                    soln.into_iter()
                        .enumerate()
                        .map(|(i, expon)| expon + particular_soln[i])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            vec![]
        }
    }
}

/// Solve homogeneous linear Diophantine equation with bounded exponents.
/// Finds integer solutions where the linear combination equals zero.
fn solve_linear_diophantine_homogeneous(coeffs: &[i32], bound: i32) -> Vec<Vec<i32>> {
    if coeffs.is_empty() {
        vec![]
    } else if coeffs.len() == 1 {
        vec![vec![0]] // ax == 0 implies x == 0
    } else if coeffs.len() == 2 {
        let (a, b) = (coeffs[0], coeffs[1]);
        let d = bezout(&[a, b]).0;
        (-bound..=bound)
            .map(|k| vec![k * (b / d), -k * (a / d)])
            .collect::<Vec<_>>()
    } else {
        let (head, tail) = coeffs
            .split_first()
            .expect("should be sound because coeffs.len() >= 3");
        (-bound..=bound)
            .flat_map(|k| {
                let mut solns = solve_linear_diophantine_rec(tail, -k * head, bound);
                for vec in solns.iter_mut() {
                    vec.insert(0, k);
                }
                solns
            })
            .collect::<Vec<_>>()
    }
}

/// Solve general linear Diophantine equation with bounded exponents.
fn solve_linear_diophantine(coeffs: &[i32], constant: i32, exponent_bound: i32) -> Vec<Vec<i32>> {
    let mut solns = solve_linear_diophantine_rec(coeffs, constant, exponent_bound);
    solns.sort();
    solns.dedup();
    solns
}
/// Get a set of solutions `x_i` of bounded complexity to
/// the equation `step_sig[0] x_0 + step_sig[1] x_1 + ... + step_sig[len - 1] x_{len-1} == equave`,
/// where `x_i` are JI ratios > 1.
/// All solutions satisfy `max(abs((x_i)_j)) <= 10` for any step size `x_i` and for any monzo component `(x_i)_j` of `x_i`.
/// Used for finding equal-tempered approximations of JI scales.
// TODO: ISSUES: (1) doesn't find all solutions with exponent bound (2) perf
pub fn solve_step_sig(step_sig: &[usize], equave: Monzo, exponent_bound: i32) -> Vec<Vec<Monzo>> {
    // Given the equation `a_1 v_1 + ... a_n v_n == equave` in monzos
    // there are at most `SMALL_PRIMES_COUNT` linear Diophantine equations
    // `a_1 x_1j ... a_n x_nj == equave_j` to solve.
    // The solution set is
    // `[solutions to a_1 x_1j ... a_n x_nj == 0] + [particular solution]`
    // assuming a particular solution exists.
    // It might happen that no solution exists,
    // since for example `gcd(a_1, ..., a_n)` might not divide a constant term.
    let step_sig = step_sig.iter().map(|i| *i as i32).collect::<Vec<_>>();
    let iter_of_iters = equave
        .0
        .into_iter()
        .map(|expon| solve_linear_diophantine(&step_sig, expon, exponent_bound).into_iter());
    let prod = iter_of_iters.multi_cartesian_product();
    let zipped: Vec<Vec<Vec<i32>>> = prod.map(|step_soln| multi_zip(&step_soln)).collect();
    let result: Vec<Vec<Vec<i32>>> = zipped
        .into_iter()
        .filter(|steps| {
            steps
                .iter()
                .all(|step| step.iter().all(|expon| expon.abs() <= exponent_bound))
        })
        .collect();
    let result: Vec<Vec<Monzo>> = result
        .into_iter()
        .map(|vs| vs.iter().map(|v| Monzo::from_slice(v)).collect::<Vec<_>>())
        .filter(|soln| soln.iter().all(|&step| step.is_positive()))
        .collect();
    result
        .into_iter()
        .filter(|soln| is_sorted_strictly_desc(soln))
        .collect::<Vec<_>>()
}

fn multi_zip<T>(vecs: &[Vec<T>]) -> Vec<Vec<T>>
where
    T: Copy,
{
    let truncate_to_this_len = vecs.iter().map(|vec| vec.len()).min().unwrap_or(0);
    (0..truncate_to_this_len)
        .map(|i| vecs.iter().map(|vec| vec[i]).collect::<Vec<_>>())
        .collect()
}

impl Sum for Monzo {
    fn sum<I: Iterator<Item = Monzo>>(iter: I) -> Self {
        iter.fold(Monzo::UNISON, |x, y| x + y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ji_ratio::RawJiRatio;

    #[test]
    fn test_monzo_macro() {
        assert_eq!(monzo![], Monzo::UNISON);
        let syntonic_comma = monzo![-4, 4, -1];
        assert_eq!(81, syntonic_comma.numer());
        assert_eq!(80, syntonic_comma.denom());
        let jacobin = monzo![9, 0, -1, 0, -3, 1];
        assert_eq!(6656, jacobin.numer());
        assert_eq!(6655, jacobin.denom());
    }
    #[test]
    fn test_diophantine_homogeneous() {
        let solutions_constant_0 = solve_linear_diophantine_homogeneous(&[5, 2], 4);
        assert!(solutions_constant_0.contains(&vec![2, -5]));
        let solutions_constant_0 = solve_linear_diophantine_homogeneous(&[5, 2, 3], 4);
        assert!(solutions_constant_0.contains(&vec![-2, -1, 4]));
        assert!(solutions_constant_0.contains(&vec![1, -1, -1]));
    }
    #[test]
    fn test_diophantine() {
        let solutions_constant_0 = solve_linear_diophantine(&[5, 2], 1, 4);
        assert!(solutions_constant_0.contains(&vec![-3, 8]));
        let solutions_constant_0 = solve_linear_diophantine(&[5, 2, 3], 0, 4);
        assert!(solutions_constant_0.contains(&vec![-2, -1, 4]));
        assert!(solutions_constant_0.contains(&vec![1, -1, -1]));
        let solutions_constant_1 = solve_linear_diophantine(&[5, 2, 3], 1, 4);
        assert!(solutions_constant_1.contains(&vec![1, 4, -4]));
    }

    #[test]
    fn test_try_from_ratio() {
        let syntonic_comma = RawJiRatio::try_new(81, 80).unwrap();
        let result_81_80 = Monzo::try_from_ratio(syntonic_comma);
        assert_eq!(result_81_80, Ok(monzo![-4, 4, -1]));
        /*
            //  The following returns `Err` only when SMALL_PRIMES.len() <= 6
            let s17 = RawJiRatio::try_new(289, 288).unwrap();
            let result_s17 = Monzo::try_from_ratio(s17);
            assert_eq!(
                result_s17,
                Err(CantMakeMonzo::NumerExceededPrimeLimit(vec![17, 17]))
            );
            let s17_inv = RawJiRatio::try_new(288, 289).unwrap();
            let result_s17 = Monzo::try_from_ratio(s17_inv);
            assert_eq!(
                result_s17,
                Err(CantMakeMonzo::DenomExceededPrimeLimit(vec![17, 17]))
            );

        */
    }

    #[test]
    fn test_try_to_ratio() {
        let monzo_81_80 = monzo![-4, 4, -1];
        let result_ratio = monzo_81_80.try_to_ratio();
        assert_eq!(result_ratio, Some(RawJiRatio::try_new(81, 80).unwrap()));
    }

    #[test]
    fn test_ord_for_monzo() {
        let monzo_9_8 = monzo![-3, 2];
        let monzo_28_27 = monzo![2, -3, 0, 1];
        assert!(monzo_28_27 < monzo_9_8);
    }
}
