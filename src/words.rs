//! Scale representation as sequences of step letters.
//!
//! A scale is represented as a word over an alphabet of step letters (L, m, s, etc.),
//! where each letter represents a step size class. For ternary scales:
//! - `0` = L (large step)
//! - `1` = m (medium step)
//! - `2` = s (small step)
//!
//! # Core Types
//!
//! - [`Letter`]: Type alias for step letters (`usize`)
//! - [`CountVector<T>`]: Multiset of elements, used for step signatures and interval classes
//! - [`Chirality`]: Scale symmetry classification (Left/Achiral/Right)
//!
//! # Key Operations
//!
//! - [`rotate`]: Rotate a scale word (change mode)
//! - [`least_mode`]: Find lexicographically smallest rotation (canonical form)
//! - [`maximum_variety`]: Compute the maximum variety of a scale
//! - [`chirality`]: Determine scale handedness
//! - [`is_monotone_mos`]: Check monotone-MOS conditions
//!
//! # Examples
//!
//! ```
//! use tern::words::{rotate, least_mode, maximum_variety, chirality, Chirality};
//!
//! // Represent the diatonic scale as a word: 5 large steps, 2 small steps
//! let lydian = vec![0, 0, 0, 1, 0, 0, 1];  // L L L s L L s
//!
//! // Rotate to get different modes
//! let mixolydian = rotate(&lydian, 1);          // L L s L L L s
//!
//! // Find canonical form (lexicographically first mode)
//! let canonical = least_mode(&lydian);
//! assert_eq!(canonical, vec![0, 0, 0, 1, 0, 0, 1]);  // Lydian is already canonical
//!
//! // MOS scales have maximum variety 2
//! assert_eq!(maximum_variety(&lydian), 2);
//!
//! // Diatonic is achiral (equal to its reversal)
//! assert_eq!(chirality(&lydian), Chirality::Achiral);
//! ```

use itertools::Itertools;
use serde::Serialize;
use std::cmp::{Ordering, max};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::hash::Hash;

use crate::helpers::{ScaleError, gcd, modinv};

/// A step letter representing a step size class.
///
/// In ternary scales: `0` = L (large), `1` = m (medium), `2` = s (small).
pub type Letter = usize;

/// Types such that when you put them into vectors, it makes sense to interpret the vectors as scales and take intervals from these scales.
pub trait Subtendable: Clone + Send + Sync + Sized {
    type Interval: Send;

    fn interval_from_slice(slice: &[Self]) -> Self::Interval;

    fn dyad_on_degree(scale: &[Self], degree: usize) -> Self::Interval {
        Self::interval_from_slice(&rotate(scale, degree))
    }
}

impl Subtendable for Letter {
    type Interval = CountVector<Letter>;
    fn interval_from_slice(slice: &[Self]) -> Self::Interval {
        CountVector::from_slice(slice)
    }
}

impl Subtendable for CountVector<Letter> {
    type Interval = CountVector<Letter>;
    fn interval_from_slice(slice: &[Self]) -> Self::Interval {
        slice
            .iter()
            .fold(CountVector::ZERO, |a, b| CountVector::add(&a, b))
    }
}

/// The [chirality](https://en.xen.wiki/w/Chirality) (handedness) of a scale.
///
/// Compares a scale to its reversal to determine symmetry.
///
/// # Examples
///
/// ```
/// use tern::words::{chirality, Chirality};
///
/// // Achiral: equal to its reversal (as a circular word)
/// let diatonic = [0, 0, 0, 1, 0, 0, 1];
/// assert_eq!(chirality(&diatonic), Chirality::Achiral);
///
/// // Chiral scales have distinct left/right forms
/// let right_handed = [0, 1, 0, 2, 0, 1, 0, 2, 0];  // diasem
/// assert_eq!(chirality(&right_handed), Chirality::Right);
/// ```
#[derive(Copy, Clone, Debug, Hash, PartialEq, Serialize)]
pub enum Chirality {
    /// Scale word > reversed word (lexicographically, in canonical form).
    Left,
    /// Scale equals its reversal as a circular word.
    Achiral,
    /// Scale word < reversed word (lexicographically, in canonical form).
    Right,
}

/// A multiset (bag) of elements, implemented as a map from elements to counts.
///
/// Used to represent:
/// - **Step signatures**: e.g., "5L 2s" as `{0: 5, 1: 2}`
/// - **Interval classes**: the step content of a scale interval
///
/// Supports group operations (addition, negation, scalar multiplication).
///
/// # Examples
///
/// ```
/// use tern::words::CountVector;
///
/// // Create from a slice (counts occurrences)
/// let steps = CountVector::from_slice(&[0, 0, 0, 1, 0, 0, 1]);
/// assert_eq!(steps.get(&0), Some(&5));  // 5 large steps
/// assert_eq!(steps.get(&1), Some(&2));  // 2 small steps
///
/// // Total step count
/// assert_eq!(steps.len(), 7);
/// ```
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CountVector<T>(BTreeMap<T, i32>);

impl<T> CountVector<T> {
    /// Wraps BTreeMap<T, i32> in `CountVector<T>`.
    pub fn from_btree_map(m: BTreeMap<T, i32>) -> Self {
        CountVector(m)
    }
    /// Creates a zero count vector.
    pub const ZERO: Self = CountVector(BTreeMap::<T, i32>::new());

    /// Whether the `CountVector` is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sum of the absolute values of the components.
    pub fn len(&self) -> usize {
        self.0.values().map(|v| v.unsigned_abs() as usize).sum()
    }
    /// The sum of two count vectors. Each component gets added.
    pub fn add(&self, w: &Self) -> Self
    where
        T: Ord + Clone,
    {
        let mut result = self.0.clone();
        for (key, value) in w.0.iter() {
            match result.get_mut(key) {
                Some(update_this) => {
                    *update_this += value;
                    if *update_this == 0 {
                        result.remove(key);
                    }
                }
                _ => {
                    result.insert((*key).clone(), *value);
                }
            }
        }
        Self(result)
    }
    /// Additive inverse of a `CountVector`.
    pub fn neg(&self) -> Self
    where
        T: Ord + Clone + Send + Sync,
    {
        Self(self.0.iter().map(|(k, v)| (k.clone(), -v)).collect())
    }

    /// Multiply a CountVector by a scalar.
    pub fn scalar_mul(&self, lambda: i32) -> Self
    where
        T: Ord + Clone + Send + Sync,
    {
        if lambda == 0 {
            // Zeroed components are not allowed!
            Self::ZERO
        } else {
            Self(
                self.0
                    .iter()
                    .map(|(k, v)| (k.clone(), lambda * v))
                    .collect(),
            )
        }
    }

    /// Convert a slice to a CountVector.
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Ord + Clone,
    {
        let mut result = BTreeMap::new();
        for key in slice {
            match result.get_mut(key) {
                Some(update_this) => {
                    *update_this += 1;
                }
                _ => {
                    result.insert((*key).clone(), 1);
                }
            }
        }
        Self(result)
    }
    /// Convert a `BTreeSet` to a `CountVector` (with every nonzero component equal to 1).
    pub fn from_btree_set(set: BTreeSet<T>) -> CountVector<T>
    where
        T: Ord + Send,
    {
        Self(
            set.into_iter()
                .map(|t| (t, 1))
                .collect::<BTreeMap<T, i32>>(),
        )
    }
    /// Convert a `BTreeSet` to a `CountVector` (with every nonzero component equal to 1).
    pub fn from_tuples(iter: impl Iterator<Item = (T, i32)>) -> CountVector<T>
    where
        T: Ord + Send,
    {
        Self::from_btree_map(BTreeMap::from_iter(iter))
    }
    /// Unwrap the underlying `BTreeMap`.
    pub fn into_inner(&self) -> BTreeMap<T, i32>
    where
        T: Clone,
    {
        self.clone().0
    }
    /// The multiset of `CountVector`s that occur as `subword_length`-steps in `scale`.
    pub fn spectrum(scale: &[T], subword_length: usize) -> CountVector<CountVector<T>>
    where
        T: Ord + Clone,
    {
        let mut result: BTreeMap<CountVector<T>, i32> = BTreeMap::new();
        for key in (0..scale.len())
            .map(|degree| CountVector::from_slice(&word_on_degree(scale, degree, subword_length)))
        {
            match result.get_mut(&key) {
                Some(update_this) => {
                    *update_this += 1;
                }
                _ => {
                    result.insert(key, 1);
                }
            }
        }
        CountVector(result)
    }

    /// The set of `CountVector`s that occur as `subword_length`-steps in `scale`.
    pub fn distinct_spectrum(scale: &[T], subword_length: usize) -> HashSet<CountVector<T>>
    where
        T: Hash + Ord + Clone + Send + Sync,
    {
        (0..scale.len())
            .map(|degree| CountVector::from_slice(&word_on_degree(scale, degree, subword_length)))
            .collect::<HashSet<_>>()
    }
    /// Get the component for the specified key.
    pub fn get(&self, arg: &T) -> Option<&i32>
    where
        T: Ord,
    {
        self.0.get(arg)
    }
    /// Get the number of keys.
    pub fn keys_count(&self) -> usize {
        self.0.len()
    }
}

pub fn countvector_to_slice(v: CountVector<usize>) -> Vec<i32> {
    if v.is_empty() {
        vec![]
    } else if let Some(max_key_value) = v.into_inner().last_key_value() {
        let max = max_key_value.0;
        let mut result = vec![0; max + 1];
        for key in v.into_inner().keys() {
            if let Some(value) = v.get(key) {
                result[*key] = *value;
            } else {
                return vec![];
            }
        }
        result
    } else {
        vec![]
    }
}

/// Treating `scale` as a circular string,
/// take a slice of length `subword_length` from `degree`; assumes `subword_length` <= `scale`.len().
/// Reduce `degree` first.
pub fn word_on_degree<T>(scale: &[T], degree: usize, subword_length: usize) -> Vec<T>
where
    T: Clone,
{
    let rotated = rotate(scale, degree);
    if subword_length < scale.len() {
        Vec::<_>::from(&rotated[..subword_length])
    } else {
        let prefix_for_div = rotated
            .iter()
            .cloned()
            .cycle()
            .take(subword_length / scale.len())
            .collect::<Vec<_>>();
        let suffix_for_rem = &rotated[0..subword_length];
        [&prefix_for_div, suffix_for_rem].concat()
    }
}

/// The dyad on the specified degree of the scale as a `CountVector`.
pub fn dyad_on_degree<T>(scale: &[T], degree: usize, interval_class: usize) -> CountVector<T>
where
    T: Ord + Clone,
{
    CountVector::from_slice(&word_on_degree(scale, degree, interval_class))
}

#[allow(unused)]
fn strand_on_degree<T>(
    scale: &[T],
    degree: usize,
    interval_class: usize,
) -> Result<Vec<CountVector<T>>, ScaleError>
where
    T: Ord + Clone + Send + Sync,
{
    if scale.len().is_multiple_of(interval_class) {
        Ok((0..(scale.len() / interval_class))
            .map(|i| dyad_on_degree(scale, degree + i * interval_class, interval_class))
            .collect())
    } else {
        Err(ScaleError::NonDivisibleSubsetError)
    }
}

/// Return the offset of vec2 to the right relative to vec1 if the Vecs are conjugate, otherwise None.
pub fn offset_vec<T>(vec1: &[T], vec2: &[T]) -> Option<usize>
where
    T: PartialEq + std::clone::Clone,
{
    if vec1.len() != vec2.len() {
        // if lengths are not equal, cannot be conjugate
        None
    } else {
        let len = vec1.len();
        for i in 0..len {
            let vec2_rotated = rotate(vec2, i); // vec2 has been rotated i steps to the left.
            if vec2_rotated == vec1 {
                return Some(i);
            }
        }
        None
    }
}

/// Computes the [maximum variety](https://en.xen.wiki/w/Maximum_variety) of a scale.
///
/// Maximum variety counts the largest number of distinct interval classes
/// at any single interval size. MOS scales have MV=2.
///
/// # Examples
///
/// ```
/// use tern::words::maximum_variety;
///
/// // MOS scales have maximum variety 2
/// let diatonic = [0, 0, 1, 0, 0, 0, 1];
/// assert_eq!(maximum_variety(&diatonic), 2);
///
/// // Ternary scales have higher MV
/// let diasem = [0, 1, 0, 2, 0, 1, 0, 2, 0];
/// assert_eq!(maximum_variety(&diasem), 3);
/// let blackdye = [2, 0, 1, 0, 2, 0, 1, 0, 2, 0];
/// assert_eq!(maximum_variety(&blackdye), 4);
/// ```
pub fn maximum_variety<T>(scale: &[T]) -> usize
where
    T: Hash + Ord + Clone + Sync + Send,
{
    let mut result = 1; // variety for 0-steps and periods
    let floor_half: usize = scale.len() / 2;
    for subword_length in 1..(floor_half + 1) {
        let sizes = CountVector::distinct_spectrum(scale, subword_length);
        result = max(result, sizes.len()); // update result
    }
    result
}

/// Says whether this scale has the given [maximum variety](https://en.xen.wiki/w/Maximum_variety).
/// Faster than comparing the output of `maximum_variety` because of short-circuiting.
///
/// Maximum variety counts the largest number of distinct interval classes
/// at any single interval size. MOS scales have MV=2.
///
/// # Examples
///
/// ```
/// use tern::words::maximum_variety_is;
///
/// // MOS scales have maximum variety 2
/// let diatonic = [0, 0, 1, 0, 0, 0, 1];
/// assert!(maximum_variety_is(&diatonic, 2));
///
/// // Ternary scales have higher MV
/// let diasem = [0, 1, 0, 2, 0, 1, 0, 2, 0];
/// assert!(maximum_variety_is(&diasem, 3));
/// let blackdye = [2, 0, 1, 0, 2, 0, 1, 0, 2, 0];
/// assert!(maximum_variety_is(&blackdye, 4));
/// ```
pub fn maximum_variety_is<T>(scale: &[T], mv: usize) -> bool
where
    T: Hash + Ord + Clone + Sync + Send,
{
    let mut result = 1; // variety for 0-steps and periods
    let floor_half: usize = scale.len() / 2;
    for subword_length in 1..(floor_half + 1) {
        let sizes = CountVector::distinct_spectrum(scale, subword_length);
        result = max(result, sizes.len()); // update result
        if result > mv {
            return false;
        }
    }
    result == mv
}

/// Whether `scale` is strict variety (the variety is the same for every non-equave step class).
pub fn is_strict_variety<T>(scale: &[T]) -> bool
where
    T: Hash + Ord + Clone + Sync + Send,
{
    let mut prev: usize = 0;
    let floor_half: usize = scale.len() / 2;
    for subword_length in 1..=floor_half {
        let sizes: HashSet<CountVector<T>> = CountVector::distinct_spectrum(scale, subword_length);
        if prev == 0 {
            prev = sizes.len();
        } else if prev != sizes.len() {
            return false;
        }
    }
    true
}

/// Return the [block balance](https://en.xen.wiki/w/Balanced_word) of `s`.
pub fn block_balance<T>(scale: &[T]) -> usize
where
    T: Hash + Ord + Clone + PartialEq + Send + Sync,
{
    if scale.len() <= 1 {
        scale.len()
    } else {
        // Only need to check half of the step size classes.
        let floor_half: usize = scale.len() / 2;
        let distinct_letters = scale.iter().cloned().collect::<BTreeSet<T>>();
        let maybe_max = (1..=floor_half)
            .flat_map(|subword_length| {
                distinct_letters.iter().map(move |letter| {
                    let counts = CountVector::distinct_spectrum(scale, subword_length)
                        .iter()
                        .filter_map(|dyad| dyad.get(letter))
                        .copied()
                        .collect::<BTreeSet<_>>();
                    // the differences to collect
                    *counts.last().expect("`counts` should be nonempty")
                        - *counts.first().expect("`counts` should be nonempty") // this will always be >= 0 for a nonempty `BTreeSet`
                })
            })
            .max();
        if let Some(max) = maybe_max {
            max as usize
        } else {
            usize::MAX
        }
    }
}

/// Return the brightest mode of the MOS axby and the bright generator, using the Bresenham line algorithm.
/// Bjorklund's algorithm is asymptotically faster, but this Bresenham implementation is faster for
/// practical MOS sizes.
pub fn brightest_mos_mode_and_gener_bresenham(
    a: usize,
    b: usize,
) -> (Vec<Letter>, CountVector<Letter>) {
    let d = gcd(a as u32, b as u32) as usize;
    if d == 1 {
        let count_gen_steps = modinv(b as i32, (a + b) as i32)
                .expect("The dark generator is a (|s|⁻¹ mod |scale|)-step, since stacking it |s| times results in the L step (mod period).")
               as usize;
        let mut result_scale: Vec<usize> = vec![];
        let (mut current_x, mut current_y) = (0usize, 0usize); // Start from the (0, 0) and walk until the dark generator is reached; we now know how many steps to walk.
        while current_x < a || current_y < b {
            if a * (current_y + 1) <= b * current_x {
                // If going north (making a (0, 1) step) doesn't lead to going above the line y == b/a*x,
                current_y += 1; // append the y step and reflect the change in the plane vector.
                result_scale.push(1);
            } else {
                // Else, make a (1, 0) step.
                current_x += 1;
                result_scale.push(0);
            }
        }
        let result_gen = CountVector::from_slice(&result_scale[0..count_gen_steps]);
        (result_scale, result_gen)
    } else {
        let (prim_mos, gener) = brightest_mos_mode_and_gener_bresenham(a / d, b / d);
        (prim_mos.repeat(d), gener)
    }
}

/// Return the brightest mode of the MOS aLbs and the bright generator, using Bjorklund's algorithm.
/// The brightest mode is the lexicographically first rotation.
pub fn brightest_mos_mode_and_gener_bjorklund(
    a: usize,
    b: usize,
) -> (Vec<Letter>, CountVector<Letter>) {
    let d = gcd(a as u32, b as u32) as usize;
    if d == 1 {
        // The bright generator is a (b⁻¹ mod |scale|)-step, since stacking it `b` times results in the L step (mod period).
        let count_gener_steps = modinv(b as i32, a as i32 + b as i32)
            .expect("Should be ok because gcd(a + b, b) == gcd(a, b) == 1")
            as usize;
        // These are the seed strings we build the brightest MOS word from.
        // The algorithm uses two subwords at each step, iteratively appending the
        // lexicographically second subword to the lexicographically first subword to ensure
        // that the lexicographically first mode is returned.
        let (mut first, mut second) = (vec![0], vec![1]);
        let (mut count_first, mut count_second) = (a, b); // aLbs(0) is the brightest mode of bsaL with L and s swapped.
        while count_second != 1 {
            // Possibly after switching, are there more copies of `first` than `second`?
            // Then all the `second`s get appended to the first `count_second` copies of `first`s,
            // and the new `second`s are the remaining copies of `first`s.
            let old_first = first.clone();
            first.extend_from_slice(&second);
            if count_first > count_second {
                second = old_first;
                (count_first, count_second) = (count_second, count_first - count_second);
            }
            // Otherwise, there are strictly fewer `first`s than `second`s (as gcd(a, b) == 1),
            // and *all* the `first`s get modified, whereas `second` is unchanged since copies of it remain.
            // `count_first` is also unchanged.
            else {
                count_second -= count_first;
            }
            // At the current step we have `count_first` `first` substrings and `count_second` `second` substrings,
            // where we must guarantee that `first < second`.
            // Thus if `first > second`, then swap them and swap the count variables.
            // Do this step before checking the while condition; we know the desired lex. ordering holds for the first step,
            // and our stopping condition requires that `first < second` actually hold to really behave correctly.
            if first > second {
                (first, second) = (second, first);
                (count_first, count_second) = (count_second, count_first);
            }
        }
        // At the end, we have `count_first` `first`s and 1 `second`,
        // so return (`first`)^`count_first` `second` (in standard mathematical word notation).
        let mut scale: Vec<usize> = first.repeat(count_first);
        scale.extend_from_slice(&second);
        // The bright generator is the first `count_gener_steps` of the scale.
        let gener = CountVector::from_slice(&scale[0..count_gener_steps]);
        (scale, gener)
    } else {
        let (primitive_mos, gener) = brightest_mos_mode_and_gener_bjorklund(a / d, b / d);
        (primitive_mos.repeat(d), gener)
    }
}

/// The mode of the MOS aLbs with a given brightness (count of bright generators up from root).
/// Brightness is taken modulo (a + b), so any non-negative value is valid.
/// Brightness 0 returns the darkest mode, brightness (a + b - 1) returns the brightest mode.
pub fn mos_mode(a: usize, b: usize, brightness: usize) -> Vec<Letter> {
    let scale_len = a + b;
    let brightness = brightness % scale_len;
    // Bresenham is faster for practical sizes
    let (mos, bright_gener) = brightest_mos_mode_and_gener_bresenham(a, b);
    let bright_gener_step_count: usize = bright_gener.len();
    // Rotate backwards from brightest mode by `(scale_len - 1 - brightness)` bright generators
    // which is equivalent to rotating forward by `brightness` dark generators from darkest mode
    let steps_from_brightest = (scale_len - 1 - brightness) * bright_gener_step_count;
    rotate(&mos, steps_from_brightest)
}

/// Rotate a scale word left by `degree` positions (change mode).
///
/// # Examples
///
/// ```
/// use tern::words::rotate;
///
/// let lydian = vec![0, 0, 0, 1, 0, 0, 1];
/// let mixolydian = rotate(&lydian, 1);  // Rotate left by 1
/// assert_eq!(mixolydian, vec![0, 0, 1, 0, 0, 1, 0]);
/// ```
pub fn rotate<T: std::clone::Clone>(slice: &[T], degree: usize) -> Vec<T> {
    let degree = degree % slice.len();
    if degree == 0 {
        slice.to_vec()
    } else {
        [&slice[degree..slice.len()], &slice[0..degree]].concat()
    }
}

/// Whether two slices with elements of type T are rotationally equivalent.
pub fn rotationally_equivalent<T>(s1: &[T], s2: &[T]) -> bool
where
    T: Clone + Eq,
{
    (s1.len() == s2.len()) && { (0..s1.len()).any(|i| rotate(s1, i) == s2.to_vec()) }
}

/// The lexicographically least mode of a word (where the letters are in their usual order).
///
/// # Examples
///
/// ```
/// use tern::words::least_mode;
///
/// let ionian = vec![0, 0, 1, 0, 0, 0, 1]; // LLsLLLs
/// let lydian = vec![0, 0, 0, 1, 0, 0, 1]; // LLLsLLs (lexicographically least mode of diatonic)
/// assert_eq!(least_mode(&ionian), lydian)
/// ```
pub fn least_mode(scale: &[Letter]) -> Vec<Letter> {
    rotate(scale, booth(scale))
}

/// The rotation required from the current word to the
/// lexicographically least mode of a word.
/// Booth's algorithm requires at most 3*n* comparisons and *n* storage locations where *n* is the input word's length.
/// See Booth, K. S. (1980). Lexicographically least circular substrings.
/// Information Processing Letters, 10(4-5), 240–242. doi:10.1016/0020-0190(80)90149-0
pub fn booth(scale: &[Letter]) -> usize {
    let scale_len = scale.len();
    // `failure_func` is the failure function of the least rotation; `usize::MAX` is used as a null value.
    // null indicates that the failure function does not point backwards in the string.
    // `usize::MAX` will behave the same way as -1 does, assuming wrapping unsigned addition
    let mut failure_func = vec![usize::MAX; 2 * scale_len];
    let mut least_rotation: usize = 0;
    // `scan_pos` loops over `scale` twice.
    for scan_pos in 1..2 * scale_len {
        let mut match_len = failure_func[scan_pos - least_rotation - 1];
        while match_len != usize::MAX
            && scale[scan_pos % scale_len]
                != scale[least_rotation.wrapping_add(match_len).wrapping_add(1) % scale_len]
        {
            // (1) If the scan_pos-th letter is less than s[(least_rotation + match_len + 1) % scale_len] then change least_rotation to scan_pos - match_len - 1,
            // in effect left-shifting the failure function and the input string.
            // This appropriately compensates for the new, shorter least substring.
            if scale[scan_pos % scale_len]
                < scale[least_rotation.wrapping_add(match_len).wrapping_add(1) % scale_len]
            {
                least_rotation = scan_pos.wrapping_sub(match_len).wrapping_sub(1);
            }
            match_len = failure_func[match_len];
        }
        if match_len == usize::MAX
            && scale[scan_pos % scale_len]
                != scale[least_rotation.wrapping_add(match_len).wrapping_add(1) % scale_len]
        {
            // See note (1) above.
            if scale[scan_pos % scale_len]
                < scale[least_rotation.wrapping_add(match_len).wrapping_add(1) % scale_len]
            {
                least_rotation = scan_pos;
            }
            failure_func[scan_pos - least_rotation] = usize::MAX;
        } else {
            failure_func[scan_pos - least_rotation] = match_len.wrapping_add(1);
        }
        // The induction hypothesis is that
        // at this point `failure_func[0..scan_pos - least_rotation]` is the failure function of `s[least_rotation..(least_rotation+scan_pos)%scale_len]`,
        // and `least_rotation` is the lexicographically least subword of the letters scanned so far.
    }
    least_rotation
}

/// [Letterwise substitution](https://en.xen.wiki/w/MOS_substitution) for scale words.
/// Note: This function does not fail even if the number of times `x` occurs in `template`
/// does not divide `filler.len()`.
///
/// # Examples
///
/// ```
/// use tern::words::subst;
///
/// let template = vec![0, 1, 1, 0, 1, 1, 1];
/// let filler = vec![1, 2];
/// let subst_scale = subst(&template, 1, &filler);
/// assert_eq!(subst_scale, vec![0, 1, 2, 0, 1, 2, 1]);
/// ```
///
pub fn subst(template: &[Letter], x: Letter, filler: &[Letter]) -> Vec<Letter> {
    let mut ret = vec![];
    let mut i: usize = 0;
    if !filler.is_empty() {
        for &letter in template {
            if letter == x {
                // Use the currently pointed-to letter of `filler` in place of `x`.
                ret.push(filler[i % filler.len()]);
                // Only update `i` when an `x` is replaced.
                i += 1;
            } else {
                ret.push(letter);
            }
        }
    } else {
        // If `filler` is empty, we return `template` but with all `x`s removed.
        return delete(template, x);
    }
    ret
}

/// Return the collection of all MOS substitution scales `subst n0 x (n1 y n2 z)`
/// where the template MOS is assumed to have step signature `n0*0 (n1 + n2)*X` (`X` is the slot letter)
/// and the filling MOS has step signature `n1*1 n2*2`.
///
/// # Examples
///
/// ```
/// use tern::words::mos_substitution_scales_one_perm;
///
/// let only_contains_one_scale = mos_substitution_scales_one_perm(6, 5, 5);
/// assert_eq!(only_contains_one_scale.len(), 1);
/// ```
pub fn mos_substitution_scales_one_perm(n0: usize, n1: usize, n2: usize) -> Vec<Vec<Letter>> {
    let (template, _) = brightest_mos_mode_and_gener_bresenham(n0, n1 + n2);
    let (filler, gener) = brightest_mos_mode_and_gener_bresenham(n1, n2);
    let filler = filler.into_iter().map(|x| x + 1).collect::<Vec<_>>();
    let gener_size = gener.len();
    let redundant_list: Vec<_> = (0..(n1 + n2))
        .map(|i| {
            subst(
                &template,
                1usize,
                &rotate(&filler, (i * gener_size) % filler.len()),
            )
        })
        .collect();
    // Canonicalize every scale and remove duplicates
    redundant_list
        .into_iter()
        .map(|scale| least_mode(&scale))
        .sorted()
        .dedup()
        .collect()
}

/// The set of all [MOS substitution](https://en.xen.wiki/w/User:Inthar/MOS_substitution) ternary scales
/// with the given step signature `sig`.
pub fn mos_substitution_scales(sig: &[usize]) -> Vec<Vec<Letter>> {
    let (n0, n1, n2) = (sig[0], sig[1], sig[2]);

    // Only need 3 permutations of (0, 1, 2) for the MOS substitution patterns n0*_ (n1*_ n2*_)
    let redundant_list = [
        // n0L (n1m n2s)
        mos_substitution_scales_one_perm(n0, n1, n2),
        // n1m (n0L n2s)
        mos_substitution_scales_one_perm(n1, n2, n0)
            .into_iter()
            .map(|scale| scale.into_iter().map(|x| (x + 1) % 3).collect())
            .collect(),
        // n2s (n0L n1m)
        mos_substitution_scales_one_perm(n2, n0, n1)
            .into_iter()
            .map(|scale| {
                scale
                    .into_iter()
                    .map(|x| if x == 0 { 2 } else { (x - 1) % 3 })
                    .collect()
            })
            .collect(),
    ]
    .concat();
    // Canonicalize every scale and remove duplicates
    redundant_list
        .into_iter()
        .map(|scale| least_mode(&scale))
        .sorted()
        .dedup()
        .collect()
}

/// Whether `scale` is a MOS substitution scale with any choice of letter as template letter.
///
/// # Examples
///
/// ```
/// use tern::words::{Letter, is_mos_subst};
///
/// let blackdye: Vec<Letter> = vec![2, 0, 1, 0, 2, 0, 1, 0, 2, 0]; // sLmLsLmLsL
/// assert!(is_mos_subst(&blackdye));
///
/// let nonexample: Vec<Letter> = vec![1, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2, 0, 0, 0];
/// assert!(!is_mos_subst(&nonexample));
/// ```
pub fn is_mos_subst(scale: &[Letter]) -> bool {
    let steps: Vec<_> = step_set(scale).into_iter().collect();
    steps.len() == 3 && {
        let (x, y, z) = (steps[0], steps[1], steps[2]);
        mos_subst_helper(scale, x, y, z)
            || mos_subst_helper(scale, y, x, z)
            || mos_subst_helper(scale, z, x, y)
    }
}

/// Whether `scale` is a MOS substitution scale subst at(bf1 cf2).
///
/// # Examples
///
/// ```
/// use tern::words::{Letter, is_mos_subst_one_perm};
///
/// let blackdye: Vec<Letter> = vec![2, 0, 1, 0, 2, 0, 1, 0, 2, 0]; // sLmLsLmLsL
/// assert!(is_mos_subst_one_perm(&blackdye, 0, 1, 2)); // XLXLXLXLXL is a MOS pattern
/// assert!(!is_mos_subst_one_perm(&blackdye, 1, 0, 2)); // XXmXXXmXXX is not a MOS pattern
/// ```
pub fn is_mos_subst_one_perm(scale: &[Letter], t: Letter, f1: Letter, f2: Letter) -> bool {
    step_variety(scale) == 3 // Is it ternary?
        && mos_subst_helper(scale, t, f1, f2)
}

// Helper for checking MOS substitution property assuming `scale` is already ternary with the given
fn mos_subst_helper(scale: &[Letter], t: Letter, f1: Letter, f2: Letter) -> bool {
    maximum_variety_is(&delete(scale, t), 2) // Is the result of deleting t a MOS?
        && maximum_variety_is(&replace(scale, f1, f2), 2) // Is the result of identifying letters of the filling MOS a MOS
}

/// Return the number of distinct steps in `scale`.
pub fn step_set(scale: &[Letter]) -> BTreeSet<usize> {
    scale.iter().copied().collect::<BTreeSet<_>>()
}

/// Return the number of distinct steps in `scale`.
pub fn step_variety(scale: &[Letter]) -> usize {
    step_set(scale).len()
}

/// `subst()` but the filler is just one letter.
pub fn replace(scale: &[Letter], from: Letter, to: Letter) -> Vec<Letter> {
    subst(scale, from, &[to])
}

/// Delete all instances of one letter.
pub fn delete(scale: &[Letter], letter: Letter) -> Vec<Letter> {
    scale.iter().filter(|x| **x != letter).cloned().collect()
}

/// If `scale` is ternary, return whether identifying L = m, m = s, and s = 0 results in a MOS.
/// Returns `false` if the scale is not ternary.
///
/// # Examples
///
/// ```
/// use tern::words::{Letter, is_monotone_mos};
///
/// let diasem_2sr = [0, 1, 0, 2, 0, 1, 0, 2, 0]; // LmLsLmLsL
/// assert!(is_monotone_mos(&diasem_2sr)); // LXLXLXLXL, XXXsXXXsX, LmLLmLL are all MOSes
///
/// let blackdye = [2, 0, 1, 0, 2, 0, 1, 0, 2, 0]; // sLmLsLmLsL
/// assert!(!is_monotone_mos(&blackdye)); // sXXXsXXXsX, X = L~m, is not a MOS pattern
/// ```
pub fn is_monotone_mos(scale: &[Letter]) -> bool {
    step_variety(scale) == 3
        && maximum_variety_is(&replace(scale, 1, 0), 2) // L = m
        && maximum_variety_is(&replace(scale, 2, 1), 2) // m = s
        && maximum_variety_is(&delete(scale, 2), 2) // s = 0
}

/// Check if the result of equating L = m is a MOS. Assumes the scale is ternary.
pub fn monotone_lm(scale: &[Letter]) -> bool {
    maximum_variety_is(&replace(scale, 1, 0), 2)
}

/// Check if the result of equating m = s is a MOS. Assumes the scale is ternary.
pub fn monotone_ms(scale: &[Letter]) -> bool {
    maximum_variety_is(&replace(scale, 2, 1), 2)
}

/// Check if the result of equating s = 0 is a MOS. Assumes the scale is ternary.
pub fn monotone_s0(scale: &[Letter]) -> bool {
    maximum_variety_is(&delete(scale, 2), 2)
}

/// Check if pairiwse identifications of two of the step sizes always results in a MOS.
/// Returns `false` if the scale is not ternary.
///
/// # Examples
///
/// ```
/// use tern::words::{Letter, is_pairwise_mos};
///
/// let diasem_2sr = [0, 1, 0, 2, 0, 1, 0, 2, 0]; // LmLsLmLsL
/// assert!(is_pairwise_mos(&diasem_2sr)); // LXLXLXLXL, XXXsXXXsX, XmXXXmXXX are all MOSes
///
/// let blackdye = [2, 0, 1, 0, 2, 0, 1, 0, 2, 0]; // sLmLsLmLsL
/// assert!(!is_pairwise_mos(&blackdye)); // sXXXsXXXsX, X = L~m, is not a MOS pattern
/// ```
pub fn is_pairwise_mos(scale: &[Letter]) -> bool {
    step_variety(scale) == 3
        && maximum_variety_is(&replace(scale, 1, 0), 2)
        && maximum_variety_is(&replace(scale, 1, 2), 2)
        && maximum_variety_is(&replace(scale, 2, 0), 2)
}

/// The repeating portion of a word.
///
/// # Examples
///
/// ```
/// use tern::words::{Letter, period_pattern};
///
/// let diminished = [0, 1, 0, 1, 0, 1, 0, 1]; // LsLsLsLs
/// assert_eq!(period_pattern::<Letter>(&diminished), vec![0, 1]);
///
/// let diasem_2sr = [0, 1, 0, 2, 0, 1, 0, 2, 0]; // LmLsLmLsL
/// assert_eq!(period_pattern::<Letter>(&diasem_2sr), diasem_2sr);
/// ```
pub fn period_pattern<T>(word: &[T]) -> Vec<T>
where
    T: PartialEq + Clone + Send + Sync,
{
    for divisor in 1..=word.len() / 2 {
        if word.len().is_multiple_of(divisor) {
            // Prefix length must divide word.len(); check both divisor and, if divisor > 1, prefix.len() / divisor
            let prefix1_repeated: Vec<_> = word
                .iter()
                .take(divisor)
                .cycle()
                .take(word.len())
                .cloned()
                .collect(); // Repeat prefix the appropriate number of times
            if word.to_vec() == prefix1_repeated {
                return word[..divisor].to_vec();
            }
        }
    }
    // If all divisors fail then return the whole word
    word.to_vec()
}

/// The minimal prefix `x` such that `word` is a prefix of `x^\infty`.
///
/// # Examples
///
/// ```
/// use tern::words::{Letter, weak_period_pattern};
///
/// let diminished = [0, 1, 0, 1, 0, 1, 0, 1]; // LsLsLsLs
/// assert_eq!(weak_period_pattern::<Letter>(&diminished), vec![0, 1]);
///
/// let diasem_2sr = [0, 1, 0, 2, 0, 1, 0, 2, 0]; // LmLsLmLsL
/// assert_eq!(weak_period_pattern::<Letter>(&diasem_2sr), vec![0, 1, 0, 2]);
/// ```
pub fn weak_period_pattern<T>(word: &[T]) -> Vec<T>
where
    T: PartialEq + Clone + Send + Sync,
{
    let l = (1..word.len()) // Only check up to prefix_len == slice.len() - 1 since the check will succeed when prefix is the whole word
        .map(|prefix_len| word.iter().take(prefix_len).cycle()) // Get infinite repetition of each prefix
        .take_while(|prefix_cycle| {
            // Try each prefix until word == the word.len()-letter prefix of prefix_cycle
            word.to_vec()
                != prefix_cycle
                    .clone()
                    .take(word.len())
                    .cloned()
                    .collect::<Vec<T>>()
        })
        .collect::<Vec<_>>()
        .len()
        + 1; // Add 1 because the final letter of the prefix where the check succeeds is not appended to the iterator
    word[..l].to_vec()
}

/// The collection of rotations of a word, in cyclic order.
/// Contains redundant rotations if the word is not primitive.
///
/// # Examples
///
/// ```
/// use tern::words::{Letter, rotations};
///
/// let diatonic = [0, 0, 1, 0, 0, 0, 1]; // LLsLLLs
/// assert_eq!(rotations::<Letter>(&diatonic).len(), 7);
///
/// // Diminished scale is a mode of limited transposition
/// let diminished = [0, 1, 0, 1, 0, 1, 0, 1]; // LsLsLsLs
/// assert_eq!(rotations::<Letter>(&diminished).len(), 2);
pub fn rotations<T>(word: &[T]) -> Vec<Vec<T>>
where
    T: Clone + Eq + Send + Sync,
{
    let period = period_pattern(word).len();
    (0..period).map(|i| rotate(word, i)).collect()
}

/// The lexicographically determined chirality of a scale word.
///
/// # Examples
///
/// ```
/// use tern::words::{Chirality, chirality};
///
/// let diasem_2sr = vec![0, 1, 0, 2, 0, 1, 0, 2, 0]; // Reversing step order results in diasem_2sl
/// assert_eq!(chirality(&diasem_2sr), Chirality::Right);
///
/// let diasem_2sl = vec![0, 2, 0, 1, 0, 2, 0, 1, 0];
/// assert_eq!(chirality(&diasem_2sl), Chirality::Left);
///
/// let blackdye = vec![2, 0, 2, 0, 1, 0, 2, 0, 1, 0];
/// assert_eq!(chirality(&blackdye), Chirality::Achiral);
/// ```
pub fn chirality(word: &[Letter]) -> Chirality {
    let least_mode_word = least_mode(word);

    let word_rev: Vec<usize> = word.iter().copied().rev().collect();
    let least_mode_word_rev = least_mode(&word_rev);

    match least_mode_word.cmp(&least_mode_word_rev) {
        Ordering::Less => Chirality::Right,
        Ordering::Equal => Chirality::Achiral,
        Ordering::Greater => Chirality::Left,
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused)]
    use crate::helpers::gcd;

    use super::*;

    #[test]
    fn test_booth() {
        let blackdye = [2, 0, 1, 0, 2, 0, 1, 0, 2, 0];
        assert_eq!(least_mode(&blackdye), vec![0, 1, 0, 2, 0, 1, 0, 2, 0, 2]);
    }
    #[test]
    fn test_word_on_degree() {
        let blackdye = [2, 0, 1, 0, 2, 0, 1, 0, 2, 0];
        let even_two_steps = (0..5)
            .map(|i| word_on_degree(&blackdye, 2 * i, 2))
            .collect::<Vec<_>>();
        assert_eq!(
            even_two_steps,
            vec![vec![2, 0], vec![1, 0], vec![2, 0], vec![1, 0], vec![2, 0],]
        );
        let odd_two_steps = (0..5)
            .map(|i| word_on_degree(&blackdye, 2 * i + 1, 2))
            .collect::<Vec<_>>();
        assert_eq!(
            odd_two_steps,
            vec![vec![0, 1], vec![0, 2], vec![0, 1], vec![0, 2], vec![0, 2],]
        );
    }
    #[test]
    fn test_spectrum() {
        let diachrome_5sc = [0, 2, 0, 2, 0, 1, 2, 0, 2, 0, 2, 1];
        for i in 1..=11 {
            assert_eq!(
                // .len() of `CountVector` is the taxicab norm, not the number of keys
                CountVector::spectrum(&diachrome_5sc, i).into_inner().len(),
                CountVector::distinct_spectrum(&diachrome_5sc, i).len()
            );
        }
    }
    #[test]
    fn test_chirality() {
        let diasem_2sr = [0, 1, 0, 2, 0, 1, 0, 2, 0];
        let blackdye = [2, 0, 1, 0, 2, 0, 1, 0, 2, 0];
        let diasem_2sl = [0, 2, 0, 1, 0, 2, 0, 1, 0];

        assert_eq!(chirality(&diasem_2sr), Chirality::Right);
        assert_eq!(chirality(&blackdye), Chirality::Achiral);
        assert_eq!(chirality(&diasem_2sl), Chirality::Left);

        let diachrome_5sr = [0, 1, 2, 0, 2, 0, 2, 0, 1, 2, 0, 2];
        let diachrome_5sc = [0, 2, 0, 2, 0, 1, 2, 0, 2, 0, 2, 1];
        let diachrome_5sl = [0, 2, 1, 0, 2, 0, 2, 0, 2, 1, 0, 2];

        assert_eq!(chirality(&diachrome_5sr), Chirality::Right);
        assert_eq!(chirality(&diachrome_5sc), Chirality::Achiral);
        assert_eq!(chirality(&diachrome_5sl), Chirality::Left);
    }

    #[test]
    fn test_period() {
        let word_012 = [0, 1, 2, 0, 1, 2, 0, 1, 2];
        let pentawood = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let diasem = [0, 1, 0, 2, 0, 1, 0, 2, 0];
        let three_periods = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1];
        assert_eq!(period_pattern(&word_012).len(), 3);
        assert_eq!(period_pattern(&pentawood).len(), 2);
        assert_eq!(period_pattern(&diasem).len(), 9);
        assert_eq!(period_pattern(&three_periods).len(), 5);
    }
    #[test]
    fn test_weak_period_pattern() {
        let word_012 = [0, 1, 2, 0, 1, 2, 0, 1, 2];
        let pentawood = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let diasem = [0, 1, 0, 2, 0, 1, 0, 2, 0];
        assert_eq!(weak_period_pattern(&word_012).len(), 3);
        assert_eq!(weak_period_pattern(&pentawood).len(), 2);
        assert_eq!(weak_period_pattern(&diasem).len(), 4);
    }
    #[test]
    fn test_maximum_variety() {
        assert_eq!(maximum_variety(&[0, 0, 0, 0]), 1); // check even and odd
        assert_eq!(maximum_variety(&[1, 1, 1, 1, 1]), 1);
        assert_eq!(maximum_variety(&[0, 0, 0, 1, 0, 0, 1]), 2); // diatonic has max variety 2
        assert_eq!(maximum_variety(&[1, 1, 1, 1, 2, 1, 2]), 3); // altered diatonic has max variety 3
        assert_eq!(maximum_variety(&[0, 1, 0, 2, 0, 1, 0, 2, 0]), 3); // diasem has max variety 3
        assert_eq!(maximum_variety(&[0, 1, 0, 2, 0, 1, 0, 2, 0, 1]), 4); // blackdye has max variety 4
        // MOS scales should be MV2.
        for a in 1usize..=10 {
            for b in 1usize..=10 {
                if gcd(a as u32, b as u32) == 1 {
                    let (mos_bjork, gener_bjork) = brightest_mos_mode_and_gener_bjorklund(a, b);
                    assert_eq!(booth(&mos_bjork), 0); // MOS scales' brightest mode is indeed the least mode
                    assert_eq!(maximum_variety(&mos_bjork), 2);
                    let (mos_bres, gener_bres) = brightest_mos_mode_and_gener_bresenham(a, b);
                    assert_eq!(mos_bres, mos_bjork); // Bjorklund and Bresenham should agree
                    assert_eq!(gener_bres, gener_bjork); // Bright generators should agree
                }
            }
        }
    }

    #[test]
    fn test_mos_block_balanced() {
        // MOS scales should have block balance 1.
        for a in 3..=20 {
            for b in 3..=20 {
                if gcd(a, b) == 1 {
                    for br in 0..(a + b) / gcd(a, b) {
                        let mos = mos_mode(a as usize, b as usize, br as usize);
                        assert_eq!(block_balance(&mos), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn test_brightest_gener_of_mos() {
        let diatonic = brightest_mos_mode_and_gener_bresenham(5, 2);
        assert_eq!(diatonic.0, vec![0, 0, 0, 1, 0, 0, 1]);
        assert_eq!(diatonic.1.into_inner(), BTreeMap::from([(0, 3), (1, 1)]));
        let oneirotonic = brightest_mos_mode_and_gener_bresenham(5, 3);
        assert_eq!(oneirotonic.0, vec![0, 0, 1, 0, 0, 1, 0, 1]);
        assert_eq!(oneirotonic.1.into_inner(), BTreeMap::from([(0, 2), (1, 1)]));
    }
}
