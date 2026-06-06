//! Necklace enumeration using Sawada's algorithm.
//!
//! A **necklace** is an equivalence class of words under rotation. This module
//! enumerates all necklaces with a given "content" (step signature), which is
//! useful for generating all scales with a specific count of each step size.
//!
//! # Algorithm
//!
//! Uses Sawada (2003)'s efficient algorithm for generating necklaces with
//! fixed content. The algorithm generates each necklace exactly once in
//! lexicographically smallest (canonical) form.
//!
//! # Examples
//!
//! ```
//! use tern::comb::necklaces_fixed_content;
//!
//! // Generate all ternary scale patterns with 2L, 2m, 1s
//! let content = [2, 2, 1];  // 2 of letter 0, 2 of letter 1, 1 of letter 2
//! let necklaces = necklaces_fixed_content(&content);
//!
//! // Each necklace has length 5 (2+2+1)
//! for necklace in &necklaces {
//!     assert_eq!(necklace.len(), 5);
//! }
//! ```
//!
//! # References
//!
//! - Sawada, J. (2003). "A fast algorithm to generate necklaces with fixed content."
//!   Theoretical Computer Science, 301(1-3), 477-489.

use std::collections::BTreeSet;

use crate::helpers::{first_index_desc, first_index_smaller};
use crate::words::Letter;

/// Recursive helper for computing partitions with exact part count.
/// Generates all partitions of `n` using parts <= `m` with exactly `parts` parts.
fn partitions_exact_part_count_rec(n: usize, m: usize, parts: usize) -> Vec<Vec<usize>> {
    match (n, m, parts) {
        (0, 0, 0) => vec![vec![]],             // Base case: empty partition
        (0, m, _) if m > 0 => vec![],          // Invalid: sum is 0 but m > 0
        (0, _, k) if k > 0 => vec![],          // Invalid: sum is 0 but need k parts
        (n, m, _) if n > 0 && m > n => vec![], // Invalid: max part > sum
        (n, _, k) if n > 0 && k > n => vec![], // Invalid: more parts than sum
        (n, 0, _) if n > 0 => vec![],          // Invalid: no parts available
        (n, _, 0) if n > 0 => vec![],          // Invalid: need parts but none allowed
        _ => (0..=m)
            // Try each possible value for the first part
            .flat_map(|l| {
                partitions_exact_part_count_rec(n - m, l, parts - 1)
                    .into_iter()
                    .map(|partition| [vec![m], partition].concat())
            })
            .collect(),
    }
}

/// Return the collection of all partitions of `n` with exactly `k` parts.
pub fn partitions_exact_part_count(n: usize, parts: usize) -> Vec<Vec<usize>> {
    (1usize..=(n + 1).saturating_sub(parts))
        .flat_map(|m| partitions_exact_part_count_rec(n, m, parts))
        .collect()
}

/// A partition of n >= 0 is a (possibly empty) sorted list of positive summands to n.
pub fn partitions(n: usize) -> Vec<Vec<usize>> {
    (1usize..=n).flat_map(|m| partitions_rec(n, m)).collect()
}

/// Recursive helper for computing all partitions of n.
/// Generates partitions using parts of size m or smaller in descending order.
fn partitions_rec(n: usize, m: usize) -> Vec<Vec<usize>> {
    match (n, m) {
        (0, 0) => {
            vec![vec![]] // Base case: only the empty partition
        }
        (0, _) => {
            vec![] // No valid partitions when sum is 0 but m > 0
        }
        (n, 0) if n > 0 => {
            vec![] // No valid partitions with no parts available
        }
        (n, m) if m > n => {
            vec![] // No valid partitions if max part size exceeds sum
        }
        _ => (0..=m)
            // Try each part size from m down to 0
            .flat_map(|k| {
                partitions_rec(n - m, k)
                    .into_iter()
                    .map(|partition| [vec![m], partition].concat())
            })
            .collect(),
    }
}

/// Generate all necklaces (rotation-equivalence classes) with a given content.
///
/// Uses Sawada (2003)'s efficient algorithm. Each necklace is returned in
/// canonical form (lexicographically smallest rotation).
///
/// # Arguments
///
/// * `content` - Array where `content[i]` is the count of letter `i`
///
/// # Examples
///
/// ```
/// use tern::comb::necklaces_fixed_content;
///
/// // All necklaces with 3 zeros and 2 ones (binary necklaces)
/// let necklaces = necklaces_fixed_content(&[3, 2]);
/// assert_eq!(necklaces.len(), 2);  // 00011 and 00101
///
/// // Ternary: 2L, 1m, 1s
/// let ternary = necklaces_fixed_content(&[2, 1, 1]);
/// for n in &ternary {
///     assert_eq!(n.len(), 4);
/// }
/// ```
///
/// # Returns
///
/// Empty vector if all content values are zero.
pub fn necklaces_fixed_content(content: &[Letter]) -> Vec<Vec<Letter>> {
    if content.iter().all(|x| *x == 0) {
        vec![]
    } else {
        let (mut rem_content, perm) = VecPerm::sift_zeros(content);
        while *rem_content
            .last()
            .expect("`rem_content` is a permutation of a nonempty `content` that is not all 0's")
            == 0
        {
            rem_content.pop();
        }
        let arity = rem_content.len();
        rem_content[0] -= 1;
        let scale_len: usize = content.iter().sum();
        let mut word = vec![0];
        word.extend(&vec![arity - 1; scale_len - 1]);
        let mut avail_letters: Vec<usize>; // List containing available letters in reverse order
        if rem_content[0] == 0 {
            // Remove 0 if we no longer have one
            avail_letters = (1..arity).rev().collect();
        } else {
            avail_letters = (0..arity).rev().collect();
        }
        let mut coll: Vec<Vec<usize>> = vec![];
        sawada_rec(
            &mut rem_content,
            &mut vec![0; scale_len],
            &mut avail_letters,
            &mut word,
            1,
            1,
            1,
            &mut coll,
        );
        coll.shrink_to_fit();
        // Rename letters of the scale. We can use the same permutation, as it's a product of disjoint transpositions, thus order 2.
        coll = coll
            .into_iter()
            .map(|scale| {
                scale
                    .into_iter()
                    .map(|letter| {
                        perm.at(letter)
                            .expect("`perm` witnesses that `letter` was in the scale.")
                    })
                    .collect()
            })
            .collect();
        coll.shrink_to_fit();
        coll
    }
}

// Recursive part of algorithm in Sawada (2002)
#[allow(clippy::too_many_arguments)]
fn sawada_rec(
    remaining_content: &mut Vec<usize>, // Remaining content to add to the prenecklace (Sawada's n)
    max_suffix_runs: &mut Vec<usize>, // Run of consecutive (arity-1)s starting at each position (Sawada's r)
    avail_letters: &mut Vec<Letter>,  // Available letters, maintained in descending order
    prenecklace: &mut Vec<Letter>,    // Current prenecklace being built (Sawada's a)
    current_pos: usize,               // Current position in prenecklace (Sawada's t)
    lyndon_prefix_len: usize,         // Length of longest Lyndon prefix (Sawada's p)
    run_start: usize,                 // Start of current run of (arity-1)s (Sawada's s)
    results: &mut Vec<Vec<Letter>>,   // Accumulator for valid necklaces
) {
    let scale_len = prenecklace.len();
    let arity = remaining_content.len(); // TODO: Strip any suffix of 0's
    if remaining_content[arity - 1] == scale_len - current_pos {
        // if the only remaining letter is `arity - 1`
        if (remaining_content[arity - 1] == max_suffix_runs[current_pos - lyndon_prefix_len]
            && scale_len.is_multiple_of(lyndon_prefix_len))
            || remaining_content[arity - 1] > max_suffix_runs[current_pos - lyndon_prefix_len]
        {
            let mut new_necklace: Vec<usize> = prenecklace.iter_mut().map(|x| *x).collect();
            new_necklace.shrink_to_fit(); // remove any extra allocation
            results.push(new_necklace);
            results.shrink_to_fit();
        } // else reject
    } else if remaining_content[0] != scale_len - current_pos {
        // else reject since it both begins and ends in a 0
        let maybe_first_letter: Option<usize> = avail_letters.first().copied();
        if let Some(letter) = maybe_first_letter {
            let mut current_letter = letter;
            while current_letter >= prenecklace[current_pos - lyndon_prefix_len] {
                max_suffix_runs[run_start] = current_pos - run_start;
                if remaining_content[current_letter] == 1 {
                    avail_letters.remove(first_index_desc(avail_letters, current_letter).expect(
                        "this is a bug; `avail_letters` should contain exactly the nonzero keys of `remaining_content`",
                    ));
                }
                remaining_content[current_letter] -= 1;
                prenecklace[current_pos] = current_letter;
                // Yield to caller before and after recursive call
                stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
                    sawada_rec(
                        remaining_content,
                        max_suffix_runs,
                        avail_letters,
                        prenecklace,
                        current_pos + 1,
                        if current_letter == prenecklace[current_pos - lyndon_prefix_len] {
                            lyndon_prefix_len
                        } else {
                            current_pos + 1
                        },
                        if current_letter == arity - 1 {
                            run_start
                        } else {
                            current_pos + 1
                        },
                        results,
                    )
                });
                // If current_letter has been removed from `avail_letters`, add it back.
                // This is how we backtrack in the tree.
                if remaining_content[current_letter] == 0 {
                    if let Some(insert_pos) = first_index_smaller(avail_letters, current_letter) {
                        avail_letters.insert(insert_pos, current_letter);
                    } else {
                        avail_letters.push(current_letter);
                    }
                }
                remaining_content[current_letter] += 1;
                // update current_letter for next iteration
                if let Some(next_idx) = first_index_smaller(avail_letters, current_letter) {
                    current_letter = avail_letters[next_idx];
                } else {
                    break;
                }
            }
        }
        prenecklace[current_pos] = arity - 1;
    }
}

#[derive(PartialEq, Hash, Debug, Clone)]
/// Error types for invalid `VecPerm` construction
pub enum PermutationError {
    /// When two perms are composed, their lengths must match
    DiffLengths,
    /// Index out of bounds for permutation domain
    IndexOutOfBounds(usize, usize),
    /// The image is not {0, ..., n - 1} (not a valid permutation)
    WrongImage(Vec<usize>),
}

impl std::fmt::Display for PermutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DiffLengths => {
                write!(f, "permutations must have matching lengths")
            }
            Self::IndexOutOfBounds(len, i) => {
                write!(f, "permutation has length {len} but the index is {i}")
            }
            Self::WrongImage(sl) => {
                write!(f, "wrong image for `VecPerm`: {sl:?}")
            }
        }
    }
}

/// Encodes how the entries of a `Vec` was permuted.
/// `pi[x]` stores where the permutation sends element `x`.
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct VecPerm {
    pi: Vec<usize>, // `pi[x]` is where the permutation sends `x`
}

impl std::fmt::Display for VecPerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.pi)
    }
}

impl VecPerm {
    /// Attempts to get the value that the permutation maps `index` to.
    /// Returns a `PermutationError` if `index` is out of bounds.
    pub fn at(&self, index: usize) -> Result<usize, PermutationError> {
        if index < self.len() {
            Ok(self.pi[index])
        } else {
            Err(PermutationError::IndexOutOfBounds(self.len(), index))
        }
    }
    /// Check if permutation is empty (domain size 0).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Tries to create a new `VecPerm` from a slice.
    /// Validates that the image equals {0, ..., n-1}.
    pub fn try_new(slice: &[usize]) -> Result<VecPerm, PermutationError> {
        let domain = (0..slice.len()).collect::<BTreeSet<_>>();
        let image = slice.iter().cloned().collect::<BTreeSet<_>>();
        if slice.is_empty() {
            // return the empty perm
            Ok(VecPerm { pi: vec![] })
        } else if image != domain {
            // not a valid perm if image != domain
            let mut image_as_vec = image.into_iter().collect::<Vec<_>>();
            image_as_vec.sort();
            image_as_vec.dedup();
            Err(PermutationError::WrongImage(image_as_vec))
        } else if slice.len() != 1 + *image.last().expect("we handled the empty slice case") {
            // if image is the same set as domain but some value corresponds to multiple indices
            Err(PermutationError::DiffLengths)
        } else {
            // valid nonempty `VecPerm`
            Ok(VecPerm { pi: slice.to_vec() })
        }
    }
    #[inline]
    /// The number of items the permutation is on.
    pub fn len(&self) -> usize {
        self.pi.len()
    }
    /// The identity element of S_n (permutation that maps each element to itself).
    #[inline]
    pub fn id(len: usize) -> VecPerm {
        VecPerm {
            pi: (0..len).collect::<Vec<_>>(),
        }
    }
    /// The permutation `(i j)` in cycle notation (the identity if `i == j`).
    /// Swaps elements at positions i and j.
    pub fn transposition(len: usize, i: usize, j: usize) -> Self {
        if i > len {
            return Self::id(len);
        }
        if j > len {
            return Self::id(len);
        }
        VecPerm {
            pi: {
                let mut v = (0..len).collect::<Vec<_>>();
                v.swap(i, j);
                v
            },
        }
    }
    /// Returns the composite `self \circ other`, mapping each `k` to `self`(`other`(`k`)).
    /// `self` and `other` must have the same domain, or this will return a `PermutationError`.
    pub fn o(&self, other: &Self) -> Result<Self, PermutationError> {
        if self.len() == other.len() {
            Ok(VecPerm {
                pi: (other.pi).iter().map(|&k| self.pi[k]).collect(),
            })
        } else {
            Err(PermutationError::DiffLengths)
        }
    }
    /// The inverse of a permutation (the permutation pi^-1 such that pi ∘ pi^-1 = id).
    #[allow(clippy::needless_range_loop)]
    pub fn inv(&self) -> Self {
        let mut pi = vec![usize::MAX; self.len()];
        for i in 0..self.len() {
            'inner: for j in 0..self.len() {
                if self.pi[j] == i {
                    pi[i] = j;
                    break 'inner;
                }
            }
        }
        VecPerm { pi }
    }
    /// Conjugate a permutation `g` by another permutation `h`.
    /// The value returned by `g.conj(&h)` is `hgh^{-1}` in mathematical notation,
    /// which is a permutation that "permutes the elements like `g` does, but as remapped by `h`".
    /// For example, if `g` acts on `0, 1, 2` by cycling through them,
    /// `g.conj(&h)` acts on `h(0)`, `h(1)`, and `h(2)` in the same way.
    /// Conjugation preserves the cycle structure of the permutation.
    pub fn conj(&self, other: &Self) -> Result<Self, PermutationError> {
        other.o(self)?.o(&other.inv())
    }

    /// Replaces elements of vector `v` starting from `index` with elements from `other` if possible.
    /// Helper function for necklace generation algorithm.
    fn nest_replacing(v: &[usize], other: &[usize], index: usize) -> Result<Vec<usize>, String> {
        if other.is_empty() {
            Ok(v.to_owned())
        } else if index + other.len() - 1 < v.len() {
            let mut result = [&v[0..index], other].concat().to_vec();
            if index + other.len() < v.len() {
                result.extend_from_slice(&v[index + other.len()..]);
            }
            Ok(result)
        } else {
            Err("index out of bounds in `nest_replacing`".into())
        }
    }
    /// Permute all the 0's in the input to the end of the vector,
    /// and return the modified vector and the overall permutation used.
    /// Used to normalize content vectors by removing trailing zeros.
    pub fn sift_zeros(slice: &[usize]) -> (Vec<usize>, VecPerm) {
        match slice.len() {
            0 | 1 => (slice.to_vec(), VecPerm::id(slice.len())), // A slice of length 0 or 1 is always valid.
            _ => {
                let mut vec = slice.to_vec();
                let mut perm = VecPerm::id(slice.len());
                let mut seek_0: usize = 0; // This index looks for the first 0 from the front.
                let mut seek_non0: usize = slice.len() - 1; // This index looks for the first non-0 from the end.
                while seek_0 < vec.len() - 1 && vec[seek_0] != 0 {
                    seek_0 += 1;
                }
                while vec[seek_non0] == 0 && seek_non0 > 0 {
                    seek_non0 -= 1;
                }
                if seek_0 < seek_non0 {
                    // If `slice` has a `0` and it's before a non-`0` element, swap 'em.
                    vec.swap(seek_0, seek_non0);
                    let p = VecPerm::transposition(vec.len(), seek_0, seek_non0).o(&perm);
                    // Multiply by the transposition.
                    if let Ok(p1) = p {
                        perm = p1;
                    } else {
                        return (slice.to_vec(), Self::id(slice.len()));
                    }
                } else {
                    // Early return, nothing to do.
                    return (vec, perm);
                }
                let (mut inner_slice, mut inner_perm) = (vec![], VecPerm::id(0));
                stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
                    (inner_slice, inner_perm) =
                        Self::sift_zeros(&slice[(seek_0 + 1)..=(seek_non0 - 1)]);
                });
                if let Ok(vec) = Self::nest_replacing(&vec, &inner_slice, seek_0 + 1) {
                    let p = Self::nest_replacing(
                        &perm.pi,
                        &inner_perm
                            .pi
                            .into_iter()
                            .map(|x| x + seek_0 + 1)
                            .collect::<Vec<_>>(),
                        seek_0 + 1,
                    );
                    if let Ok(perm) = p {
                        (vec, VecPerm { pi: perm })
                    } else {
                        (slice.to_vec(), Self::id(slice.len()))
                    }
                } else {
                    (slice.to_vec(), Self::id(slice.len()))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::BTreeSet, iter::FromIterator};

    #[test]
    fn test_necklace_generation_with_zeros_in_content() {
        let content = [1usize, 1, 0, 0, 1];
        let attempt: BTreeSet<Vec<usize>> = BTreeSet::from_iter(necklaces_fixed_content(&content));
        let correct_scales = vec![vec![0, 1, 4], vec![0, 4, 1]];
        let correct_result: BTreeSet<Vec<usize>> = BTreeSet::from_iter(correct_scales);
        assert_eq!(attempt, correct_result);
        let content = [5usize, 0, 0, 0, 2];
        let attempt: BTreeSet<Vec<usize>> = BTreeSet::from_iter(necklaces_fixed_content(&content));
        let correct_result: BTreeSet<Vec<usize>> = BTreeSet::from_iter(vec![
            vec![0, 0, 0, 4, 0, 0, 4],
            vec![0, 0, 0, 0, 4, 0, 4],
            vec![0, 0, 0, 0, 0, 4, 4],
        ]);
        assert_eq!(attempt, correct_result);
    }
    #[test]
    fn test_sift_zeros_base_cases() {
        let empty: Vec<usize> = vec![];
        let zero_singleton = vec![0usize];
        let nonzero_singleton = vec![1usize];
        assert_eq!((vec![], VecPerm::id(0)), VecPerm::sift_zeros(&empty));
        assert_eq!(
            (vec![0], VecPerm::id(1)),
            VecPerm::sift_zeros(&zero_singleton)
        );
        assert_eq!(
            (vec![1], VecPerm::id(1)),
            VecPerm::sift_zeros(&nonzero_singleton)
        );
    }
    #[test]
    fn test_sift_zeros_len_2() {
        let vec_01: Vec<usize> = vec![0, 1];
        let correct_perm = VecPerm::try_new(&[1, 0]).unwrap();
        assert_eq!((vec![1, 0], correct_perm), VecPerm::sift_zeros(&vec_01));
    }
    #[test]
    fn test_sift_zeros_bigger_cases() {
        let vec_102030: Vec<usize> = vec![1, 0, 2, 0, 3, 0];
        let correct_perm = VecPerm::try_new(&[0, 4, 2, 3, 1, 5]).unwrap();
        assert_eq!(
            (vec![1, 3, 2, 0, 0, 0], correct_perm),
            VecPerm::sift_zeros(&vec_102030)
        );
        let long_boi: Vec<usize> = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let correct_perm = VecPerm::try_new(&[9, 8, 7, 6, 4, 5, 3, 2, 1, 0]).unwrap();
        assert_eq!(
            (vec![1, 1, 1, 1, 1, 1, 0, 0, 0, 0], correct_perm),
            VecPerm::sift_zeros(&long_boi)
        );
        let long_boi: Vec<usize> = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
        let correct_perm = VecPerm::try_new(&[9, 8, 7, 6, 4, 5, 3, 2, 1, 0]).unwrap();
        assert_eq!(
            (vec![1, 1, 1, 1, 0, 0, 0, 0, 0, 0], correct_perm),
            VecPerm::sift_zeros(&long_boi)
        );
        let long_boi: Vec<usize> = vec![0, 0, 1, 0, 0, 0, 1, 1, 1, 1];
        let correct_perm = VecPerm::try_new(&[9, 8, 2, 7, 6, 5, 4, 3, 1, 0]).unwrap();
        assert_eq!(
            (vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0], correct_perm),
            VecPerm::sift_zeros(&long_boi)
        );
    }
    #[test]
    fn test_nest_replacing() {
        let vec = vec![1, 2, 3];
        let empty = vec![];
        let len1 = vec![6];
        let len2 = vec![4, 5];
        assert_eq!(VecPerm::nest_replacing(&vec, &empty, 1), Ok(vec![1, 2, 3]));
        assert_eq!(VecPerm::nest_replacing(&vec, &len1, 0), Ok(vec![6, 2, 3]));
        assert_eq!(VecPerm::nest_replacing(&vec, &len1, 1), Ok(vec![1, 6, 3]));
        assert_eq!(VecPerm::nest_replacing(&vec, &len1, 2), Ok(vec![1, 2, 6]));
        assert_eq!(VecPerm::nest_replacing(&vec, &len2, 0), Ok(vec![4, 5, 3]));
        assert_eq!(VecPerm::nest_replacing(&vec, &len2, 1), Ok(vec![1, 4, 5]));
        assert!(VecPerm::nest_replacing(&vec, &len2, 2).is_err());
    }
    #[test]
    fn test_perm() {
        let perm_210: VecPerm = VecPerm::try_new(&[2, 1, 0]).unwrap(); // order-2 permutation
        let id = VecPerm::id(3);
        let should_be_id = perm_210.o(&perm_210).unwrap();
        assert_eq!(should_be_id, id);
        let nother_perm_210 = VecPerm::transposition(3, 0, 2);
        assert_eq!(perm_210, nother_perm_210);
        let perm_201 = VecPerm::try_new(&[2, 0, 1]).unwrap(); // order-3 permutation
        let perm_120 = VecPerm::try_new(&[1, 2, 0]).unwrap();
        assert_eq!(perm_201.o(&perm_201).unwrap(), perm_120);
        assert_eq!(perm_201.o(&perm_201).unwrap().o(&perm_201).unwrap(), id);
        assert_eq!(perm_210.inv(), perm_210);
        assert_eq!(perm_120.inv(), perm_201);
    }
    #[test]
    fn test_perm_conjugation() {
        // The perm (0 1)(2 3)
        let perm_1032 = VecPerm::try_new(&[1, 0, 3, 2]).unwrap();
        // The perm (0 1 2 3)
        let perm_1230 = VecPerm::try_new(&[1, 2, 3, 0]).unwrap();
        // This should be (1 2)(3 0)
        let conjugate = perm_1032.conj(&perm_1230).unwrap();
        let perm_3210 = VecPerm::try_new(&[3, 2, 1, 0]).unwrap();
        assert_eq!(conjugate, perm_3210);
    }
    #[test]
    fn test_partition_with_given_part_count() {
        assert_eq!(Vec::<Vec<usize>>::new(), partitions_exact_part_count(2, 3));
        // Should return empty vec if # of parts is too large
    }
    #[test]
    fn test_sawada_225() {
        let attempt: BTreeSet<Vec<usize>> =
            BTreeSet::from_iter(necklaces_fixed_content(&[2, 2, 5]));
        let correct_scales = vec![
            vec![0, 2, 2, 1, 1, 0, 2, 2, 2],
            vec![0, 2, 2, 1, 0, 2, 2, 2, 1],
            vec![0, 2, 2, 1, 0, 2, 2, 1, 2],
            vec![0, 2, 2, 0, 2, 2, 2, 1, 1],
            vec![0, 2, 2, 0, 2, 2, 1, 2, 1],
            vec![0, 2, 2, 0, 2, 2, 1, 1, 2],
            vec![0, 2, 1, 2, 2, 1, 0, 2, 2],
            vec![0, 2, 1, 2, 2, 0, 2, 2, 1],
            vec![0, 2, 1, 2, 1, 2, 0, 2, 2],
            vec![0, 2, 1, 2, 1, 0, 2, 2, 2],
            vec![0, 2, 1, 2, 0, 2, 2, 2, 1],
            vec![0, 2, 1, 2, 0, 2, 2, 1, 2],
            vec![0, 2, 1, 2, 0, 2, 1, 2, 2],
            vec![0, 2, 1, 1, 2, 2, 0, 2, 2],
            vec![0, 2, 1, 1, 2, 0, 2, 2, 2],
            vec![0, 2, 1, 1, 0, 2, 2, 2, 2],
            vec![0, 2, 1, 0, 2, 2, 2, 2, 1],
            vec![0, 2, 1, 0, 2, 2, 2, 1, 2],
            vec![0, 2, 1, 0, 2, 2, 1, 2, 2],
            vec![0, 2, 1, 0, 2, 1, 2, 2, 2],
            vec![0, 2, 0, 2, 2, 2, 2, 1, 1],
            vec![0, 2, 0, 2, 2, 2, 1, 2, 1],
            vec![0, 2, 0, 2, 2, 2, 1, 1, 2],
            vec![0, 2, 0, 2, 2, 1, 2, 2, 1],
            vec![0, 2, 0, 2, 2, 1, 2, 1, 2],
            vec![0, 2, 0, 2, 2, 1, 1, 2, 2],
            vec![0, 2, 0, 2, 1, 2, 2, 2, 1],
            vec![0, 2, 0, 2, 1, 2, 2, 1, 2],
            vec![0, 2, 0, 2, 1, 2, 1, 2, 2],
            vec![0, 2, 0, 2, 1, 1, 2, 2, 2],
            vec![0, 1, 2, 2, 2, 2, 1, 0, 2],
            vec![0, 1, 2, 2, 2, 2, 0, 2, 1],
            vec![0, 1, 2, 2, 2, 1, 2, 0, 2],
            vec![0, 1, 2, 2, 2, 1, 0, 2, 2],
            vec![0, 1, 2, 2, 2, 0, 2, 2, 1],
            vec![0, 1, 2, 2, 2, 0, 2, 1, 2],
            vec![0, 1, 2, 2, 1, 2, 2, 0, 2],
            vec![0, 1, 2, 2, 1, 2, 0, 2, 2],
            vec![0, 1, 2, 2, 1, 0, 2, 2, 2],
            vec![0, 1, 2, 2, 0, 2, 2, 2, 1],
            vec![0, 1, 2, 2, 0, 2, 2, 1, 2],
            vec![0, 1, 2, 2, 0, 2, 1, 2, 2],
            vec![0, 1, 2, 2, 0, 1, 2, 2, 2],
            vec![0, 1, 2, 1, 2, 2, 2, 0, 2],
            vec![0, 1, 2, 1, 2, 2, 0, 2, 2],
            vec![0, 1, 2, 1, 2, 0, 2, 2, 2],
            vec![0, 1, 2, 1, 0, 2, 2, 2, 2],
            vec![0, 1, 2, 0, 2, 2, 2, 2, 1],
            vec![0, 1, 2, 0, 2, 2, 2, 1, 2],
            vec![0, 1, 2, 0, 2, 2, 1, 2, 2],
            vec![0, 1, 2, 0, 2, 1, 2, 2, 2],
            vec![0, 1, 2, 0, 1, 2, 2, 2, 2],
            vec![0, 1, 1, 2, 2, 2, 2, 0, 2],
            vec![0, 1, 1, 2, 2, 2, 0, 2, 2],
            vec![0, 1, 1, 2, 2, 0, 2, 2, 2],
            vec![0, 1, 1, 2, 0, 2, 2, 2, 2],
            vec![0, 1, 1, 0, 2, 2, 2, 2, 2],
            vec![0, 1, 0, 2, 2, 2, 2, 2, 1],
            vec![0, 1, 0, 2, 2, 2, 2, 1, 2],
            vec![0, 1, 0, 2, 2, 2, 1, 2, 2],
            vec![0, 1, 0, 2, 2, 1, 2, 2, 2],
            vec![0, 1, 0, 2, 1, 2, 2, 2, 2],
            vec![0, 1, 0, 1, 2, 2, 2, 2, 2],
            vec![0, 0, 2, 2, 2, 2, 2, 1, 1],
            vec![0, 0, 2, 2, 2, 2, 1, 2, 1],
            vec![0, 0, 2, 2, 2, 2, 1, 1, 2],
            vec![0, 0, 2, 2, 2, 1, 2, 2, 1],
            vec![0, 0, 2, 2, 2, 1, 2, 1, 2],
            vec![0, 0, 2, 2, 2, 1, 1, 2, 2],
            vec![0, 0, 2, 2, 1, 2, 2, 2, 1],
            vec![0, 0, 2, 2, 1, 2, 2, 1, 2],
            vec![0, 0, 2, 2, 1, 2, 1, 2, 2],
            vec![0, 0, 2, 2, 1, 1, 2, 2, 2],
            vec![0, 0, 2, 1, 2, 2, 2, 2, 1],
            vec![0, 0, 2, 1, 2, 2, 2, 1, 2],
            vec![0, 0, 2, 1, 2, 2, 1, 2, 2],
            vec![0, 0, 2, 1, 2, 1, 2, 2, 2],
            vec![0, 0, 2, 1, 1, 2, 2, 2, 2],
            vec![0, 0, 1, 2, 2, 2, 2, 2, 1],
            vec![0, 0, 1, 2, 2, 2, 2, 1, 2],
            vec![0, 0, 1, 2, 2, 2, 1, 2, 2],
            vec![0, 0, 1, 2, 2, 1, 2, 2, 2],
            vec![0, 0, 1, 2, 1, 2, 2, 2, 2],
            vec![0, 0, 1, 1, 2, 2, 2, 2, 2],
        ];
        let correct_result: BTreeSet<Vec<usize>> = BTreeSet::from_iter(correct_scales);
        assert_eq!(attempt, correct_result);
    }

    #[test]
    fn test_sawada_213() {
        let attempt: BTreeSet<Vec<usize>> =
            BTreeSet::from_iter(necklaces_fixed_content(&[2, 1, 3]));
        let correct_scales = vec![
            vec![0, 0, 1, 2, 2, 2],
            vec![0, 0, 2, 1, 2, 2],
            vec![0, 0, 2, 2, 1, 2],
            vec![0, 0, 2, 2, 2, 1],
            vec![0, 1, 0, 2, 2, 2],
            vec![0, 1, 2, 0, 2, 2],
            vec![0, 1, 2, 2, 0, 2],
            vec![0, 2, 0, 2, 1, 2],
            vec![0, 2, 0, 2, 2, 1],
            vec![0, 2, 1, 0, 2, 2],
        ];
        let correct_result: BTreeSet<Vec<usize>> = BTreeSet::from_iter(correct_scales);
        assert_eq!(attempt, correct_result);
    }

    #[test]
    fn test_partitions_7() {
        let attempt = BTreeSet::<Vec<_>>::from_iter(partitions(7));
        let correct_set = BTreeSet::from_iter(vec![
            vec![7],
            vec![6, 1],
            vec![5, 2],
            vec![5, 1, 1],
            vec![4, 3],
            vec![4, 2, 1],
            vec![4, 1, 1, 1],
            vec![3, 3, 1],
            vec![3, 2, 2],
            vec![3, 2, 1, 1],
            vec![3, 1, 1, 1, 1],
            vec![2, 2, 2, 1],
            vec![2, 2, 1, 1, 1],
            vec![2, 1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1, 1, 1],
        ]);
        assert_eq!(attempt, correct_set);
    }

    #[test]
    fn test_partitions_8() {
        let attempt = BTreeSet::<Vec<_>>::from_iter(partitions(8));
        let correct_set = BTreeSet::from_iter(vec![
            vec![8],
            vec![7, 1],
            vec![6, 2],
            vec![6, 1, 1],
            vec![5, 3],
            vec![5, 2, 1],
            vec![5, 1, 1, 1],
            vec![4, 4],
            vec![4, 3, 1],
            vec![4, 2, 2],
            vec![4, 2, 1, 1],
            vec![4, 1, 1, 1, 1],
            vec![3, 3, 2],
            vec![3, 3, 1, 1],
            vec![3, 2, 2, 1],
            vec![3, 2, 1, 1, 1],
            vec![3, 1, 1, 1, 1, 1],
            vec![2, 2, 2, 2],
            vec![2, 2, 2, 1, 1],
            vec![2, 2, 1, 1, 1, 1],
            vec![2, 1, 1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1, 1, 1, 1],
        ]);
        assert_eq!(attempt, correct_set);
    }

    #[test]
    fn test_sawada_5() {
        let attempt: BTreeSet<Vec<usize>> = BTreeSet::from_iter(necklaces_fixed_content(&[5]));
        let correct_set: BTreeSet<Vec<usize>> = BTreeSet::from_iter(vec![vec![0, 0, 0, 0, 0]]);
        assert_eq!(attempt, correct_set);
    }

    #[test]
    fn test_sawada_52() {
        let attempt: BTreeSet<Vec<usize>> = BTreeSet::from_iter(necklaces_fixed_content(&[5, 2]));
        let correct_set: BTreeSet<Vec<usize>> = BTreeSet::from_iter(vec![
            vec![0, 0, 0, 1, 0, 0, 1],
            vec![0, 0, 0, 0, 1, 0, 1],
            vec![0, 0, 0, 0, 0, 1, 1],
        ]);
        assert_eq!(attempt, correct_set);
    }

    #[test]
    fn test_sawada_1() {
        let attempt: BTreeSet<Vec<usize>> = BTreeSet::from_iter(necklaces_fixed_content(&[1]));
        let correct_set: BTreeSet<Vec<usize>> = BTreeSet::from_iter(vec![vec![0]]);
        assert_eq!(attempt, correct_set);
    }

    #[test]
    fn test_sawada_111() {
        let attempt: BTreeSet<Vec<usize>> =
            BTreeSet::from_iter(necklaces_fixed_content(&[1, 1, 1]));
        let correct_set: BTreeSet<Vec<usize>> =
            BTreeSet::from_iter(vec![vec![0, 1, 2], vec![0, 2, 1]]);
        assert_eq!(attempt, correct_set);
    }

    #[test]
    fn test_sawada_321() {
        let attempt: BTreeSet<Vec<usize>> =
            BTreeSet::from_iter(necklaces_fixed_content(&[3, 2, 1]));
        let correct_set: BTreeSet<Vec<usize>> = BTreeSet::from_iter(vec![
            vec![0, 1, 0, 1, 0, 2],
            vec![0, 0, 2, 1, 0, 1],
            vec![0, 0, 2, 0, 1, 1],
            vec![0, 0, 1, 2, 0, 1],
            vec![0, 0, 1, 1, 0, 2],
            vec![0, 0, 1, 0, 2, 1],
            vec![0, 0, 1, 0, 1, 2],
            vec![0, 0, 0, 2, 1, 1],
            vec![0, 0, 0, 1, 2, 1],
            vec![0, 0, 0, 1, 1, 2],
        ]);
        assert_eq!(attempt, correct_set);
    }

    #[test]
    fn test_sawada_231() {
        let attempt: BTreeSet<Vec<usize>> =
            BTreeSet::from_iter(necklaces_fixed_content(&[2, 3, 1]));
        let correct_set: BTreeSet<Vec<usize>> = BTreeSet::from_iter(vec![
            vec![0, 1, 1, 1, 0, 2],
            vec![0, 1, 1, 0, 2, 1],
            vec![0, 1, 1, 0, 1, 2],
            vec![0, 1, 0, 2, 1, 1],
            vec![0, 1, 0, 1, 2, 1],
            vec![0, 1, 0, 1, 1, 2],
            vec![0, 0, 2, 1, 1, 1],
            vec![0, 0, 1, 2, 1, 1],
            vec![0, 0, 1, 1, 2, 1],
            vec![0, 0, 1, 1, 1, 2],
        ]);
        assert_eq!(attempt, correct_set);
    }

    /*
    #[test]
    fn test_sawada_1111111111() {
        let attempt: BTreeSet<Vec<usize>> = BTreeSet::from_iter(
            necklaces_fixed_content(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).into_iter(),
        );
        assert_eq!(attempt.len(), 362880); // (arity - 1)!
    }
    */

    #[test]
    fn test_sawada_523() {
        let attempt: BTreeSet<Vec<usize>> =
            BTreeSet::from_iter(necklaces_fixed_content(&[5, 2, 3]));
        let correct_set: BTreeSet<Vec<usize>> = BTreeSet::from_iter(vec![
            vec![0, 1, 0, 2, 0, 1, 0, 2, 0, 2],
            vec![0, 1, 0, 1, 0, 2, 0, 2, 0, 2],
            vec![0, 0, 2, 2, 0, 2, 0, 1, 0, 1],
            vec![0, 0, 2, 2, 0, 1, 0, 2, 0, 1],
            vec![0, 0, 2, 2, 0, 1, 0, 1, 0, 2],
            vec![0, 0, 2, 1, 0, 2, 0, 2, 0, 1],
            vec![0, 0, 2, 1, 0, 2, 0, 1, 0, 2],
            vec![0, 0, 2, 1, 0, 1, 0, 2, 0, 2],
            vec![0, 0, 2, 1, 0, 1, 0, 0, 2, 2],
            vec![0, 0, 2, 1, 0, 0, 2, 2, 0, 1],
            vec![0, 0, 2, 1, 0, 0, 2, 1, 0, 2],
            vec![0, 0, 2, 0, 2, 2, 0, 1, 0, 1],
            vec![0, 0, 2, 0, 2, 1, 0, 2, 0, 1],
            vec![0, 0, 2, 0, 2, 1, 0, 1, 0, 2],
            vec![0, 0, 2, 0, 2, 1, 0, 0, 2, 1],
            vec![0, 0, 2, 0, 2, 0, 2, 1, 0, 1],
            vec![0, 0, 2, 0, 2, 0, 2, 0, 1, 1],
            vec![0, 0, 2, 0, 2, 0, 1, 2, 0, 1],
            vec![0, 0, 2, 0, 2, 0, 1, 1, 0, 2],
            vec![0, 0, 2, 0, 2, 0, 1, 0, 2, 1],
            vec![0, 0, 2, 0, 2, 0, 1, 0, 1, 2],
            vec![0, 0, 2, 0, 2, 0, 0, 2, 1, 1],
            vec![0, 0, 2, 0, 1, 2, 0, 2, 0, 1],
            vec![0, 0, 2, 0, 1, 2, 0, 1, 0, 2],
            vec![0, 0, 2, 0, 1, 2, 0, 0, 2, 1],
            vec![0, 0, 2, 0, 1, 1, 0, 2, 0, 2],
            vec![0, 0, 2, 0, 1, 1, 0, 0, 2, 2],
            vec![0, 0, 2, 0, 1, 0, 2, 2, 0, 1],
            vec![0, 0, 2, 0, 1, 0, 2, 1, 0, 2],
            vec![0, 0, 2, 0, 1, 0, 2, 0, 2, 1],
            vec![0, 0, 2, 0, 1, 0, 2, 0, 1, 2],
            vec![0, 0, 2, 0, 1, 0, 1, 2, 0, 2],
            vec![0, 0, 2, 0, 1, 0, 1, 0, 2, 2],
            vec![0, 0, 2, 0, 1, 0, 0, 2, 2, 1],
            vec![0, 0, 2, 0, 1, 0, 0, 2, 1, 2],
            vec![0, 0, 2, 0, 0, 2, 2, 1, 0, 1],
            vec![0, 0, 2, 0, 0, 2, 2, 0, 1, 1],
            vec![0, 0, 2, 0, 0, 2, 1, 2, 0, 1],
            vec![0, 0, 2, 0, 0, 2, 1, 1, 0, 2],
            vec![0, 0, 2, 0, 0, 2, 1, 0, 2, 1],
            vec![0, 0, 2, 0, 0, 2, 1, 0, 1, 2],
            vec![0, 0, 2, 0, 0, 2, 0, 2, 1, 1],
            vec![0, 0, 2, 0, 0, 2, 0, 1, 2, 1],
            vec![0, 0, 2, 0, 0, 2, 0, 1, 1, 2],
            vec![0, 0, 1, 2, 2, 0, 1, 0, 0, 2],
            vec![0, 0, 1, 2, 2, 0, 0, 2, 0, 1],
            vec![0, 0, 1, 2, 1, 0, 2, 0, 0, 2],
            vec![0, 0, 1, 2, 1, 0, 0, 2, 0, 2],
            vec![0, 0, 1, 2, 0, 2, 1, 0, 0, 2],
            vec![0, 0, 1, 2, 0, 2, 0, 2, 0, 1],
            vec![0, 0, 1, 2, 0, 2, 0, 1, 0, 2],
            vec![0, 0, 1, 2, 0, 2, 0, 0, 2, 1],
            vec![0, 0, 1, 2, 0, 1, 2, 0, 0, 2],
            vec![0, 0, 1, 2, 0, 1, 0, 2, 0, 2],
            vec![0, 0, 1, 2, 0, 1, 0, 0, 2, 2],
            vec![0, 0, 1, 2, 0, 0, 2, 2, 0, 1],
            vec![0, 0, 1, 2, 0, 0, 2, 1, 0, 2],
            vec![0, 0, 1, 2, 0, 0, 2, 0, 2, 1],
            vec![0, 0, 1, 2, 0, 0, 2, 0, 1, 2],
            vec![0, 0, 1, 2, 0, 0, 1, 2, 0, 2],
            vec![0, 0, 1, 1, 2, 0, 2, 0, 0, 2],
            vec![0, 0, 1, 1, 2, 0, 0, 2, 0, 2],
            vec![0, 0, 1, 1, 0, 2, 2, 0, 0, 2],
            vec![0, 0, 1, 1, 0, 2, 0, 2, 0, 2],
            vec![0, 0, 1, 1, 0, 2, 0, 0, 2, 2],
            vec![0, 0, 1, 1, 0, 0, 2, 2, 0, 2],
            vec![0, 0, 1, 1, 0, 0, 2, 0, 2, 2],
            vec![0, 0, 1, 0, 2, 2, 1, 0, 0, 2],
            vec![0, 0, 1, 0, 2, 2, 0, 2, 0, 1],
            vec![0, 0, 1, 0, 2, 2, 0, 1, 0, 2],
            vec![0, 0, 1, 0, 2, 2, 0, 0, 2, 1],
            vec![0, 0, 1, 0, 2, 2, 0, 0, 1, 2],
            vec![0, 0, 1, 0, 2, 1, 2, 0, 0, 2],
            vec![0, 0, 1, 0, 2, 1, 0, 2, 0, 2],
            vec![0, 0, 1, 0, 2, 1, 0, 0, 2, 2],
            vec![0, 0, 1, 0, 2, 0, 2, 2, 0, 1],
            vec![0, 0, 1, 0, 2, 0, 2, 1, 0, 2],
            vec![0, 0, 1, 0, 2, 0, 2, 0, 2, 1],
            vec![0, 0, 1, 0, 2, 0, 2, 0, 1, 2],
            vec![0, 0, 1, 0, 2, 0, 1, 2, 0, 2],
            vec![0, 0, 1, 0, 2, 0, 1, 0, 2, 2],
            vec![0, 0, 1, 0, 2, 0, 0, 2, 2, 1],
            vec![0, 0, 1, 0, 2, 0, 0, 2, 1, 2],
            vec![0, 0, 1, 0, 2, 0, 0, 1, 2, 2],
            vec![0, 0, 1, 0, 1, 2, 2, 0, 0, 2],
            vec![0, 0, 1, 0, 1, 2, 0, 2, 0, 2],
            vec![0, 0, 1, 0, 1, 2, 0, 0, 2, 2],
            vec![0, 0, 1, 0, 1, 0, 2, 2, 0, 2],
            vec![0, 0, 1, 0, 1, 0, 2, 0, 2, 2],
            vec![0, 0, 1, 0, 1, 0, 0, 2, 2, 2],
            vec![0, 0, 1, 0, 0, 2, 2, 2, 0, 1],
            vec![0, 0, 1, 0, 0, 2, 2, 1, 0, 2],
            vec![0, 0, 1, 0, 0, 2, 2, 0, 2, 1],
            vec![0, 0, 1, 0, 0, 2, 2, 0, 1, 2],
            vec![0, 0, 1, 0, 0, 2, 1, 2, 0, 2],
            vec![0, 0, 1, 0, 0, 2, 1, 0, 2, 2],
            vec![0, 0, 1, 0, 0, 2, 0, 2, 2, 1],
            vec![0, 0, 1, 0, 0, 2, 0, 2, 1, 2],
            vec![0, 0, 1, 0, 0, 2, 0, 1, 2, 2],
            vec![0, 0, 1, 0, 0, 1, 2, 2, 0, 2],
            vec![0, 0, 1, 0, 0, 1, 2, 0, 2, 2],
            vec![0, 0, 1, 0, 0, 1, 0, 2, 2, 2],
            vec![0, 0, 0, 2, 2, 2, 1, 0, 0, 1],
            vec![0, 0, 0, 2, 2, 2, 0, 1, 0, 1],
            vec![0, 0, 0, 2, 2, 2, 0, 0, 1, 1],
            vec![0, 0, 0, 2, 2, 1, 2, 0, 0, 1],
            vec![0, 0, 0, 2, 2, 1, 1, 0, 0, 2],
            vec![0, 0, 0, 2, 2, 1, 0, 2, 0, 1],
            vec![0, 0, 0, 2, 2, 1, 0, 1, 0, 2],
            vec![0, 0, 0, 2, 2, 1, 0, 0, 2, 1],
            vec![0, 0, 0, 2, 2, 1, 0, 0, 1, 2],
            vec![0, 0, 0, 2, 2, 0, 2, 1, 0, 1],
            vec![0, 0, 0, 2, 2, 0, 2, 0, 1, 1],
            vec![0, 0, 0, 2, 2, 0, 1, 2, 0, 1],
            vec![0, 0, 0, 2, 2, 0, 1, 1, 0, 2],
            vec![0, 0, 0, 2, 2, 0, 1, 0, 2, 1],
            vec![0, 0, 0, 2, 2, 0, 1, 0, 1, 2],
            vec![0, 0, 0, 2, 2, 0, 0, 2, 1, 1],
            vec![0, 0, 0, 2, 2, 0, 0, 1, 2, 1],
            vec![0, 0, 0, 2, 2, 0, 0, 1, 1, 2],
            vec![0, 0, 0, 2, 1, 2, 2, 0, 0, 1],
            vec![0, 0, 0, 2, 1, 2, 1, 0, 0, 2],
            vec![0, 0, 0, 2, 1, 2, 0, 2, 0, 1],
            vec![0, 0, 0, 2, 1, 2, 0, 1, 0, 2],
            vec![0, 0, 0, 2, 1, 2, 0, 0, 2, 1],
            vec![0, 0, 0, 2, 1, 2, 0, 0, 1, 2],
            vec![0, 0, 0, 2, 1, 1, 2, 0, 0, 2],
            vec![0, 0, 0, 2, 1, 1, 0, 2, 0, 2],
            vec![0, 0, 0, 2, 1, 1, 0, 0, 2, 2],
            vec![0, 0, 0, 2, 1, 0, 2, 2, 0, 1],
            vec![0, 0, 0, 2, 1, 0, 2, 1, 0, 2],
            vec![0, 0, 0, 2, 1, 0, 2, 0, 2, 1],
            vec![0, 0, 0, 2, 1, 0, 2, 0, 1, 2],
            vec![0, 0, 0, 2, 1, 0, 1, 2, 0, 2],
            vec![0, 0, 0, 2, 1, 0, 1, 0, 2, 2],
            vec![0, 0, 0, 2, 1, 0, 0, 2, 2, 1],
            vec![0, 0, 0, 2, 1, 0, 0, 2, 1, 2],
            vec![0, 0, 0, 2, 1, 0, 0, 1, 2, 2],
            vec![0, 0, 0, 2, 0, 2, 2, 1, 0, 1],
            vec![0, 0, 0, 2, 0, 2, 2, 0, 1, 1],
            vec![0, 0, 0, 2, 0, 2, 1, 2, 0, 1],
            vec![0, 0, 0, 2, 0, 2, 1, 1, 0, 2],
            vec![0, 0, 0, 2, 0, 2, 1, 0, 2, 1],
            vec![0, 0, 0, 2, 0, 2, 1, 0, 1, 2],
            vec![0, 0, 0, 2, 0, 2, 0, 2, 1, 1],
            vec![0, 0, 0, 2, 0, 2, 0, 1, 2, 1],
            vec![0, 0, 0, 2, 0, 2, 0, 1, 1, 2],
            vec![0, 0, 0, 2, 0, 1, 2, 2, 0, 1],
            vec![0, 0, 0, 2, 0, 1, 2, 1, 0, 2],
            vec![0, 0, 0, 2, 0, 1, 2, 0, 2, 1],
            vec![0, 0, 0, 2, 0, 1, 2, 0, 1, 2],
            vec![0, 0, 0, 2, 0, 1, 1, 2, 0, 2],
            vec![0, 0, 0, 2, 0, 1, 1, 0, 2, 2],
            vec![0, 0, 0, 2, 0, 1, 0, 2, 2, 1],
            vec![0, 0, 0, 2, 0, 1, 0, 2, 1, 2],
            vec![0, 0, 0, 2, 0, 1, 0, 1, 2, 2],
            vec![0, 0, 0, 2, 0, 0, 2, 2, 1, 1],
            vec![0, 0, 0, 2, 0, 0, 2, 1, 2, 1],
            vec![0, 0, 0, 2, 0, 0, 2, 1, 1, 2],
            vec![0, 0, 0, 2, 0, 0, 1, 2, 2, 1],
            vec![0, 0, 0, 2, 0, 0, 1, 2, 1, 2],
            vec![0, 0, 0, 2, 0, 0, 1, 1, 2, 2],
            vec![0, 0, 0, 1, 2, 2, 2, 0, 0, 1],
            vec![0, 0, 0, 1, 2, 2, 1, 0, 0, 2],
            vec![0, 0, 0, 1, 2, 2, 0, 2, 0, 1],
            vec![0, 0, 0, 1, 2, 2, 0, 1, 0, 2],
            vec![0, 0, 0, 1, 2, 2, 0, 0, 2, 1],
            vec![0, 0, 0, 1, 2, 2, 0, 0, 1, 2],
            vec![0, 0, 0, 1, 2, 1, 2, 0, 0, 2],
            vec![0, 0, 0, 1, 2, 1, 0, 2, 0, 2],
            vec![0, 0, 0, 1, 2, 1, 0, 0, 2, 2],
            vec![0, 0, 0, 1, 2, 0, 2, 2, 0, 1],
            vec![0, 0, 0, 1, 2, 0, 2, 1, 0, 2],
            vec![0, 0, 0, 1, 2, 0, 2, 0, 2, 1],
            vec![0, 0, 0, 1, 2, 0, 2, 0, 1, 2],
            vec![0, 0, 0, 1, 2, 0, 1, 2, 0, 2],
            vec![0, 0, 0, 1, 2, 0, 1, 0, 2, 2],
            vec![0, 0, 0, 1, 2, 0, 0, 2, 2, 1],
            vec![0, 0, 0, 1, 2, 0, 0, 2, 1, 2],
            vec![0, 0, 0, 1, 2, 0, 0, 1, 2, 2],
            vec![0, 0, 0, 1, 1, 2, 2, 0, 0, 2],
            vec![0, 0, 0, 1, 1, 2, 0, 2, 0, 2],
            vec![0, 0, 0, 1, 1, 2, 0, 0, 2, 2],
            vec![0, 0, 0, 1, 1, 0, 2, 2, 0, 2],
            vec![0, 0, 0, 1, 1, 0, 2, 0, 2, 2],
            vec![0, 0, 0, 1, 1, 0, 0, 2, 2, 2],
            vec![0, 0, 0, 1, 0, 2, 2, 2, 0, 1],
            vec![0, 0, 0, 1, 0, 2, 2, 1, 0, 2],
            vec![0, 0, 0, 1, 0, 2, 2, 0, 2, 1],
            vec![0, 0, 0, 1, 0, 2, 2, 0, 1, 2],
            vec![0, 0, 0, 1, 0, 2, 1, 2, 0, 2],
            vec![0, 0, 0, 1, 0, 2, 1, 0, 2, 2],
            vec![0, 0, 0, 1, 0, 2, 0, 2, 2, 1],
            vec![0, 0, 0, 1, 0, 2, 0, 2, 1, 2],
            vec![0, 0, 0, 1, 0, 2, 0, 1, 2, 2],
            vec![0, 0, 0, 1, 0, 1, 2, 2, 0, 2],
            vec![0, 0, 0, 1, 0, 1, 2, 0, 2, 2],
            vec![0, 0, 0, 1, 0, 1, 0, 2, 2, 2],
            vec![0, 0, 0, 1, 0, 0, 2, 2, 2, 1],
            vec![0, 0, 0, 1, 0, 0, 2, 2, 1, 2],
            vec![0, 0, 0, 1, 0, 0, 2, 1, 2, 2],
            vec![0, 0, 0, 1, 0, 0, 1, 2, 2, 2],
            vec![0, 0, 0, 0, 2, 2, 2, 1, 0, 1],
            vec![0, 0, 0, 0, 2, 2, 2, 0, 1, 1],
            vec![0, 0, 0, 0, 2, 2, 1, 2, 0, 1],
            vec![0, 0, 0, 0, 2, 2, 1, 1, 0, 2],
            vec![0, 0, 0, 0, 2, 2, 1, 0, 2, 1],
            vec![0, 0, 0, 0, 2, 2, 1, 0, 1, 2],
            vec![0, 0, 0, 0, 2, 2, 0, 2, 1, 1],
            vec![0, 0, 0, 0, 2, 2, 0, 1, 2, 1],
            vec![0, 0, 0, 0, 2, 2, 0, 1, 1, 2],
            vec![0, 0, 0, 0, 2, 1, 2, 2, 0, 1],
            vec![0, 0, 0, 0, 2, 1, 2, 1, 0, 2],
            vec![0, 0, 0, 0, 2, 1, 2, 0, 2, 1],
            vec![0, 0, 0, 0, 2, 1, 2, 0, 1, 2],
            vec![0, 0, 0, 0, 2, 1, 1, 2, 0, 2],
            vec![0, 0, 0, 0, 2, 1, 1, 0, 2, 2],
            vec![0, 0, 0, 0, 2, 1, 0, 2, 2, 1],
            vec![0, 0, 0, 0, 2, 1, 0, 2, 1, 2],
            vec![0, 0, 0, 0, 2, 1, 0, 1, 2, 2],
            vec![0, 0, 0, 0, 2, 0, 2, 2, 1, 1],
            vec![0, 0, 0, 0, 2, 0, 2, 1, 2, 1],
            vec![0, 0, 0, 0, 2, 0, 2, 1, 1, 2],
            vec![0, 0, 0, 0, 2, 0, 1, 2, 2, 1],
            vec![0, 0, 0, 0, 2, 0, 1, 2, 1, 2],
            vec![0, 0, 0, 0, 2, 0, 1, 1, 2, 2],
            vec![0, 0, 0, 0, 1, 2, 2, 2, 0, 1],
            vec![0, 0, 0, 0, 1, 2, 2, 1, 0, 2],
            vec![0, 0, 0, 0, 1, 2, 2, 0, 2, 1],
            vec![0, 0, 0, 0, 1, 2, 2, 0, 1, 2],
            vec![0, 0, 0, 0, 1, 2, 1, 2, 0, 2],
            vec![0, 0, 0, 0, 1, 2, 1, 0, 2, 2],
            vec![0, 0, 0, 0, 1, 2, 0, 2, 2, 1],
            vec![0, 0, 0, 0, 1, 2, 0, 2, 1, 2],
            vec![0, 0, 0, 0, 1, 2, 0, 1, 2, 2],
            vec![0, 0, 0, 0, 1, 1, 2, 2, 0, 2],
            vec![0, 0, 0, 0, 1, 1, 2, 0, 2, 2],
            vec![0, 0, 0, 0, 1, 1, 0, 2, 2, 2],
            vec![0, 0, 0, 0, 1, 0, 2, 2, 2, 1],
            vec![0, 0, 0, 0, 1, 0, 2, 2, 1, 2],
            vec![0, 0, 0, 0, 1, 0, 2, 1, 2, 2],
            vec![0, 0, 0, 0, 1, 0, 1, 2, 2, 2],
            vec![0, 0, 0, 0, 0, 2, 2, 2, 1, 1],
            vec![0, 0, 0, 0, 0, 2, 2, 1, 2, 1],
            vec![0, 0, 0, 0, 0, 2, 2, 1, 1, 2],
            vec![0, 0, 0, 0, 0, 2, 1, 2, 2, 1],
            vec![0, 0, 0, 0, 0, 2, 1, 2, 1, 2],
            vec![0, 0, 0, 0, 0, 2, 1, 1, 2, 2],
            vec![0, 0, 0, 0, 0, 1, 2, 2, 2, 1],
            vec![0, 0, 0, 0, 0, 1, 2, 2, 1, 2],
            vec![0, 0, 0, 0, 0, 1, 2, 1, 2, 2],
            vec![0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
        ]);
        assert_eq!(attempt, correct_set);
    }
}
