//! JI-agnostic 2D pitch class lattice visualization.
//!
//! This module computes lattice coordinates for scale pitches, enabling
//! geometric visualization of scale structure independent of specific tunings.
//!
//! # Key Concepts
//!
//! - **Pitch class lattice**: A 2D projection of scale degrees onto a plane. We require that the lattice have no torsion (i.e. the equave not be a multiple of any lattice element).
//! - **Unimodular basis**: A list of vectors whose determinant is ±1
//! - **Parallelogram**: The fundamental domain tiled by the lattice
//!
//! # Algorithm
//!
//! 1. Find a unimodular basis from the scale's guide frames
//! 2. Project each pitch onto the 2D plane spanned by the basis vectors
//! 3. The result shows the geometric structure of the scale
//!
//! # Examples
//!
//! ```
//! use tern::lattice::try_pitch_class_lattice;
//!
//! // Diasem scale
//! let diasem = [0, 1, 0, 2, 0, 1, 0, 2, 0];
//!
//! // Get lattice coordinates if a unimodular basis exists
//! if let Some((coords, basis)) = try_pitch_class_lattice(&diasem) {
//!     // coords contains 2D coordinates for each pitch class
//!     assert_eq!(coords.len(), 9);  // One coordinate per scale degree
//! }
//! ```

use std::iter::IntoIterator;

use itertools::Itertools;
use serde::Serialize;

use crate::GuideResult;
use crate::countvector_to_u16_vec;
use crate::equal::direct_approx;
use crate::guide::*;
use crate::guide_frame_to_result;
use crate::ji_ratio::RawJiRatio;
use crate::matrix;
use crate::matrix::unimodular_inv;
use crate::word_to_sig;
use crate::words::CountVector;

/// An ordered pair of vectors forming a unimodular basis for the pitch class lattice.
///
/// The basis vectors are in scale-step coordinates `[L, m, s]`, where each component
/// represents the count of that step type. Together with the equave (step signature),
/// these vectors have determinant ±1.
///
/// # Examples
///
/// ```
/// use tern::lattice::PitchClassLatticeBasis;
///
/// // A basis with two generators
/// let basis = PitchClassLatticeBasis::from_slices(&[1, 1, 0], &[0, 1, 1]);
/// assert_eq!(basis.vx(), &[1, 1, 0]);
/// assert_eq!(basis.vy(), &[0, 1, 1]);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PitchClassLatticeBasis {
    vx: Vec<i32>,
    vy: Vec<i32>,
}

impl PitchClassLatticeBasis {
    // Create a new `PitchClassLatticeBasis` from two slices (representing the two basis vectors).
    pub fn from_slices(vx: &[i32], vy: &[i32]) -> Self {
        Self {
            vx: vx.to_vec(),
            vy: vy.to_vec(),
        }
    }
    /// Get the first basis vector (represented as the x-direction in the lattice diagram).
    pub fn vx(&self) -> &[i32] {
        &self.vx
    }
    /// Get the second basis vector (represented as the y-direction in the lattice diagram).
    pub fn vy(&self) -> &[i32] {
        &self.vy
    }

    pub fn equave_reduce(&self, step_sig: &[i32]) -> Self {
        let mut new_vx = self.vx.clone();
        while (0..3).any(|i| new_vx[i] < 0) {
            (0..3).for_each(|i| new_vx[i] += step_sig[i]);
        }
        while (0..3).any(|i| new_vx[i] > step_sig[i]) {
            (0..3).for_each(|i| new_vx[i] -= step_sig[i]);
        }
        let mut new_vy = self.vy.clone();
        while (0..3).any(|i| new_vy[i] < 0) {
            (0..3).for_each(|i| new_vy[i] += step_sig[i]);
        }
        while (0..3).any(|i| new_vy[i] > step_sig[i]) {
            (0..3).for_each(|i| new_vy[i] -= step_sig[i]);
        }
        Self::from_slices(&new_vx, &new_vy)
    }
}

impl IntoIterator for PitchClassLatticeBasis {
    type Item = Vec<i32>;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        vec![self.vx, self.vy].into_iter()
    }
}

/// A struct representing a substring of a row-by-row traversal of a lattice parallelogram.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParallelogramSubstring {
    row_count: i32,
    full_row_len: i32,
    first_row_len: i32,
    last_row_len: i32,
}

impl ParallelogramSubstring {
    fn new(row_count: i32, full_row_len: i32, first_row_len: i32, last_row_len: i32) -> Self {
        ParallelogramSubstring {
            row_count,
            full_row_len,
            first_row_len,
            last_row_len,
        }
    }
    /// Get the number of rows, full or not.
    pub fn row_count(&self) -> i32 {
        self.row_count
    }
    /// Get the length of a full row.
    pub fn full_row_len(&self) -> i32 {
        self.full_row_len
    }
    /// Get the length of the first row.
    pub fn first_row_len(&self) -> i32 {
        self.first_row_len
    }
    /// Get the length of the last row.
    pub fn last_row_len(&self) -> i32 {
        self.last_row_len
    }
}

pub fn get_unimodular_basis(
    structures: &[GuideFrame],
    step_sig: &[i32],
) -> Option<(Vec<Vec<i32>>, GuideResult)> {
    for structure in structures {
        let result = guide_frame_to_result(structure);
        let offset = &result
            .clone()
            .offset_chord
            .into_iter()
            .map(|x| x.into_iter().map(|x| x as i32).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let gs = &result
            .clone()
            .gs
            .into_iter()
            .map(|x| x.into_iter().map(|x| x as i32).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        if structure.multiplicity() == 1 {
            for i in 0..gs.len() {
                for j in (i + 1)..gs.len() {
                    if matrix::det3(step_sig, &gs[i], &gs[j]).abs() == 1 {
                        return Some((vec![gs[i].clone(), gs[j].clone()], result));
                    }
                }
            }
            for v in offset {
                for w in gs {
                    if matrix::det3(step_sig, v, w).abs() == 1 {
                        return Some((vec![v.clone(), w.clone()], result));
                    }
                }
            }
        } else {
            // this branch handles multiplicity > 1 scales
            // Check all pairs from gs and offset_chord
            for v in offset {
                for w in gs {
                    if matrix::det3(step_sig, w, v).abs() == 1 {
                        return Some((vec![w.clone(), v.clone()], result));
                    }
                }
            }
        }
    }
    None
}

/// Project pitches onto the pitch class lattice using the given basis.
/// Takes a scale (word) and a basis, and returns the pitch classes projected onto that basis.
pub fn pitch_classes(
    query: &[usize],
    basis: &PitchClassLatticeBasis,
) -> (Vec<Vec<i32>>, PitchClassLatticeBasis) {
    let sig = word_to_sig(query)
        .iter()
        .map(|x| *x as u16)
        .collect::<Vec<u16>>();

    let (equave, gener_1, gener_2) = (
        sig.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        basis.vx().to_vec(),
        basis.vy().to_vec(),
    );

    // Invert [equave, gener_1, gener_2] to get basis change matrix
    let basis_change = unimodular_inv(&equave, &gener_1, &gener_2);

    // Now project all vectors to the (generator1, generator2)-plane
    // Remove the first coordinate
    let mut pitch_classes = vec![];
    let mut count_vector = CountVector::ZERO;
    for step in query {
        // Add the current step
        count_vector = count_vector.add(&CountVector::from_slice(&[*step]));
        let v_i = countvector_to_u16_vec(&count_vector)
            .iter()
            .map(|x| *x as i32)
            .collect::<Vec<_>>();
        let v_i_transformed =
            matrix::matrix_times_vector(&basis_change[0], &basis_change[1], &basis_change[2], &v_i);
        let v_i_projected = v_i_transformed[1..3].to_vec();
        pitch_classes.push(v_i_projected);
    }
    (pitch_classes, basis.clone())
}

/// Compute 2D lattice coordinates for each pitch in a scale.
///
/// Finds a unimodular basis from the scale's guide frames and projects
/// each pitch class onto the 2D plane spanned by that basis.
///
/// # Arguments
///
/// * `query` - A scale word (sequence of step letters)
///
/// # Returns
///
/// `Some((coordinates, basis))` if a unimodular basis exists, where:
/// - `coordinates[i]` is the `[x, y]` position of pitch class `i`
/// - `basis` is the [`PitchClassLatticeBasis`] used for projection
///
/// Returns `None` if no unimodular basis can be found.
///
/// # Examples
///
/// ```
/// use tern::lattice::try_pitch_class_lattice;
///
/// let diasem = [0, 1, 0, 2, 0, 1, 0, 2, 0];
/// if let Some((coords, _basis)) = try_pitch_class_lattice(&diasem) {
///     // Each pitch has a 2D coordinate
///     for coord in &coords {
///         assert_eq!(coord.len(), 2);
///     }
/// }
/// ```
pub fn try_pitch_class_lattice(query: &[usize]) -> Option<(Vec<Vec<i32>>, PitchClassLatticeBasis)> {
    let gfs = guide_frames(query);
    let sig = word_to_sig(query)
        .iter()
        .map(|x| *x as i32)
        .collect::<Vec<_>>();
    get_unimodular_basis(&gfs, &sig).map(|(basis_, _)| {
        // For every pitch in the scale expressed as a CountVector,
        // do a change of basis from scale steps basis
        // to (equave, generator1, generator2) basis.
        // basis_ doesn't have the equave, so we add it back first.
        let (equave, gener_1, gener_2) = (sig.clone(), basis_[0].clone(), basis_[1].clone());
        // Invert [equave, gener_1, gener_2] to get basis change matrix
        let basis_change = unimodular_inv(&equave, &gener_1, &gener_2);
        // Now project all vectors to the (generator1, generator2)-plane
        // Remove the first coordinate
        let mut pitch_classes = vec![];
        let mut count_vector = CountVector::ZERO;
        for step in query {
            // Add the current step
            count_vector = count_vector.add(&CountVector::from_slice(&[*step]));
            let v_i = countvector_to_u16_vec(&count_vector)
                .iter()
                .map(|x| *x as i32)
                .collect::<Vec<_>>();
            let v_i_transformed = matrix::matrix_times_vector(
                &basis_change[0],
                &basis_change[1],
                &basis_change[2],
                &v_i,
            );
            let v_i_projected = v_i_transformed[1..3].to_vec();
            pitch_classes.push(v_i_projected);
        }
        (
            pitch_classes,
            PitchClassLatticeBasis::from_slices(&gener_1, &gener_2),
        )
    })
}

/// Whether the result is `Some` or `None` depends on
/// whether the pitch classes form a substring of a parallelogram traversal,
/// i.e. its pitch classes forming a substring of a traversal
/// ```text
/// -> (0,0), (0,1), (0,2), ..., (0,n)
/// -> (1,0), (1,1), (1,2), ..., (1,n)
/// -> (2,0), (2,1), (2,2), ..., (2,n)
/// ...
/// -> (m,0), (m,1), (m,2), ..., (m,n)
/// ```
/// under some choice of coordinate vectors (v, w).
/// If `Some`, returns a `ParallelogramSubstring` struct containing the following info:
/// row count, length of a full row, length of first row, length of last row.
/// Also return the corresponding basis written in scale step coordinates.
///
/// Ideally, this function should prioritize pitch class lattice bases with a generator (whether a row generator or not)
/// that is likely to be a fifth/fourth, provided such a basis exists for a given scale.
pub fn parallelogram_substring_info(
    pitch_classes: &[&[i32]],
    old_basis: &PitchClassLatticeBasis,
) -> Option<(ParallelogramSubstring, PitchClassLatticeBasis)> {
    let scale_size = pitch_classes.len();
    // `basis` is written in scale step coordinates (L, m, s).
    // The pitch classes are written in coordinates given by `basis`.
    // Get all pairwise differences between distinct points.
    let mut pairwise_differences: Vec<Vec<i32>> = vec![];
    for i in 0..scale_size {
        for j in i + 1..scale_size {
            let diff = vec![
                pitch_classes[j][0] - pitch_classes[i][0],
                pitch_classes[j][1] - pitch_classes[i][1],
            ];
            pairwise_differences.push(diff);
        }
    }
    // Sort pairwise differences so that
    // bases that likely have fifths or fourths (determined by patent val mapping for scale_size-edo) come first.
    pairwise_differences.sort_by_key(|diff| {
        let scale_size_i32 = scale_size as i32;
        let fifth_mapping =
            direct_approx(RawJiRatio::PYTH_5TH, scale_size as f64, RawJiRatio::OCTAVE);
        let fourth_mapping = scale_size_i32 - fifth_mapping;
        let taxicab_len_lms: i32 = (0..3)
            .map(|i| (diff[0] * old_basis.vx[i] + diff[1] * old_basis.vy[i]).abs())
            .sum();
        // Negate because false < true and sorting is in ascending order
        !(taxicab_len_lms % scale_size_i32 == fifth_mapping
            || taxicab_len_lms % scale_size_i32 == fourth_mapping)
    });
    // Look for a unimodular basis that witnesses the parallelogram substring property.
    // If this basis turns out to work, just use the basis vectors' components as coefficients
    // to write the basis in step size coordinates.
    for (i, vx) in pairwise_differences.iter().enumerate() {
        for vy in pairwise_differences.iter().skip(i + 1) {
            if (vx[0] * vy[1] - vx[1] * vy[0]).abs() == 1 {
                // Change coordinates to basis (v1, v2)
                let basis_change: Vec<Vec<i32>> = vec![vec![vy[1], -vx[1]], vec![-vy[0], vx[0]]];
                let mut pitch_classes_transformed = pitch_classes
                    .iter()
                    .map(|v| {
                        vec![
                            basis_change[0][0] * v[0] + basis_change[1][0] * v[1],
                            basis_change[0][1] * v[0] + basis_change[1][1] * v[1],
                        ]
                    })
                    .collect::<Vec<_>>();
                // Get window dimensions: x_min, x_max, y_min, y_max
                let mut xs: Vec<_> = pitch_classes_transformed.iter().map(|v| v[0]).collect();
                let mut ys: Vec<_> = pitch_classes_transformed.iter().map(|v| v[1]).collect();
                xs.sort();
                ys.sort();
                let x_min = xs[0];
                let x_max = xs[xs.len() - 1];
                let y_min = ys[0];
                let y_max = ys[ys.len() - 1];
                // Check all 4 possible traversals:
                // 1. each row LTR (increases in x), rows go BTT (increases in y) (equivalently each row RTL, rows go TTB)
                // 2. each row RTL (decreases in x), rows go BTT (increases in y) (equivalently each row LTR, rows go TTB)
                // 3. each row BTT (increases in y), rows go LTR (increases in x) (equivalently each row TTB, rows go RTL)
                // 4. each row TTB (decreases in y), rows go LTR (increases in x) (equivalently each row BTT, rows go RTL)
                'traversal12: {
                    // Sort pitch_classes_transformed in lex order for traversal 1
                    pitch_classes_transformed.sort_by(|v1, v2| {
                        // Sort by *ascending* y values, if y values are equal sort by *ascending* x values
                        v1[1].cmp(&v2[1]).then(v1[0].cmp(&v2[0]))
                    });
                    let mut index = 0; // index into pitch_classes_transformed
                    // Check if middle rows are fully occupied; if not break out of block early
                    let ys_middle = (y_min + 1)..=(y_max - 1);
                    let full_row_len = x_max - x_min + 1; // Required length of each middle row
                    for y in ys_middle {
                        let mut row_counter = 0; // Count pitches with this y value
                        while pitch_classes_transformed[index][1] < y {
                            index += 1;
                        }
                        while pitch_classes_transformed[index][1] == y {
                            row_counter += 1;
                            index += 1;
                        }
                        if row_counter != full_row_len {
                            break 'traversal12;
                        }
                    }
                    'traversal1: {
                        // Check outer rows for traversal 1
                        // Last row must be a prefix of a row traversal
                        let mut last_row = vec![];
                        while pitch_classes_transformed[index][1] < y_max {
                            index += 1;
                        }
                        while index < pitch_classes_transformed.len() {
                            last_row.push(pitch_classes_transformed[index].clone());
                            index += 1;
                        }
                        // First x value in last_row == x_min AND
                        // |last x value in row - first x value in row| + 1 == last_row.len()
                        // (last_row should be sorted by *ascending* x values)
                        let last_row_is_prefix = !last_row.is_empty()
                            && last_row[0][0] == x_min
                            && (last_row[last_row.len() - 1][0] - x_min + 1) as usize
                                == last_row.len();
                        if !last_row_is_prefix {
                            break 'traversal1;
                        }
                        // First row must be a suffix of a row traversal
                        index = 0;
                        let mut first_row = vec![];
                        while pitch_classes_transformed[index][1] == y_min {
                            first_row.push(pitch_classes_transformed[index].clone());
                            index += 1;
                        }
                        // Last x value in first_row == x_max AND
                        // |last x value in row - first x value in row| + 1 == first_row.len()
                        // (first_row should be sorted by *ascending* x values)
                        let first_row_is_suffix = !first_row.is_empty()
                            && first_row[first_row.len() - 1][0] == x_max
                            && (x_max - first_row[0][0] + 1) as usize == first_row.len();

                        // IMPORTANT: When there are no middle rows,
                        // verify that first and last rows span the same x-range.
                        let row_count = y_max - y_min + 1;
                        let rows_compatible = if row_count == 2 || full_row_len == 2 {
                            // At least one row must span the full range [x_min, x_max]
                            (first_row.len() as i32 == full_row_len)
                                || (last_row.len() as i32 == full_row_len)
                        } else {
                            true // Middle rows already enforce consistency
                        };

                        if first_row_is_suffix && rows_compatible {
                            let row_count = y_max - y_min + 1;
                            let first_row_len = first_row.len() as i32;
                            let last_row_len = last_row.len() as i32;
                            let vx_lms = (0..3) // for each of L, m, s
                                .map(|i| vx[0] * old_basis.vx[i] + vx[1] * old_basis.vy[i])
                                .collect::<Vec<_>>();
                            let vy_lms = (0..3)
                                .map(|i| vy[0] * old_basis.vx[i] + vy[1] * old_basis.vy[i])
                                .collect::<Vec<_>>();
                            return Some((
                                ParallelogramSubstring::new(
                                    row_count,
                                    full_row_len,
                                    first_row_len,
                                    last_row_len,
                                ),
                                PitchClassLatticeBasis::from_slices(&vx_lms, &vy_lms), // put row generator first
                            ));
                        }
                    }
                    // Sort pitch_classes_transformed in lex order for traversal 2
                    pitch_classes_transformed.sort_by(|v1, v2| {
                        // Sort by *ascending* y values, if y values are equal sort by *descending* x values
                        v1[1].cmp(&v2[1]).then(v1[0].cmp(&v2[0]).reverse())
                    });
                    'traversal2: {
                        // Check outer rows for traversal 2
                        // Last row must be a prefix of a row traversal
                        let mut last_row = vec![];
                        // Only need to check at most `full_row_len` elements from end
                        index = pitch_classes_transformed
                            .len()
                            .saturating_sub(full_row_len as usize);
                        while index < pitch_classes_transformed.len()
                            && pitch_classes_transformed[index][1] > y_min
                        {
                            index += 1;
                        }
                        while index < pitch_classes_transformed.len() {
                            last_row.push(pitch_classes_transformed[index].clone());
                            index += 1;
                        }
                        // First x value in last_row == x_max AND
                        // |last x value in row - first x value in row| + 1 == last_row.len()
                        // (last_row should be sorted by *descending* x values)
                        let last_row_is_prefix = !last_row.is_empty()
                            && last_row[0][0] == x_max
                            && (x_max - last_row[last_row.len() - 1][0] + 1) as usize
                                == last_row.len();
                        if !last_row_is_prefix {
                            break 'traversal2;
                        }
                        // First row must be a suffix of a row traversal
                        index = 0;
                        let mut first_row = vec![];
                        while pitch_classes_transformed[index][1] == y_max {
                            first_row.push(pitch_classes_transformed[index].clone());
                            index += 1;
                        }
                        // Last x value in first_row == x_min AND
                        // |last x value in row - first x value in row| + 1 == first_row.len()
                        // (first_row should be sorted by *descending* x values)
                        let first_row_is_suffix = !first_row.is_empty()
                            && first_row[first_row.len() - 1][0] == x_min
                            && (first_row[0][0] - x_min + 1) as usize == first_row.len();

                        // IMPORTANT: When there are no middle rows,
                        // verify that first and last rows span the same x-range.
                        let row_count = y_max - y_min + 1;
                        let rows_compatible = if row_count == 2 || full_row_len == 2 {
                            // At least one row must span the full range [x_min, x_max]
                            (first_row.len() as i32 == full_row_len)
                                || (last_row.len() as i32 == full_row_len)
                        } else {
                            true // Middle rows already enforce consistency
                        };

                        if first_row_is_suffix && rows_compatible {
                            let row_count = y_max - y_min + 1;
                            let first_row_len = first_row.len() as i32;
                            let last_row_len = last_row.len() as i32;
                            let vx_lms = (0..3)
                                .map(|i| vx[0] * old_basis.vx[i] + vx[1] * old_basis.vy[i])
                                .collect::<Vec<_>>();
                            let vy_lms = (0..3)
                                .map(|i| vy[0] * old_basis.vx[i] + vy[1] * old_basis.vy[i])
                                .collect::<Vec<_>>();
                            return Some((
                                ParallelogramSubstring::new(
                                    row_count,
                                    full_row_len,
                                    first_row_len,
                                    last_row_len,
                                ),
                                PitchClassLatticeBasis::from_slices(&vx_lms, &vy_lms), // put row generator first
                            ));
                        }
                    }
                }
                'traversal34: {
                    // Sort pitch_classes_transformed in lex order for traversal 3
                    pitch_classes_transformed.sort_by(|v1, v2| {
                        // Sort by *ascending* x values, if x values are equal sort by *ascending* y values
                        v1[0].cmp(&v2[0]).then(v1[1].cmp(&v2[1]))
                    });
                    let mut index = 0; // index into pitch_classes_transformed
                    // Check if middle rows are fully occupied; if not break out of block early
                    let xs_middle = (x_min + 1)..=(x_max - 1);
                    let full_row_len = y_max - y_min + 1; // Required length of each middle row
                    for x in xs_middle {
                        let mut row_counter = 0; // Count pitches with this y value
                        while pitch_classes_transformed[index][0] < x {
                            index += 1;
                        }
                        while pitch_classes_transformed[index][0] == x {
                            row_counter += 1;
                            index += 1;
                        }
                        if row_counter != full_row_len {
                            break 'traversal34;
                        }
                    }
                    'traversal3: {
                        // Check outer rows for traversal 3
                        // Last row must be a prefix of a row traversal
                        let mut last_row = vec![];
                        while pitch_classes_transformed[index][0] < x_max {
                            index += 1;
                        }
                        while index < pitch_classes_transformed.len() {
                            last_row.push(pitch_classes_transformed[index].clone());
                            index += 1;
                        }
                        // First y value in last_row == y_min AND
                        // |last y value in row - first y value in row| + 1 == last_row.len()
                        // (last_row should be sorted by *ascending* y values)
                        let last_row_is_prefix = !last_row.is_empty()
                            && last_row[0][1] == y_min
                            && (last_row[last_row.len() - 1][1] - y_min + 1) as usize
                                == last_row.len();
                        if !last_row_is_prefix {
                            break 'traversal3;
                        }
                        // First row must be a suffix of a row traversal
                        index = 0;
                        let mut first_row = vec![];
                        while pitch_classes_transformed[index][0] == x_min {
                            first_row.push(pitch_classes_transformed[index].clone());
                            index += 1;
                        }
                        // Last y value in first_row == y_max AND
                        // |last y value in row - first y value in row| + 1 == first_row.len()
                        // (first_row should be sorted by *ascending* y values)
                        let first_row_is_suffix = !first_row.is_empty()
                            && first_row[first_row.len() - 1][1] == y_max
                            && (y_max - first_row[0][1] + 1) as usize == first_row.len();

                        // IMPORTANT: When there are no middle rows,
                        // verify that first and last rows span the same y-range.
                        // Otherwise we might accept two rows with overlapping but different ranges.
                        let row_count = x_max - x_min + 1;
                        let rows_compatible = if row_count == 2 || full_row_len == 2 {
                            // At least one row must span the full range [y_min, y_max]
                            (first_row.len() as i32 == full_row_len)
                                || (last_row.len() as i32 == full_row_len)
                        } else {
                            true // Middle rows already enforce consistency
                        };

                        if first_row_is_suffix && rows_compatible {
                            let row_count = x_max - x_min + 1;
                            let first_row_len = first_row.len() as i32;
                            let last_row_len = last_row.len() as i32;
                            let vx_lms = (0..3)
                                .map(|i| vx[0] * old_basis.vx[i] + vx[1] * old_basis.vy[i])
                                .collect::<Vec<_>>();
                            let vy_lms = (0..3)
                                .map(|i| vy[0] * old_basis.vx[i] + vy[1] * old_basis.vy[i])
                                .collect::<Vec<_>>();
                            return Some((
                                ParallelogramSubstring::new(
                                    row_count,
                                    full_row_len,
                                    first_row_len,
                                    last_row_len,
                                ),
                                PitchClassLatticeBasis::from_slices(&vy_lms, &vx_lms), // put row generator first
                            ));
                        }
                    }
                    // Sort pitch_classes_transformed in lex order for traversal 4
                    pitch_classes_transformed.sort_by(|v1, v2| {
                        // Sort by *ascending* x values, if y values are equal sort by *descending* y values
                        v1[0].cmp(&v2[0]).then(v1[1].cmp(&v2[1]).reverse())
                    });
                    'traversal4: {
                        // Check outer rows for traversal 4
                        // Last row must be a prefix of a row traversal
                        let mut last_row = vec![];
                        // Only need to check at most `full_row_len` elements from end
                        index = pitch_classes_transformed
                            .len()
                            .saturating_sub(full_row_len as usize);
                        while index < pitch_classes_transformed.len()
                            && pitch_classes_transformed[index][0] > x_min
                        {
                            index += 1;
                        }
                        while index < pitch_classes_transformed.len() {
                            last_row.push(pitch_classes_transformed[index].clone());
                            index += 1;
                        }
                        // First y value in last_row == y_max AND
                        // |last y value in row - first y value in row| + 1 == last_row.len()
                        // (last_row should be sorted by descending y values)
                        let last_row_is_prefix = !last_row.is_empty()
                            && last_row[0][1] == y_max
                            && (y_max - last_row[last_row.len() - 1][1] + 1) as usize
                                == last_row.len();
                        if !last_row_is_prefix {
                            break 'traversal4;
                        }
                        // First row must be a suffix of a row traversal
                        index = 0;
                        let mut first_row = vec![];
                        while pitch_classes_transformed[index][1] == y_max {
                            first_row.push(pitch_classes_transformed[index].clone());
                            index += 1;
                        }
                        // Last y value in first_row == y_min AND
                        // |last y value in row - first y value in row| + 1 == first_row.len()
                        // (last_row should be sorted by *descending* x values)
                        let first_row_is_suffix = !first_row.is_empty()
                            && first_row[first_row.len() - 1][1] == y_min
                            && (first_row[0][1] - y_min + 1) as usize == first_row.len();

                        // IMPORTANT: When there are no middle rows,
                        // verify that first and last rows span the same y-range.
                        let row_count = x_max - x_min + 1;
                        let rows_compatible = if row_count == 2 || full_row_len == 2 {
                            // At least one row must span the full range [y_min, y_max]
                            (first_row.len() as i32 == full_row_len)
                                || (last_row.len() as i32 == full_row_len)
                        } else {
                            true // Middle rows already enforce consistency
                        };

                        if first_row_is_suffix && rows_compatible {
                            let row_count = x_max - x_min + 1;
                            let first_row_len = first_row.len() as i32;
                            let last_row_len = last_row.len() as i32;
                            let vx_lms = (0..3)
                                .map(|i| vx[0] * old_basis.vx[i] + vx[1] * old_basis.vy[i])
                                .collect::<Vec<_>>();
                            let vy_lms = (0..3)
                                .map(|i| vy[0] * old_basis.vx[i] + vy[1] * old_basis.vy[i])
                                .collect::<Vec<_>>();
                            return Some((
                                ParallelogramSubstring::new(
                                    row_count,
                                    full_row_len,
                                    first_row_len,
                                    last_row_len,
                                ),
                                PitchClassLatticeBasis::from_slices(&vy_lms, &vx_lms), // put row generator first
                            ));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Try to find a better basis for a more arbitrary set of lattice points.
/// If the pitch classes do not form a parallelogram substring,
/// this still provides a less sheared basis by finding the most common change in max x value between rows
/// (or y value between columns) and deshearing by that amount.
pub fn deshear(
    pitch_classes: &[Vec<i32>],
    old_basis: &PitchClassLatticeBasis,
) -> PitchClassLatticeBasis {
    let mut x_values = pitch_classes.iter().map(|v| v[0]).collect::<Vec<_>>();
    let mut y_values = pitch_classes.iter().map(|v| v[1]).collect::<Vec<_>>();

    x_values.sort();
    y_values.sort();

    let x_min = x_values[0];
    let x_max = x_values[x_values.len() - 1];

    let y_min = y_values[0];
    let y_max = y_values[y_values.len() - 1];

    if (x_min + 1..=x_max - 1).any(|x| {
        pitch_classes
            .iter()
            .filter(|&v| v[0] == x)
            .collect::<Vec<_>>()
            .len()
            == 1
    }) {
        // If any nonperipheral x-coordinate has a singleton
        //    x x x x x x             =>   x x x x x x
        //              x x x x x            x x x x x
        let mut dxs: Vec<i32> = vec![];
        let mut prev_x_max = pitch_classes
            .iter()
            .filter(|v| v[1] == y_min)
            .map(|v| v[0])
            .sorted()
            .last()
            .unwrap();

        // Walk up in y and get all changes in max x value between rows
        for y in y_min + 1..=y_max {
            let this_x_max = pitch_classes
                .iter()
                .filter(|v| v[1] == y)
                .map(|v| v[0])
                .sorted()
                .last()
                .unwrap_or(prev_x_max);
            dxs.push(this_x_max - prev_x_max);
            prev_x_max = this_x_max;
        }

        // Let dx0 := the most common change in max x value
        // We want to deshear (dx0, 1) to (0, 1), so we want to make the new basis (1, 0), (dx0, 1)

        dxs.sort();

        let mut dx0_count: usize = 1;
        let mut dx0 = dxs[0];

        let mut curr_dx_count = 1;
        for i in 0..=dxs.len() - 1 {
            if i == dxs.len() - 1 || dxs[i + 1] > dxs[i] {
                if curr_dx_count > dx0_count {
                    dx0 = dxs[i];
                    dx0_count = curr_dx_count;
                }
                // reset curr_dx_count for next run
                curr_dx_count = 1;
            } else {
                curr_dx_count += 1;
            }
        }

        // Now change the basis
        let PitchClassLatticeBasis {
            vx: vx_lms,
            vy: vy_lms,
        } = old_basis;

        let new_vy = vec![
            dx0 * vx_lms[0] + vy_lms[0],
            dx0 * vx_lms[1] + vy_lms[1],
            dx0 * vx_lms[2] + vy_lms[2],
        ];

        PitchClassLatticeBasis::from_slices(vx_lms, &new_vy)
    } else if (y_min + 1..=y_max - 1).any(|y| {
        pitch_classes
            .iter()
            .filter(|&v| v[1] == y)
            .collect::<Vec<_>>()
            .len()
            == 1
    }) {
        // If any nonperipheral y-coordinate has a singleton
        let mut dys: Vec<i32> = vec![];
        let mut prev_y_max = pitch_classes
            .iter()
            .filter(|v| v[0] == x_min)
            .map(|v| v[1])
            .sorted()
            .last()
            .unwrap();

        // Walk up in x and get all changes in max y value between rows
        for x in x_min + 1..=x_max {
            let this_y_max = pitch_classes
                .iter()
                .filter(|v| v[0] == x)
                .map(|v| v[1])
                .sorted()
                .last()
                .unwrap_or(prev_y_max);
            dys.push(this_y_max - prev_y_max);
            prev_y_max = this_y_max;
        }

        // Let dy0 := the most common change in max y value
        // We want to deshear (1, dy0) to (1, 0), so we want to make the new basis (1, dy0), (0, 1)

        dys.sort();

        let mut dy0_count: usize = 1;
        let mut dy0 = dys[0];

        let mut curr_dy_count = 1;
        for i in 0..=dys.len() - 1 {
            if i == dys.len() - 1 || dys[i + 1] > dys[i] {
                if curr_dy_count > dy0_count {
                    dy0 = dys[i];
                    dy0_count = curr_dy_count;
                }
                // reset curr_dy_count for next run
                curr_dy_count = 1;
            } else {
                curr_dy_count += 1;
            }
        }

        // Now change the basis
        let PitchClassLatticeBasis {
            vx: vx_lms,
            vy: vy_lms,
        } = old_basis;

        let new_vx = vec![
            dy0 * vy_lms[0] + vx_lms[0],
            dy0 * vy_lms[1] + vx_lms[1],
            dy0 * vy_lms[2] + vx_lms[2],
        ];

        PitchClassLatticeBasis::from_slices(&new_vx, vy_lms)
    } else {
        // Default: just return the input lattice and basis with no change
        old_basis.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        // comb::partitions_exact_part_count,
        lattice::{
            /*ParallelogramSubstring, PitchClassLatticeBasis,*/
            // ParallelogramSubstring, get_unimodular_basis, parallelogram_info,
            parallelogram_substring_info, try_pitch_class_lattice,
        },
    };

    // use crate::words::mos_substitution_scales;
    // use std::fs;
    #[test]
    fn test_parallelogram_substring() {
        let eps = 1e-8;

        let diasem_rh = [0, 1, 0, 2, 0, 1, 0, 2, 0];
        let diasem_lattices_and_basis = try_pitch_class_lattice(&diasem_rh).unwrap();
        let diasem_rh_result = parallelogram_substring_info(
            &crate::helpers::slicify_each(&diasem_lattices_and_basis.0),
            &diasem_lattices_and_basis.1,
        );
        println!("{:?}", diasem_rh_result);
        assert!(diasem_rh_result.is_some());
        if let Some((ps, b)) = diasem_rh_result {
            // Diasem has this structure:
            //   x x x x
            //   x x x x x
            // or
            //   x x x x x
            //   x x x x
            // This is a full parallelogram with one note missing.
            // So it can be interpreted as both of the following:
            // 1. row_count == 2, full_row_len == 5, first and last row lengths == {5, 4}
            // 2. row_count == 5, full_row_len == 2, first and last row lengths == {2, 1}.
            assert!(
                (ps.row_count == 2 && ps.full_row_len == 5)
                    || (ps.row_count == 5 && ps.full_row_len == 2)
            );

            // Use JI diasem step sizes: L == 9/8, m == 28/27, s == 64/63
            // Assert that basis reduces to gener: 3/2 or 4/3
            // and offset: 7/6 (or complement) or 8/7 (or complement)
            let three_to_two = f64::log2(3.0 / 2.0) * 1200.0;
            let seven_to_six = f64::log2(7.0 / 6.0) * 1200.0;
            let eight_to_seven = f64::log2(8.0 / 7.0) * 1200.0;

            let nine_to_eight = f64::log2(9.0 / 8.0) * 1200.0;
            let twenty_eight_to_twenty_seven = f64::log2(28.0 / 27.0) * 1200.0;
            let sixty_four_to_sixty_three = f64::log2(64.0 / 63.0) * 1200.0;

            let g1_in_ji = (b.vx[0] as f64) * nine_to_eight
                + (b.vx[1] as f64) * twenty_eight_to_twenty_seven
                + (b.vx[2] as f64) * sixty_four_to_sixty_three;
            let g1_in_ji_reduced = f64::rem_euclid(g1_in_ji, 1200.0);
            let g2_in_ji = (b.vy[0] as f64) * nine_to_eight
                + (b.vy[1] as f64) * twenty_eight_to_twenty_seven
                + (b.vy[2] as f64) * sixty_four_to_sixty_three;
            let g2_in_ji_reduced = f64::rem_euclid(g2_in_ji, 1200.0);

            if ps.row_count == 2 && ps.full_row_len == 5 {
                assert!(
                    (g1_in_ji_reduced - three_to_two).abs() < eps
                        || (g1_in_ji_reduced - (1200.0 - three_to_two)).abs() < eps
                );
                assert!(
                    (g2_in_ji_reduced - seven_to_six).abs() < eps
                        || (g2_in_ji_reduced - (1200.0 - seven_to_six)).abs() < eps
                        || (g2_in_ji_reduced - eight_to_seven).abs() < eps
                        || (g2_in_ji_reduced - (1200.0 - eight_to_seven)).abs() < eps
                );
            } else {
                assert!(
                    (g1_in_ji_reduced - seven_to_six).abs() < eps
                        || (g1_in_ji_reduced - (1200.0 - seven_to_six)).abs() < eps
                        || (g1_in_ji_reduced - eight_to_seven).abs() < eps
                        || (g1_in_ji_reduced - (1200.0 - eight_to_seven)).abs() < eps
                );
                assert!(
                    (g2_in_ji_reduced - three_to_two).abs() < eps
                        || (g2_in_ji_reduced - (1200.0 - three_to_two)).abs() < eps
                );
            }
        }

        let blackdye = [0, 1, 0, 2, 0, 1, 0, 2, 0, 2];
        let blackdye_lattices_and_basis = try_pitch_class_lattice(&blackdye).unwrap();
        let blackdye_result = parallelogram_substring_info(
            &crate::helpers::slicify_each(&blackdye_lattices_and_basis.0),
            &blackdye_lattices_and_basis.1,
        );
        assert!(blackdye_result.is_some());
        if let Some((ps, b)) = blackdye_result {
            // Since this is a full parallelogram, the parallelogram itself doesn't give
            // a canonical choice of which is the "generator" and which is the "offset",
            // so we check the following:
            assert!(
                (ps.row_count == 2 && ps.full_row_len == 5)
                    || (ps.row_count == 5 && ps.full_row_len == 2)
            );

            // Use JI blackdye step sizes: L == 10/9, m == 16/15, s == 81/80
            // Assert that basis reduces to gener: 3/2 or 4/3 and offset: 10/9 or 9/5
            let three_to_two = f64::log2(3.0 / 2.0) * 1200.0;
            let four_to_three = f64::log2(4.0 / 3.0) * 1200.0;
            let nine_to_five = f64::log2(9.0 / 5.0) * 1200.0;

            let ten_to_nine = f64::log2(10.0 / 9.0) * 1200.0;
            let sixteen_to_fifteen = f64::log2(16.0 / 15.0) * 1200.0;
            let eighty_one_to_eighty = f64::log2(81.0 / 80.0) * 1200.0;
            let g1_in_ji = (b.vx[0] as f64) * ten_to_nine
                + (b.vx[1] as f64) * sixteen_to_fifteen
                + (b.vx[2] as f64) * eighty_one_to_eighty;
            let g1_in_ji_reduced = f64::rem_euclid(g1_in_ji, 1200.0);
            let g2_in_ji = (b.vy[0] as f64) * ten_to_nine
                + (b.vy[1] as f64) * sixteen_to_fifteen
                + (b.vy[2] as f64) * eighty_one_to_eighty;
            let g2_in_ji_reduced = f64::rem_euclid(g2_in_ji, 1200.0);
            if ps.row_count == 2 && ps.full_row_len == 5 {
                assert!(
                    (g1_in_ji_reduced - three_to_two).abs() < eps
                        || (g1_in_ji_reduced - four_to_three).abs() < eps
                );
                assert!(
                    (g2_in_ji_reduced - nine_to_five).abs() < eps
                        || (g2_in_ji_reduced - ten_to_nine).abs() < eps
                );
            } else {
                assert!(
                    (g1_in_ji_reduced - nine_to_five).abs() < eps
                        || (g1_in_ji_reduced - ten_to_nine).abs() < eps
                );
                assert!(
                    (g2_in_ji_reduced - three_to_two).abs() < eps
                        || (g2_in_ji_reduced - four_to_three).abs() < eps
                );
            }
        }

        let diaslen_4sc = [2, 0, 1, 0, 2, 0, 2, 0, 1, 0, 2]; // sLmLsLsLmLs
        let diaslen_lattices_and_basis = try_pitch_class_lattice(&diaslen_4sc).unwrap();
        let diaslen_result = parallelogram_substring_info(
            &crate::helpers::slicify_each(&diaslen_lattices_and_basis.0),
            &diaslen_lattices_and_basis.1,
        );
        assert!(diaslen_result.is_some());
        /*
        if let Some((ps, b)) = diaslen_result {
            // The structure we expect:
            //   x x x x
            //     x x x
            //     x x x x
            // This forces row_count == 5 and full_row_len == 3, first_row_len == 1, last_row_len == 1.
            //
            // However, this can still be interpreted in another way:
            // each row goes northwest/southeast, which is still row_count == 5, full_row_len == 3, first_row_len == 1, last_row_len == 1.
            // The latter interpretation means that each row is stacked by 32/21 and the offset is 3/2.
            assert!((ps.row_count == 5 && ps.full_row_len == 3));

            // Use JI diamech step sizes: L == 9/8, m == 49/48, s == 64/63
            // Assert that basis reduces to gener: 3/2 or 4/3 and offset: 8/7 or 7/4
            let three_to_two = f64::log2(3.0 / 2.0) * 1200.0;
            let four_to_three = f64::log2(4.0 / 3.0) * 1200.0;
            let seven_to_four = f64::log2(7.0 / 4.0) * 1200.0;
            let eight_to_seven = f64::log2(8.0 / 7.0) * 1200.0;

            let nine_to_eight = f64::log2(9.0 / 8.0) * 1200.0;
            let forty_nine_to_forty_eight = f64::log2(49.0 / 48.0) * 1200.0;
            let sixty_four_to_sixty_three = f64::log2(64.0 / 63.0) * 1200.0;

            let g1_in_ji = (b.v1[0] as f64) * nine_to_eight
                + (b.v1[1] as f64) * forty_nine_to_forty_eight
                + (b.v1[2] as f64) * sixty_four_to_sixty_three;
            let g1_in_ji_reduced = f64::rem_euclid(g1_in_ji, 1200.0);
            let g2_in_ji = (b.v2[0] as f64) * nine_to_eight
                + (b.v2[1] as f64) * forty_nine_to_forty_eight
                + (b.v2[2] as f64) * sixty_four_to_sixty_three;
            let g2_in_ji_reduced = f64::rem_euclid(g2_in_ji, 1200.0);

            // Accept either (3/2, 8/7) or (8/7, 3/2) ordering
            let has_fifth = (g1_in_ji_reduced - three_to_two).abs() < eps
                || (g1_in_ji_reduced - four_to_three).abs() < eps
                || (g2_in_ji_reduced - three_to_two).abs() < eps
                || (g2_in_ji_reduced - four_to_three).abs() < eps;
            let has_seventh = (g1_in_ji_reduced - seven_to_four).abs() < eps
                || (g1_in_ji_reduced - eight_to_seven).abs() < eps
                || (g2_in_ji_reduced - seven_to_four).abs() < eps
                || (g2_in_ji_reduced - eight_to_seven).abs() < eps;
            assert!(
                has_fifth && has_seventh,
                "Expected basis with 3/2 (or 4/3) and 8/7 (or 7/4), got {:.1}¢ and {:.1}¢",
                g1_in_ji_reduced,
                g2_in_ji_reduced
            );
        }
        */
        let diachrome_5sc = [0, 2, 0, 2, 0, 1, 2, 0, 2, 0, 2, 1]; // LsLsLmsLsLsm
        let diachrome_lattices_and_basis = try_pitch_class_lattice(&diachrome_5sc).unwrap();
        let diachrome_result = parallelogram_substring_info(
            &crate::helpers::slicify_each(&diachrome_lattices_and_basis.0),
            &diachrome_lattices_and_basis.1,
        );
        assert!(diachrome_result.is_some());

        let example_6 = [
            0, 0, 2, 1, 2, 0, 1, 2, 0, 2, 0, 1, 2, 0, 2, 1, 0, 2, 0, 2, 1, 0, 2, 1, 2,
        ]; // LLsmsLmsLsLmsLsmLsLsmLsms
        let lattices_and_basis = try_pitch_class_lattice(&example_6).unwrap();
        assert!(
            parallelogram_substring_info(
                &crate::helpers::slicify_each(&lattices_and_basis.0),
                &lattices_and_basis.1
            )
            .is_some()
        );

        let nonexample = [0, 0, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 0, 1, 2]; // LLLLsLmLLsLLLms
        let lattices_and_basis = try_pitch_class_lattice(&nonexample).unwrap();
        assert!(
            parallelogram_substring_info(
                &crate::helpers::slicify_each(&lattices_and_basis.0),
                &lattices_and_basis.1
            )
            .is_none()
        );

        let nonexample_2 = [0, 2, 0, 2, 0, 2, 1, 2, 0, 2, 0, 2, 1, 2, 2]; // LsLsLsmsLsLsmss
        let lattices_and_basis = try_pitch_class_lattice(&nonexample_2).unwrap();
        assert!(
            parallelogram_substring_info(
                &crate::helpers::slicify_each(&lattices_and_basis.0),
                &lattices_and_basis.1
            )
            .is_none()
        );
    }
}
