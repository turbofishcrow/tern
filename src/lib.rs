//! # Ternary
//!
//! A library for studying **ternary scales** — musical scales with exactly three step sizes.
//!
//! This crate provides algorithms for generating, analyzing, and tuning ternary scales,
//! with applications to microtonal music theory and xenharmonic scale research.
//!
//! ## Glossary
//!
//! ### Scale Basics
//!
//! - **Ternary scale**: A scale with exactly 3 distinct step sizes, conventionally labeled
//!   L (large), m (medium), and s (small). Example: "LLmLLms" is a 7-note ternary scale.
//!
//! - **Equave**: The interval of equivalence used by a scale; usually the octave, but sometimes 3/1 or other intervals.
//!
//! - **Step signature**: The multiset of step sizes in a scale, written as aLbmcs where
//!   a, b, c are counts. Example: 5L2m2s means 5 large, 2 medium, and 2 small steps.
//!
//! - **Word**: A scale represented as a sequence of step letters. Two words are equivalent
//!   if one is a rotation of the other (same scale, different starting note).
//!
//! - **Mode**: A rotation of a scale word, representing a different starting degree.
//!   The "brightest mode" starts from the position that gives the lexicographically
//!   smallest rotation.
//!
//! - **Necklace**: An equivalence class of words under rotation. Used to enumerate
//!   distinct scales without counting rotations as different.
//!
//! ### Scale Properties
//!
//! - **Maximum variety (MV)**: The maximum number of distinct interval sizes for any
//!   fixed number of steps. MV=2 scales are called MOS (Moment of Symmetry).
//!
//! - **Constant structure (CS)**: A scale where no interval appears in two different
//!   interval classes. If 3/2 is a 4-step somewhere, it's a 4-step everywhere.
//!
//! - **Chirality**: Scale handedness. A scale is *achiral* if it equals its reversal
//!   (under rotation), *left-handed* if lexicographically greater than its reversal,
//!   *right-handed* otherwise.
//!
//! - **Monotone-MOS**: Conditions where the scale is required to be a MOS
//!   when two step sizes coincide in a way that preserves the ordering on step sizes.
//!   L=m monotone, m=s monotone, and s=0 monotone are the three types.
//!
//! ### Guided Generator Sequences
//!
//! - **Generator sequence (GS)**: A sequence of intervals that, when stacked and
//!   octave-reduced, produces a scale.
//!
//! - **Guided Generator Sequence (GGS)**: A GS that produces a detempered MOS subscale
//!   of the original ternary scale. See Keenan Pepper's work on GGS.
//!
//! - **Guide frame**: A GGS together with offset information. Multiple guide frames
//!   can describe interleaved scale structures.
//!
//! - **Complexity**: The length of the shortest GGS for a scale. Lower complexity
//!   suggests simpler melodic structure.
//!
//! ### Just Intonation
//!
//! - **Monzo**: A vector of prime exponents representing a JI ratio.
//!   Example: `[-1, 1]` = 3/2 = 2⁻¹ × 3¹.
//!
//! - **Odd limit**: JI intervals with numerator and denominator ≤ n after factoring out
//!   all factors of 2.
//!
//! - **Cumulative form**: Scale as intervals from tonic: `[9/8, 5/4, 4/3, ...]`.
//!
//! - **Step form**: Scale as consecutive steps: `[9/8, 10/9, 16/15, ...]`.
//!
//! ### Equal Temperament
//!
//! - **ED (equal division)**: Division of an equave into equal steps.
//!   "12edo" = 12 equal divisions of the octave.
//!
//! - **Val**: A covector mapping JI intervals to ET steps. The *patent val* maps
//!   each prime to its nearest integer step count.
//!
//! - **Tuning range**: For ternary scales, the valid tunings based on degenerate
//!   cases where step sizes collapse (L=m, m=s, or s=0).
//!
//! ### Lattice Visualization
//!
//! - **Pitch class lattice**: A 2D projection of scale degrees using a unimodular
//!   basis including the equave. Useful for visualizing scale structure.
//!
//! - **Unimodular basis**: A list of step-count vectors with determinant ±1.
//!
//! ## Module Overview
//!
//! - [`words`]: Scale representation and word operations
//! - [`mod@monzo`]: Prime-factorized JI intervals
//! - [`ji_ratio`]: JI ratio arithmetic
//! - [`ji`]: JI scale analysis and tuning solvers
//! - [`equal`]: Equal temperament calculations
//! - [`guide`]: Guided Generator Sequences
//! - [`comb`]: Necklace enumeration
//! - [`lattice`]: Pitch class lattice visualization

// #![deny(warnings)]
pub mod comb;
#[macro_use]
pub mod equal;
pub mod guide;
pub mod helpers;
pub mod interval;
pub mod ji;
pub mod ji_ratio;
pub mod lattice;
pub mod matrix;
#[macro_use]
pub mod monzo;
pub mod interpretations;
pub mod primes;
pub mod vector;
pub mod words;

use interval::JiRatio;
use itertools::Itertools;
use ji_ratio::RawJiRatio;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
use words::Chirality;
use words::Letter;
use words::{chirality, is_mos_subst_one_perm, word_on_degree};

#[cfg(feature = "wasm")]
#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(feature = "wasm")]
#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

const STEP_LETTERS: [&str; 12] = [
    "",                                                     // 0
    "X",                                                    // 1
    "Ls",                                                   // 2
    "Lms",                                                  // 3
    "Lmns",                                                 // 4
    "HLmns",                                                // 5
    "HLmnst",                                               // 6
    "BHLmnst",                                              // 7
    "BHLmnstw",                                             // 8
    "BCHLmnstw",                                            // 9
    "BCHLmnpstw",                                           // 10
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", // >= 11
];

use std::cmp::Ordering;
use std::cmp::min;
use std::collections::HashSet;

use serde::Serialize;
#[cfg(feature = "wasm")]
use serde_wasm_bindgen::to_value;

use guide::GuideFrame;
use guide::guide_frames;
#[cfg(feature = "wasm")]
use words::maximum_variety_is;
use words::{CountVector, least_mode, maximum_variety, monotone_lm, monotone_ms, monotone_s0};

use crate::ji::solve_step_sig_fast;
#[cfg(feature = "wasm")]
use crate::lattice::deshear;
use crate::lattice::get_unimodular_basis;
use crate::monzo::Monzo;
use crate::words::mos_substitution_scales;

/// Compute the determinant of a 3x3 matrix formed by three row vectors.
/// Used to check if vectors form a unimodular basis (determinant ±1).
// A representation of a GuideFrame that should be WASM-readable
#[derive(Clone, Debug, Serialize)]
pub struct GuideResult {
    /// Either Guided GS or multiple interleaved Guided GSes
    /// `guided_gs` generates a guided generator sequence (detempered single-period MOS) subscale.
    /// The `JsValue` is an array of 3 numbers where each entry is the count of the corresp. step size.
    pub gs: Vec<Vec<u16>>,
    /// The aggregate generator
    pub aggregate: Vec<u16>,
    /// `offset_chord` is the set of intervals that each guided generator sequence chain is based on. Always includes the unison.
    /// The `JsValue` is an array of 3 numbers where each entry is the count of the corresp. step size.
    pub offset_chord: Vec<Vec<u16>>,
    /// complexity result
    /// The base GS chains in a multiple GS structure don't form interleaved scales. Instead they form a detempered copy of m-ed.
    pub multiplicity: u16,
    pub complexity: u16,
}

/// A representation of a Scale Profile. Doesn't include tunings.
#[derive(Debug, Serialize)]
pub struct ScaleProfile {
    /// brightest word
    word: String,
    /// unimodular basis for lattice is there is one
    lattice_basis: Option<Vec<Vec<i32>>>,
    /// chirality
    chirality: Chirality,
    /// brightest mode of reversed word
    reversed: String,
    /// lowest-complexity guide frame structure provided there is one
    structure: Option<GuideResult>,
    /// whether scale is L=m monotone MOS
    lm: bool,
    /// whether scale is m=s monotone MOS
    ms: bool,
    /// whether scale is s=0 monotone MOS
    s0: bool,
    /// whether scale is a subst aL(bmcs)
    subst_l_ms: bool,
    /// whether scale is a subst bm(aLcs)
    subst_m_ls: bool,
    /// whether scale is a subst cs(aLbm)
    subst_s_lm: bool,
    /// Temperament-agnostic ed join
    ed_join: (i32, i32, i32),
    /// maximum variety of scale
    mv: u16,
}

#[derive(Debug, Serialize)]
pub struct SigResult {
    profiles: Vec<ScaleProfile>,
    ji_tunings: Vec<Vec<String>>,
    ed_tunings: Vec<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct WordResult {
    profile: ScaleProfile,
    ji_tunings: Vec<Vec<String>>,
    ed_tunings: Vec<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct LatticeResult {
    coordinates: Vec<Vec<i32>>,
    basis: Vec<Vec<i16>>,
}

#[cfg(feature = "wasm")]
fn string_to_numbers(word: &str) -> Vec<usize> {
    let mut result = vec![];
    let arity = word.chars().collect::<HashSet<_>>().len();
    for c in word.chars() {
        if let Some(letter) = STEP_LETTERS[min(arity, 12)].find(c) {
            result.push(letter);
        }
    }
    result
}

fn word_to_sig(input: &[usize]) -> Vec<usize> {
    let mut result = vec![0, 0, 0];
    for &i in input {
        if i < 3 {
            result[i] += 1;
        }
    }
    result
}

fn numbers_to_string(word: &[usize]) -> String {
    let mut result = "".to_string();
    let arity = word.iter().collect::<HashSet<_>>().len();
    for i in word {
        if *i <= arity {
            result.push(STEP_LETTERS[min(arity, 12)].chars().nth(*i).unwrap_or('?'));
        }
    }
    result
}

/// Convert a CountVector to a 3-element u16 vector for serialization
fn countvector_to_u16_vec(count_vector: &CountVector<usize>) -> Vec<u16> {
    let btreemap = count_vector.into_inner();
    vec![
        *btreemap.get(&0).unwrap_or(&0) as u16,
        *btreemap.get(&1).unwrap_or(&0) as u16,
        *btreemap.get(&2).unwrap_or(&0) as u16,
    ]
}

fn guide_frame_to_result(structure: &GuideFrame) -> GuideResult {
    let GuideFrame { gs, offset_chord } = structure;
    let aggregate_cv: CountVector<usize> = gs
        .iter()
        .fold(CountVector::<usize>::ZERO, |acc, v| acc.add(v));

    GuideResult {
        gs: gs.iter().map(countvector_to_u16_vec).collect(),
        aggregate: countvector_to_u16_vec(&aggregate_cv),
        offset_chord: offset_chord.iter().map(countvector_to_u16_vec).collect(),
        multiplicity: structure.multiplicity() as u16,
        complexity: structure.complexity() as u16,
    }
}

pub fn word_to_profile(query: &[usize]) -> ScaleProfile {
    let brightest = numbers_to_string(&least_mode(query));
    let lm = monotone_lm(query);
    let ms = monotone_ms(query);
    let s0 = monotone_s0(query);
    let chirality = chirality(query);
    let reversed = least_mode(&query.iter().copied().rev().collect::<Vec<usize>>());
    let reversed = numbers_to_string(&reversed);
    let mv = maximum_variety(query) as u16;
    let step_sig = word_to_sig(query)
        .iter()
        .map(|x| *x as i32)
        .collect::<Vec<i32>>();
    let (n_l, n_m, n_s) = (step_sig[0], step_sig[1], step_sig[2]);
    let ed_join = (
        3 * n_l + 2 * n_m + n_s,
        4 * n_l + 2 * n_m + n_s,
        4 * n_l + 3 * n_m + n_s,
    );
    let subst_l_ms = is_mos_subst_one_perm(query, 0, 1, 2);
    let subst_m_ls = is_mos_subst_one_perm(query, 1, 0, 2);
    let subst_s_lm = is_mos_subst_one_perm(query, 2, 0, 1);
    if let Some(pair) = get_unimodular_basis(&guide_frames(query), &step_sig) {
        let (lattice_basis, structure) = pair;
        ScaleProfile {
            word: brightest,
            lattice_basis: Some(lattice_basis),
            // ploidacot: Ploidacot::try_get_ploidacot(query),
            chirality,
            reversed,
            structure: Some(structure),
            lm,
            ms,
            s0,
            subst_l_ms,
            subst_m_ls,
            subst_s_lm,
            ed_join,
            mv,
        }
    } else {
        ScaleProfile {
            word: brightest,
            lattice_basis: None,
            // ploidacot: Ploidacot::try_get_ploidacot(query),
            chirality,
            reversed,
            structure: None,
            lm,
            ms,
            s0,
            subst_l_ms,
            subst_m_ls,
            subst_s_lm,
            ed_join,
            mv,
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn word_result(
    query: String,
    equave_num: u32,
    equave_den: u32,
    ed_bound: i32,
    s_lower: f64,
    s_upper: f64,
) -> Result<JsValue, JsValue> {
    let equave = RawJiRatio::try_new(equave_num, equave_den).unwrap_or(RawJiRatio::OCTAVE);
    let word_as_numbers = string_to_numbers(&query);
    let step_sig = word_to_sig(&word_as_numbers);

    Ok(to_value(&WordResult {
        profile: word_to_profile(&word_as_numbers),
        ji_tunings: sig_to_ji_tunings(&step_sig, equave, s_lower, s_upper),
        ed_tunings: sig_to_ed_tunings(&step_sig, equave, ed_bound, s_lower, s_upper),
    })?)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn word_to_brightest(query: String) -> String {
    let word_in_numbers = string_to_numbers(&query);
    let brightest = least_mode(&word_in_numbers);
    numbers_to_string(&brightest)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn word_to_mv(query: String) -> u16 {
    let word_in_numbers = string_to_numbers(&query);
    maximum_variety(&word_in_numbers) as u16
}

/// Get lattice coordinates for pitch classes if a unimodular basis exists.
/// Returns None if no unimodular basis can be found.
/// The coordinates are 2D projections suitable for plotting.
/// Prioritizes the basis from parallelogram_substring_info if one exists.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn word_to_lattice(query: String) -> Result<JsValue, JsValue> {
    let word_in_numbers = string_to_numbers(&query);
    let step_sig = word_to_sig(&word_in_numbers)
        .iter()
        .map(|x| *x as i32)
        .collect::<Vec<i32>>();
    // First get the initial lattice and basis
    if let Some((pitch_classes, initial_basis)) = lattice::try_pitch_class_lattice(&word_in_numbers)
    {
        // Try to find a better basis using parallelogram_substring_info
        let pitch_class_refs: Vec<&[i32]> = pitch_classes.iter().map(|v| v.as_slice()).collect();

        let (final_coordinates, final_basis) = if let Some((_qp, better_basis)) =
            lattice::parallelogram_substring_info(&pitch_class_refs, &initial_basis)
        {
            // Re-project pitch classes using the better basis
            let (coords, _) = lattice::pitch_classes(&word_in_numbers, &better_basis);
            // Equave reduce the basis
            let better_basis_rd = better_basis.equave_reduce(&step_sig);
            (coords, better_basis_rd)
        } else {
            // No better basis found, use deshear()
            let desheared_basis = lattice::deshear(&pitch_classes, &initial_basis);
            // Re-project pitch classes using the better basis
            let (coords, _) = lattice::pitch_classes(&word_in_numbers, &desheared_basis);
            // Equave reduce the basis
            let desheared_basis_rd = desheared_basis.equave_reduce(&step_sig);
            (coords, desheared_basis_rd)
        };

        // Convert the basis to Vec<Vec<i16>> for serialization
        let basis_as_vecs = vec![
            vec![
                final_basis.vx()[0] as i16,
                final_basis.vx()[1] as i16,
                final_basis.vx()[2] as i16,
            ],
            vec![
                final_basis.vy()[0] as i16,
                final_basis.vy()[1] as i16,
                final_basis.vy()[2] as i16,
            ],
        ];

        Ok(to_value(&Some(LatticeResult {
            coordinates: final_coordinates,
            basis: basis_as_vecs,
        }))?)
    } else {
        // No unimodular basis found - return early
        Ok(to_value(&None::<LatticeResult>)?)
    }
}

// Dumb scoring function for tunings, higher is better
fn score(word: &[Letter], tuning: &[Monzo]) -> f64 {
    let (l_monzo, m_monzo, s_monzo) = (tuning[0], tuning[1], tuning[2]);
    let mut score = 0.0;
    // For every interval in the interval matrix
    for degree in 0..word.len() {
        for ordinal in 1..word.len() {
            let subword = word_on_degree(word, degree, ordinal);
            let interval_monzo: Monzo = subword
                .into_iter()
                .map(|i| match i {
                    0 => l_monzo,
                    1 => m_monzo,
                    2 => s_monzo,
                    _ => Monzo::UNISON,
                })
                .sum();
            let interval_monzo_odds_only = interval_monzo - monzo![interval_monzo[0]];
            score += if let Some(ratio) = interval_monzo_odds_only.try_to_ratio() {
                // Simpler ratios => lower exponent => higher number
                0.9f64.powf(ratio.numer() as f64 + ratio.denom() as f64)
            } else {
                0.0
            };
        }
    }
    score
}

fn sort_scales(scale1: &[Letter], scale2: &[Letter]) -> Ordering {
    let complexity1 = if let Some(simplest) = guide_frames(scale1).first() {
        simplest.complexity()
    } else {
        usize::MAX
    };
    let ps1 = if let Some((pitch_classes, initial_basis)) = lattice::try_pitch_class_lattice(scale1)
    {
        let pitch_class_refs: Vec<&[i32]> = pitch_classes.iter().map(|v| v.as_slice()).collect();
        if let Some((ps1, _)) =
            lattice::parallelogram_substring_info(&pitch_class_refs, &initial_basis)
        {
            Some(ps1)
        } else {
            None
        }
    } else {
        None
    };
    let complexity2 = if let Some(simplest) = guide_frames(scale2).first() {
        simplest.complexity()
    } else {
        usize::MAX
    };
    let ps2 = if let Some((pitch_classes, initial_basis)) = lattice::try_pitch_class_lattice(scale2)
    {
        let pitch_class_refs: Vec<&[i32]> = pitch_classes.iter().map(|v| v.as_slice()).collect();
        if let Some((ps1, _)) =
            lattice::parallelogram_substring_info(&pitch_class_refs, &initial_basis)
        {
            Some(ps1)
        } else {
            None
        }
    } else {
        None
    };
    // Reverse ps1 and ps2 since None < Some and we want Some to come first
    ps2.cmp(&ps1).then(complexity1.cmp(&complexity2))
}

/// Get JI tunings for a step signature using 81-odd-limit intervals.
/// This is not an exhaustive search - it only considers intervals < 300 cents
/// and requires steps to be strictly descending in size.
pub fn sig_to_ji_tunings(
    step_sig: &[usize],
    equave: RawJiRatio,
    cents_lower_bound: f64,
    cents_upper_bound: f64,
) -> Vec<Vec<String>> {
    let best_scale = mos_substitution_scales(step_sig)
        .into_iter()
        .sorted_by(|scale1, scale2| sort_scales(scale1, scale2))
        .next()
        .unwrap();
    let equave_monzo = Monzo::try_from_ratio(equave).ok();
    if let Some(equave_monzo) = equave_monzo {
        let mut tunings =
            solve_step_sig_fast(step_sig, equave_monzo, cents_lower_bound, cents_upper_bound);

        tunings.sort_by(|tuning1, tuning2| {
            let tuning1_score = score(&best_scale, tuning1);
            let tuning2_score = score(&best_scale, tuning2);
            tuning2_score.total_cmp(&tuning1_score) // Higher is better
        });
        tunings.dedup();
        tunings
            .into_iter()
            .map(|steps| {
                steps
                    .into_iter()
                    .map(|m| {
                        m.try_to_ratio()
                            .map(|r| r.to_string())
                            .unwrap_or_else(|| m.to_string())
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    }
}

/// Get more JI tunings using the slow solver (shifts by 270edo commas).
/// Returns tunings that are NOT already in the fast solver results.
pub fn sig_to_ji_tunings_slow(
    step_sig: &[usize],
    equave: RawJiRatio,
    cents_lower_bound: f64,
    cents_upper_bound: f64,
) -> Vec<Vec<String>> {
    let best_scale = mos_substitution_scales(step_sig)
        .into_iter()
        .sorted_by_key(|scale| {
            let gfs = guide_frames(scale);
            if let Some(simplest) = gfs.first() {
                simplest.complexity()
            } else {
                usize::MAX
            }
        })
        .next()
        .unwrap();
    if let Ok(equave_monzo) = Monzo::try_from_ratio(equave) {
        let slow =
            ji::solve_step_sig_slow(step_sig, equave_monzo, cents_lower_bound, cents_upper_bound);
        let fast =
            ji::solve_step_sig_fast(step_sig, equave_monzo, cents_lower_bound, cents_upper_bound);
        let mut union_ = [slow, fast].concat();
        union_.sort_by(|tuning1, tuning2| {
            let tuning1_score = score(&best_scale, tuning1);
            let tuning2_score = score(&best_scale, tuning2);
            tuning2_score.total_cmp(&tuning1_score) // Higher is better
        });
        union_.dedup();
        union_
            .into_iter()
            .map(|steps| {
                steps
                    .into_iter()
                    .map(|m| {
                        m.try_to_ratio()
                            .map(|r| r.to_string())
                            .unwrap_or_else(|| m.to_string())
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    }
}

pub fn sig_to_ed_tunings(
    step_sig: &[usize],
    equave: RawJiRatio,
    ed_bound: i32,
    s_lower: f64,
    s_upper: f64,
) -> Vec<Vec<String>> {
    let ed_tunings =
        crate::equal::ed_tunings_for_ternary(step_sig, equave, ed_bound, s_lower, s_upper);
    ed_tunings
        .into_iter()
        .map(|v| {
            let ed: i32 = v
                .iter()
                .enumerate()
                .map(|(i, steps)| step_sig[i] as i32 * steps)
                .sum();
            if equave == RawJiRatio::OCTAVE {
                v.iter().map(|i| format!("{i}\\{ed}")).collect::<Vec<_>>()
            } else {
                v.iter()
                    .map(|i| format!("{i}\\{ed}<{}/{}>", equave.numer(), equave.denom()))
                    .collect::<Vec<_>>()
            }
        })
        .collect::<Vec<_>>()
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn sig_result(
    query: Vec<u8>,
    lm: bool,
    ms: bool,
    s0: bool,
    ggs_len: u8,
    ggs_len_constraint: String,
    mv: u8,
    mv_constraint: String,
    scale_type: String,
    equave_num: u32,
    equave_den: u32,
    ed_bound: i32,
    s_lower: f64,
    s_upper: f64,
) -> Result<JsValue, JsValue> {
    let equave = RawJiRatio::try_new(equave_num, equave_den).unwrap_or(RawJiRatio::OCTAVE);
    let step_sig = query;
    let filtering_cond = |scale: &[Letter]| {
        (!lm || monotone_lm(scale))
            && (!ms || monotone_ms(scale))
            && (!s0 || monotone_s0(scale))
            && (match ggs_len {
                0 => true,
                l => {
                    let guide_frames = guide_frames(scale);
                    if ggs_len_constraint == "exactly" {
                        !guide_frames.is_empty() && guide_frames[0].gs.len() == l as usize
                    } else {
                        !guide_frames.is_empty() && guide_frames[0].gs.len() <= l as usize
                    }
                }
            })
            && (match mv {
                0 => true,
                mv => {
                    if mv_constraint == "exactly" {
                        maximum_variety_is(scale, mv as usize)
                    } else {
                        maximum_variety(scale) <= mv as usize
                    }
                }
            })
    };
    let step_sig = step_sig.iter().map(|x| *x as usize).collect::<Vec<_>>();
    let scales = if scale_type == "mos-subst" {
        words::mos_substitution_scales(&step_sig)
    } else {
        crate::comb::necklaces_fixed_content(&step_sig)
    }; // Now filter
    let scales = scales
        .into_iter()
        .filter(|scale| filtering_cond(scale))
        .collect::<Vec<_>>();
    Ok(to_value(&SigResult {
        profiles: scales
            .iter()
            // prioritize parallelogram substring scales
            .sorted_by(|scale1, scale2| sort_scales(scale1, scale2))
            .map(|scale| word_to_profile(scale))
            .collect(),
        ji_tunings: sig_to_ji_tunings(&step_sig, equave, s_lower, s_upper),
        ed_tunings: sig_to_ed_tunings(&step_sig, equave, ed_bound, s_lower, s_upper),
    })?)
}

/// Get more JI tunings using the slow solver (shifts by 270edo commas).
/// Used by the "more-sols" button in the UI.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn more_ji_tunings(
    step_sig: Vec<u8>,
    equave_num: u32,
    equave_den: u32,
    s_lower: f64,
    s_upper: f64,
) -> Result<JsValue, JsValue> {
    let equave = RawJiRatio::try_new(equave_num, equave_den).unwrap_or(RawJiRatio::OCTAVE);
    let step_sig = step_sig.iter().map(|x| *x as usize).collect::<Vec<_>>();
    Ok(to_value(&sig_to_ji_tunings_slow(
        &step_sig, equave, s_lower, s_upper,
    ))?)
}
