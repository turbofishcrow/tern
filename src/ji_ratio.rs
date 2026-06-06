//! Operations on Just Intonation ratios.
//!
//! This module provides [`RawJiRatio`], a simple representation of JI ratios
//! as numerator/denominator pairs. It supports:
//!
//! - Arithmetic operations (stacking, unstacking intervals)
//! - Octave reduction and other equave reductions
//! - Conversion to cents
//! - Built-in scale constants (Pythagorean, Zarlino, etc.)
//!
//! # Examples
//!
//! ```
//! use tern::ji_ratio::RawJiRatio;
//! use tern::interval::Dyad;
//!
//! // Create ratios
//! let fifth = RawJiRatio::PYTH_5TH;     // 3/2
//! let fourth = RawJiRatio::PYTH_4TH;    // 4/3
//!
//! // Arithmetic
//! let octave = fifth.stack(fourth);      // 3/2 × 4/3 = 2/1
//! assert_eq!(octave, RawJiRatio::OCTAVE);
//!
//! // Octave reduction
//! let ninth = fifth.stack(fifth);        // 9/4
//! let reduced = ninth.rd(RawJiRatio::OCTAVE);  // 9/8
//! assert_eq!(reduced.cents().round(), 204.0);
//! ```
//!
//! # Comparison with [`Monzo`](crate::monzo::Monzo)
//!
//! - `RawJiRatio`: Simple numerator/denominator, good for display and small ratios
//! - `Monzo`: Prime-exponent vector, better for arithmetic and avoiding overflow

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Div, DivAssign, Mul, MulAssign};

use num_traits::{CheckedDiv, CheckedMul};

use crate::helpers::gcd;
use crate::interval::{Dyad, JiRatio};

// ERRORS

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
/// Error type for attempts to construct a RawJiRatio from a non-positive ratio.
pub struct IllegalJiRatio {
    // Show what the attempt was
    numer: u32,
    denom: u32,
}

impl fmt::Display for IllegalJiRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tried to create invalid JI ratio {}/{}",
            self.numer, self.denom
        )
    }
}

impl std::error::Error for IllegalJiRatio {}

/// Error type for invalid outputs of JI interval arithmetic.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BadJiArith {
    /// logarithmic division by 0
    LogDivByUnison,
}

impl std::fmt::Display for BadJiArith {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            &Self::LogDivByUnison => {
                write!(f, "tried to log divide a JI ratio by the unison")
            }
        }
    }
}

impl std::error::Error for BadJiArith {}

// STRUCTS

/// A Just Intonation ratio represented as numerator/denominator.
///
/// Automatically reduces to lowest terms. Implements `Copy` for efficiency.
///
/// # Examples
///
/// ```
/// use tern::ji_ratio::RawJiRatio;
/// use tern::interval::{Dyad, JiRatio};
///
/// // Create and verify reduction to lowest terms
/// let ratio = RawJiRatio::try_new(6, 4).unwrap();
/// assert_eq!(ratio.numer(), 3);
/// assert_eq!(ratio.denom(), 2);
///
/// // Use constants for common intervals
/// let fifth = RawJiRatio::PYTH_5TH;
/// assert!((fifth.cents() - 701.96).abs() < 0.01);
/// ```
#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct RawJiRatio {
    numer: u32,
    denom: u32,
}

impl JiRatio for RawJiRatio {
    fn numer(&self) -> u32 {
        self.numer
    }
    fn denom(&self) -> u32 {
        self.denom
    }
}

impl PartialEq for RawJiRatio {
    fn eq(&self, other: &Self) -> bool {
        // Compare ratios by cross-multiplication to avoid floating point
        self.numer * other.denom == other.numer * self.denom
    }
}

impl Eq for RawJiRatio {}

impl PartialOrd for RawJiRatio {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Delegate to Ord impl which handles all cases
        Some(self.cmp(other))
    }
}

impl std::iter::Product for RawJiRatio {
    fn product<I: Iterator<Item = RawJiRatio>>(iter: I) -> Self {
        // Multiply all ratios in iterator, starting from unison (1/1)
        iter.fold(RawJiRatio::UNISON, |x, y| x * y)
    }
}

impl Ord for RawJiRatio {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare ratios by cross-multiplication: a/b < c/d iff a*d < b*c
        (self.numer * other.denom).cmp(&(other.numer * self.denom))
    }
}

impl RawJiRatio {
    /// The Pythagorean major pentatonic scale.
    pub const PYTH_5: [RawJiRatio; 5] = [
        RawJiRatio { numer: 9, denom: 8 },
        RawJiRatio {
            numer: 81,
            denom: 64,
        },
        RawJiRatio::PYTH_5TH,
        RawJiRatio {
            numer: 27,
            denom: 16,
        },
        RawJiRatio::OCTAVE,
    ];

    /// The Pythagorean Ionian mode.
    pub const PYTH_7: [RawJiRatio; 7] = [
        RawJiRatio { numer: 9, denom: 8 },
        RawJiRatio {
            numer: 81,
            denom: 64,
        },
        RawJiRatio::PYTH_4TH,
        RawJiRatio::PYTH_5TH,
        RawJiRatio {
            numer: 27,
            denom: 16,
        },
        RawJiRatio {
            numer: 243,
            denom: 128,
        },
        RawJiRatio::OCTAVE,
    ];

    /// The JI scale 12:14:16:18:21:24.
    pub const TAS_5: [RawJiRatio; 5] = [
        RawJiRatio { numer: 7, denom: 6 },
        RawJiRatio::PYTH_4TH,
        RawJiRatio::PYTH_5TH,
        RawJiRatio { numer: 7, denom: 4 },
        RawJiRatio::OCTAVE,
    ];

    /// The septal diasem scale.
    pub const TAS_9: [RawJiRatio; 9] = [
        RawJiRatio { numer: 9, denom: 8 },
        RawJiRatio { numer: 7, denom: 6 },
        RawJiRatio {
            numer: 21,
            denom: 16,
        },
        RawJiRatio::PYTH_4TH,
        RawJiRatio::PYTH_5TH,
        RawJiRatio {
            numer: 14,
            denom: 9,
        },
        RawJiRatio { numer: 7, denom: 4 },
        RawJiRatio {
            numer: 16,
            denom: 9,
        },
        RawJiRatio::OCTAVE,
    ];

    /// The pental blackdye scale.
    pub const BLACKDYE: [RawJiRatio; 10] = [
        RawJiRatio {
            numer: 10,
            denom: 9,
        },
        RawJiRatio { numer: 9, denom: 8 },
        RawJiRatio { numer: 6, denom: 5 },
        RawJiRatio::PYTH_4TH,
        RawJiRatio {
            numer: 27,
            denom: 20,
        },
        RawJiRatio::PYTH_5TH,
        RawJiRatio { numer: 8, denom: 5 },
        RawJiRatio {
            numer: 16,
            denom: 9,
        },
        RawJiRatio { numer: 9, denom: 5 },
        RawJiRatio::OCTAVE,
    ];

    /// The pental Zarlino scale, aka Ptolemy's intense diatonic.
    pub const ZARLINO: [RawJiRatio; 7] = [
        RawJiRatio { numer: 9, denom: 8 },
        RawJiRatio { numer: 5, denom: 4 },
        RawJiRatio::PYTH_4TH,
        RawJiRatio::PYTH_5TH,
        RawJiRatio { numer: 5, denom: 3 },
        RawJiRatio {
            numer: 15,
            denom: 8,
        },
        RawJiRatio::OCTAVE,
    ];

    /// The septal version of Zarlino, modifying Pythagorean intervals by 64/63 rather than flattening them by 81/80.
    pub const ARCHYLINO: [RawJiRatio; 7] = [
        RawJiRatio { numer: 9, denom: 8 },
        RawJiRatio { numer: 9, denom: 7 },
        RawJiRatio::PYTH_4TH,
        RawJiRatio::PYTH_5TH,
        RawJiRatio {
            numer: 12,
            denom: 7,
        },
        RawJiRatio {
            numer: 27,
            denom: 14,
        },
        RawJiRatio::OCTAVE,
    ];

    /// Zhea Erose's Eurybia scale.
    pub const EURYBIA: [RawJiRatio; 12] = [
        RawJiRatio {
            numer: 23,
            denom: 22,
        },
        RawJiRatio {
            numer: 25,
            denom: 22,
        },
        RawJiRatio {
            numer: 13,
            denom: 11,
        },
        RawJiRatio {
            numer: 14,
            denom: 11,
        },
        RawJiRatio {
            numer: 15,
            denom: 11,
        },
        RawJiRatio {
            numer: 31,
            denom: 22,
        },
        RawJiRatio::PYTH_5TH,
        RawJiRatio {
            numer: 35,
            denom: 22,
        },
        RawJiRatio {
            numer: 37,
            denom: 22,
        },
        RawJiRatio {
            numer: 39,
            denom: 22,
        },
        RawJiRatio {
            numer: 21,
            denom: 11,
        },
        RawJiRatio::OCTAVE,
    ];

    /// Multiply all ratios in iterator, returning None if any multiplication overflows.
    pub fn checked_product<I: Iterator<Item = RawJiRatio>>(iter: &mut I) -> Option<Self> {
        iter.try_fold(RawJiRatio::UNISON, |x, y| x.checked_mul(&y))
    }

    /// Raise the ratio to an integer power, returning None if any operation overflows.
    pub fn checked_pow(&self, n: i32) -> Option<Self> {
        if n >= 0 {
            // Positive exponent: multiply self n times
            let mut result = Some(Self::UNISON);
            for _ in 0..n {
                result = result?.checked_mul(self);
                result?;
            }
            result
        } else {
            // Negative exponent: divide by self |n| times
            let mut result = Some(Self::UNISON);
            for _ in 0..-n {
                result = result?.checked_div(self);
                result?;
            }
            result
        }
    }

    /// Creates a new `RawJiRatio`, validating and reducing to lowest terms.
    ///
    /// # Errors
    ///
    /// Returns [`IllegalJiRatio`] if numerator or denominator is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use tern::ji_ratio::RawJiRatio;
    /// use tern::interval::JiRatio;
    ///
    /// // Valid ratio, automatically reduced
    /// let ratio = RawJiRatio::try_new(12, 8).unwrap();
    /// assert_eq!(ratio.numer(), 3);
    /// assert_eq!(ratio.denom(), 2);
    ///
    /// // Zero denominator is invalid
    /// assert!(RawJiRatio::try_new(5, 0).is_err());
    /// ```
    #[inline(always)]
    pub fn try_new(numer: u32, denom: u32) -> Result<RawJiRatio, IllegalJiRatio> {
        if (denom == 0) || (numer == 0) {
            // Reject non-positive ratios
            Err(IllegalJiRatio { numer, denom })
        } else {
            // Reduce to canonical form by dividing both by their GCD
            let d = gcd(numer, denom);
            Ok(RawJiRatio {
                numer: numer / d,
                denom: denom / d,
            })
        }
    }
    /// The reciprocal of a `RawJiRatio`.
    #[inline(always)]
    pub fn reciprocal(&self) -> RawJiRatio {
        RawJiRatio {
            numer: self.denom,
            denom: self.numer,
        }
    }
    /// 1/1, the unison.
    pub const UNISON: RawJiRatio = RawJiRatio { numer: 1, denom: 1 };
    /// 2/1, the octave.
    pub const OCTAVE: RawJiRatio = RawJiRatio { numer: 2, denom: 1 };
    /// 3/2, the Pythagorean perfect fifth.
    pub const PYTH_5TH: RawJiRatio = RawJiRatio { numer: 3, denom: 2 };
    /// 4/3, the Pythagorean perfect fourth.
    pub const PYTH_4TH: RawJiRatio = RawJiRatio { numer: 4, denom: 3 };
    /// 3/2, the tritave.
    pub const TRITAVE: RawJiRatio = RawJiRatio { numer: 3, denom: 1 };
    /// 5/4, the pental major third.
    pub const PENTAL_MAJ3: RawJiRatio = RawJiRatio { numer: 5, denom: 4 };
    /// 7/4, the harmonic seventh.
    pub const SEPTIMAL_MIN7: RawJiRatio = RawJiRatio { numer: 7, denom: 4 };
    /// 11/8, the harmonic half-sharp fourth.
    pub const SEMIAUGMENTED_4TH: RawJiRatio = RawJiRatio {
        numer: 11,
        denom: 8,
    };

    /// Get the nth harmonic (ratio n/1).
    pub fn harm(n: u32) -> Result<Self, IllegalJiRatio> {
        if n == 0 {
            // Zero is not a valid harmonic
            Err(IllegalJiRatio { numer: 0, denom: 1 })
        } else {
            // gcd(n, 1) == 1, so ratio is already in lowest terms
            Ok(RawJiRatio { numer: n, denom: 1 })
        }
    }

    /// Reduce the ratio r modulo the logarithmic absolute value of m (equivalent to reduction modulo an octave).
    pub fn checked_rd(self, equave: Self) -> Result<Self, BadJiArith> {
        if equave.numer() == equave.denom() {
            // Cannot reduce modulo unison
            Err(BadJiArith::LogDivByUnison)
        } else {
            // Ensure equave is >= unison for consistent comparison
            let equave = if equave.numer() < equave.denom() {
                equave.reciprocal()
            } else {
                equave
            };
            let mut ret = self;
            if ret >= RawJiRatio::UNISON {
                // Ratios >= 1/1: keep dividing by equave until below it
                while ret >= equave {
                    ret /= equave;
                }
                Ok(ret)
            } else {
                // Ratios < 1/1: keep multiplying by equave until >= 1/1
                while ret < RawJiRatio::UNISON {
                    ret *= equave;
                }
                Ok(ret)
            }
        }
    }
    /// Logarithmic absolute value of a JI ratio (always >= 1/1, so intervals are always upward).
    /// Returns reciprocal if ratio < 1/1, otherwise returns the ratio itself.
    pub fn magnitude(self) -> Self {
        if self < Self::UNISON {
            // Ratio < 1/1: flip it to get magnitude
            self.reciprocal()
        } else {
            // Ratio >= 1/1: already in magnitude form
            self
        }
    }
}

impl Dyad for RawJiRatio {
    /// Stack two intervals (multiply ratios).
    fn stack(self, rhs: Self) -> Self {
        self * rhs
    }
    /// Unstack rhs from self (divide ratios).
    fn unstack(self, rhs: Self) -> Self {
        self / rhs
    }
    /// Logarithmic inverse (reciprocal).
    fn log_inv(self) -> Self {
        self.reciprocal()
    }
    /// Size of interval in cents (log base 2 * 1200).
    fn cents(self) -> f64 {
        ((self.numer as f64) / (self.denom as f64)).log2() * 1200.0
    }
    /// Natural logarithm of the ratio.
    fn ln(self) -> f64 {
        ((self.numer as f64) / (self.denom as f64)).ln()
    }
    /// Return the unison (identity element).
    fn unison() -> Self {
        Self::UNISON
    }
    /// Raise interval to integer power (stack n copies).
    fn pow(self, n: i32) -> Self {
        (0..n).fold(Self::UNISON, |acc, _| self * acc)
    }

    /// Reduce interval modulo an equave (e.g., octave reduction).
    fn rd(self, modulo: Self) -> Self {
        if modulo == RawJiRatio::UNISON {
            panic!("division by zero (log division by unison)")
        } else {
            // Ensure modulo >= 1/1 for consistent reduction
            let modulo = if modulo.numer() < modulo.denom() {
                modulo.reciprocal()
            } else {
                modulo
            };
            let mut ret = self;
            if ret >= Self::UNISON {
                // Intervals >= 1/1: keep dividing by modulo until below it
                while ret >= modulo {
                    ret /= modulo;
                }
                ret
            } else {
                // Intervals < 1/1: keep multiplying by modulo until >= 1/1
                while ret < Self::UNISON {
                    ret *= modulo;
                }
                ret
            }
        }
    }
}

impl fmt::Display for RawJiRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numer, self.denom)
    }
}

impl Mul for RawJiRatio {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        // Multiply: (a/b) * (c/d) = (a*c)/(b*d), then reduce to lowest terms
        let d = gcd(self.numer * other.numer, self.denom * other.denom);
        RawJiRatio {
            numer: self.numer * other.numer / d,
            denom: self.denom * other.denom / d,
        }
    }
}
impl MulAssign for RawJiRatio {
    fn mul_assign(&mut self, other: Self) {
        // In-place multiplication with reduction
        let d = gcd(self.numer * other.numer, self.denom * other.denom);
        self.numer *= other.numer;
        self.numer /= d;
        self.denom *= other.denom;
        self.denom /= d;
    }
}

impl CheckedMul for RawJiRatio {
    fn checked_mul(&self, other: &Self) -> Option<Self> {
        // Multiplication with overflow checking
        let d = gcd(self.numer * other.numer, self.denom * other.denom);
        Some(RawJiRatio {
            numer: self.numer.checked_mul(other.numer)? / d,
            denom: self.denom.checked_mul(other.denom)? / d,
        })
    }
}

impl Div for RawJiRatio {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        // Divide: (a/b) / (c/d) = (a*d)/(b*c), then reduce to lowest terms
        let d = gcd(self.numer * other.denom, self.denom * other.numer);
        RawJiRatio {
            numer: self.numer * other.denom / d,
            denom: self.denom * other.numer / d,
        }
    }
}

impl DivAssign for RawJiRatio {
    fn div_assign(&mut self, other: Self) {
        // In-place division with reduction
        let d = gcd(self.numer * other.denom, self.denom * other.numer);
        self.numer *= other.denom;
        self.numer /= d;
        self.denom *= other.numer;
        self.denom /= d;
    }
}

impl CheckedDiv for RawJiRatio {
    fn checked_div(&self, other: &Self) -> Option<Self> {
        let d = gcd(self.numer * other.denom, self.denom * other.numer);
        Some(RawJiRatio {
            numer: self.numer.checked_mul(other.denom)? / d,
            denom: self.denom.checked_mul(other.numer)? / d,
        })
    }
}
