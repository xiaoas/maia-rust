//! High‑level Rust interface to the Maia2 chess evaluation model.
//!
//!
//! This crate wraps an ONNX Runtime session and provides convenient
//! helpers for converting FEN strings or `shakmaty` setups into the
//! tensor format expected by the neural network. It also includes
//! utilities for mapping elo ratings to the categorical buckets used by
//! the model.
//!
//! The principal type is [`Maia`], which exposes single‑position and
//! batched evaluation methods. Results include a policy (legal moves
//! with associated probabilities) as well as a value (win probability
//! for the side to move).
//!
//! The library re‑exports `shakmaty` to make position construction easy.

mod error;
mod maia;
mod moves;
mod tensor;
mod types;

/// Main model wrapper.
pub use maia::Maia;

/// Error type produced by library operations.
pub use error::MaiaError;

/// Output data structures returned by evaluations.
pub use types::{EvaluationResult, MoveProbability};

/// Re-export of `shakmaty` for convenience when building positions.
pub use shakmaty;