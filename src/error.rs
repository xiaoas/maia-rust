//! Error types for the Maia chess evaluation library.
//!
//! This crate uses `thiserror` to provide a convenient enumeration of
//! errors that may occur while interacting with Maia3. The variants
//! wrap underlying errors from ONNX Runtime, chess parsing, and
//! tensor operations, giving the caller a single error type to handle.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    /// Wraps an error returned by the underlying ONNX Runtime bindings.
    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),

    /// The provided FEN string could not be parsed.
    #[error("Invalid FEN: {0}")]
    InvalidFen(#[from] shakmaty::fen::ParseFenError),

    /// A parsed position is invalid from the perspective of `shakmaty`.
    #[error("Invalid Chess Position: {0}")]
    InvalidPosition(Box<shakmaty::PositionError<shakmaty::Chess>>),

    /// Occurs when an ndarray has an unexpected shape during tensor
    /// preparation or extraction.
    #[error("Tensor shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),
}

impl From<shakmaty::PositionError<shakmaty::Chess>> for Error {
    fn from(err: shakmaty::PositionError<shakmaty::Chess>) -> Self {
        Error::InvalidPosition(Box::new(err))
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}
