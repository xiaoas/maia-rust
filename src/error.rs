use thiserror::Error;

#[derive(Error, Debug)]
pub enum MaiaError {
    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),
    #[error("Invalid FEN: {0}")]
    InvalidFen(#[from] shakmaty::fen::ParseFenError),
    #[error("Invalid Chess Position: {0}")]
    InvalidPosition(#[from] shakmaty::PositionError<shakmaty::Chess>),
    #[error("Tensor shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),
}
