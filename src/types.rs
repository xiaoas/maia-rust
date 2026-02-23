use shakmaty::uci::UciMove;

#[derive(Debug, Clone)]
pub struct MoveProbability {
    pub uci: UciMove,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// List of legal moves and their probabilities, sorted highest to lowest
    pub policy: Vec<MoveProbability>,
    /// Win probability (0.0 to 1.0) for the side to move
    pub value: f32,
}
