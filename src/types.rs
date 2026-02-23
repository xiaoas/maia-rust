use shakmaty::uci::UciMove;

/// A move paired with the model's estimated probability of being the
/// best choice.
#[derive(Debug, Clone)]
pub struct MoveProbability {
    /// Move in UCI notation.
    pub uci: UciMove,
    /// Probability (0.0–1.0) assigned by the policy head after
    /// softmax normalization.
    pub probability: f32,
}

/// Output returned by the Maia evaluator.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Policy head results: legal moves sorted by descending
    /// probability.
    pub policy: Vec<MoveProbability>,
    /// Win probability for the side to move, normalized to the range
    /// [0, 1].
    pub value: f32,
}
