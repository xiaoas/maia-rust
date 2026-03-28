# maia-rust

Rust bindings and utilities for the Maia3 chess evaluation model.

This crate wraps an ONNX Runtime session and provides a convenient API
for inference on Maia3 policy/value heads.

## Features

- Load a Maia3 ONNX model from disk or memory.
- Evaluate single positions or batches.
- Convert FEN strings / `shakmaty::Setup` values into Maia3 token input.
- Use raw floating-point Elo conditioning (`elo_self`, `elo_oppo`).
- Return legal move probabilities plus White/draw/Black outcome probabilities.

## Usage

```rust
use maia_rust::Maia;

fn main() -> Result<(), maia_rust::Error> {
    // Load the model (example path -- replace with your own)
    let mut maia = Maia::from_file("maia3_simplified.onnx")?;

    let eval = maia.evaluate_fen(
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        1500.0, // self elo
        1500.0, // opponent elo
    )?;

    println!(
        "policy: {:?}\nwhite_wr: {}\ndraw: {}\nblack_wr: {}",
        eval.policy, eval.white_wr, eval.draw, eval.black_wr
    );
    Ok(())
}
```

Batched inference is supported via `Maia::batch_evaluate`, plus
`batch_evaluate_async` and `batch_evaluate_with_options`.

## License

Original code released under the MIT/Apache-2.0 license. See `LICENSE`
for details.
