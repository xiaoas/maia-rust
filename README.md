# maia-rust

Rust bindings and utilities for the [Maia2](https://github.com/CSSLab/maia2) chess evaluation model.

The Maia2 network was trained to predict move probabilities and win
values for positions conditioned on two ELO ratings.  This crate wraps
an ONNX Runtime session and provides a convenient API for inference.

## Features

- Load a Maia2 model from disk or memory
- Evaluate single positions or batches
- Automatically convert FEN strings / `shakmaty` setups into the
  tensor format expected by the network
- Bucketing of ELO ratings to match the training setup
- Results include move probabilities sorted in descending order and a
  win probability for the side to move

## Usage

```rust
use maia_rust::Maia;

fn main() -> Result<(), maia_rust::MaiaError> {
    // load the model (example path -- replace with your own)
    let mut maia = Maia::from_file("maia_rapid.onnx")?;

    // evaluate a position by FEN
    let eval = maia.evaluate(
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        1200, // self elo
        1500, // opponent elo
    )?;

    println!("policy: {:?}\nvalue: {}", eval.policy, eval.value);

    Ok(())
}
```

Batched inference is supported via [`Maia::batch_evaluate`], which
accepts an iterator of `shakmaty::Setup` and parallel slices of ELO
ratings.

```rust
use shakmaty::fen::Fen;
let setups = vec![
    Fen::from_str("..."?)?.into_setup(),
    // more setups...
];
let results = maia.batch_evaluate(setups, &[1200, 1800], &[1600, 1400])?;
```

## License

Original code released under the MIT/Apache-2.0 license. See
`LICENSE` for details.
