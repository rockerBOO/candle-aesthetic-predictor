# candle_aesthetic_predictor

<!--toc:start-->

- [candle_aesthetic_predictor](#candleaestheticpredictor)
  - [Options](#options)
  - [Usage](#usage)
  - [Predictor simple](#predictor-simple)
  - [Conv predictor](#conv-predictor)
  - [Contributions](#contributions)
  <!--toc:end-->

Trainer and inference for aesthetic predictor. Currently a WIP and may not be working...

## Options

- train
- predictor

- train_conv
- predictor_conv

- server

## Usage

The above are the binaries which you can use with `cargo run --bin train -- --help`

## Predictor simple

Simple linear network using CLIP image embeddings. Uses ImprovedAestheticPredictor network as a base with improvements or adjustments.

## Conv predictor

We get the latents from the SD VAE and train the model on those. Currently waiting for `conv_transpose2d` to get backwards support.

## Contributions

Open for feedback and help getting some things working but very much a WIP currently. Limited support.
