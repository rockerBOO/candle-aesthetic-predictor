use candle_core::{Module, Tensor};
use crate::adaptive_avg_pool_2d;

#[derive(Debug)]
pub struct SqueezeExcitation {
    fc1: candle_nn::Conv2d,
    fc2: candle_nn::Conv2d,
}

impl SqueezeExcitation {
    pub fn new(in_channels: usize, squeeze_channels: usize, vb: candle_nn::VarBuilder) -> candle_core::Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            ..Default::default()
        };

        Ok(SqueezeExcitation {
            fc1: candle_nn::conv2d(in_channels, squeeze_channels, 1, cfg, vb.pp("fc1"))?,
            fc2: candle_nn::conv2d(squeeze_channels, in_channels, 1, cfg, vb.pp("fc2"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = adaptive_avg_pool_2d(xs)?;
        let xs = candle_nn::ops::silu(&self.fc1.forward(&xs)?)?;
        candle_nn::ops::sigmoid(&self.fc2.forward(&xs)?)
    }
}

