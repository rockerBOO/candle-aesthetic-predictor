// layers.0.linear.bias
// layers.0.linear.weight
// layers.1.linear.bias
// layers.1.linear.weight
// layers.2.linear.bias
// layers.2.linear.weight
// layers.3.linear.bias
// layers.3.linear.weight
// layers.4.linear.bias
// layers.4.linear.weight
//

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{ops, VarBuilder};
// use candle_nn::VarBuilder;

fn adaptive_avg_pool_2d(xs: &Tensor) -> Result<Tensor> {
    xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)
}

#[derive(Debug)]
pub struct AestheticPredictorSimpleDown {
    linear: candle_nn::Linear,
    dropout: f32,
}

impl AestheticPredictorSimpleDown {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        dropout: f32,
        vs: VarBuilder,
    ) -> Result<Self> {
        let linear = candle_nn::linear(in_channels, out_channels, vs)?;

        Ok(AestheticPredictorSimpleDown { linear, dropout })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        ops::dropout(&ops::silu(&self.linear.forward(xs)?)?, self.dropout)
    }
}

pub struct Config {
    pub embed_dim: usize,
    pub in_channels: usize,
}

#[derive(Debug)]
pub struct AestheticPredictorSimple {
    down0: AestheticPredictorSimpleDown,
    down1: AestheticPredictorSimpleDown,
    down2: AestheticPredictorSimpleDown,
    down3: AestheticPredictorSimpleDown,
    down4: AestheticPredictorSimpleDown,
    linear_5: candle_nn::Linear,
    linear_6: candle_nn::Linear, // linear_7: candle_nn::Linear,
                                 // linear_8: candle_nn::Linear,
}

impl AestheticPredictorSimple {
    pub fn new(vs: candle_nn::VarBuilder, c: &Config) -> Result<Self> {
        let down0 = AestheticPredictorSimpleDown::new(
            c.embed_dim,
            c.in_channels,
            0.2,
            vs.pp("layers.0.linear"),
        )?;
        let down1 = AestheticPredictorSimpleDown::new(
            c.in_channels,
            c.in_channels / 2,
            0.2,
            vs.pp("layers.1.linear"),
        )?;
        let down2 = AestheticPredictorSimpleDown::new(
            c.in_channels / 2,
            c.in_channels / 4,
            0.2,
            vs.pp("layers.2.linear"),
        )?;
        let down3 = AestheticPredictorSimpleDown::new(
            c.in_channels / 4,
            c.in_channels / 8,
            0.2,
            vs.pp("layers.3.linear"),
        )?;

        let down4 = AestheticPredictorSimpleDown::new(
            c.in_channels / 8,
            c.in_channels / 16,
            0.2,
            vs.pp("layers.4.linear"),
        )?;

        let linear_5 = candle_nn::linear(
            c.in_channels / 16,
            c.in_channels / 64,
            vs.pp("layers.5.linear"),
        )?;
        let linear_6 = candle_nn::linear(c.in_channels / 64, 1, vs.pp("layers.6.linear"))?;
        // let linear_7 = candle_nn::linear(
        //     c.in_channels / 64,
        //     c.in_channels / 128,
        //     vs.pp("layers.7.linear"),
        // )?;
        //
        // let linear_8 = candle_nn::linear(
        //     c.in_channels / 128,
        //     c.in_channels / 256,
        //     vs.pp("layers.8.linear"),
        // )?;

        Ok(AestheticPredictorSimple {
            down0,
            down1,
            down2,
            down3,
            down4,
            linear_5,
            linear_6,
            // linear_7,
            // linear_8,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.down0.forward(xs)?;
        let xs = self.down1.forward(&xs)?;
        let xs = self.down2.forward(&xs)?;
        let xs = self.down3.forward(&xs)?;
        let xs = self.down4.forward(&xs)?;
        let xs = self.linear_5.forward(&xs)?;
        self.linear_6.forward(&xs)
        // let xs = self.linear_7.forward(&xs)?;
        // self.linear_8.forward(&xs)

        // adaptive_avg_pool_2d(&xs)
        // ops::log_softmax(&xs, 1)
    }
}

impl Load for AestheticPredictorSimple {
    fn load_file<P: AsRef<std::path::Path>>(
        weights_path: P,
        config: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vs = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
        };

        AestheticPredictorSimple::new(vs, config)
    }
}

trait Load {
    fn load_file<P: AsRef<std::path::Path>>(
        weights_path: P,
        config: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<AestheticPredictorSimple>;
}
