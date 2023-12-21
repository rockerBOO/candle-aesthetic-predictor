use crate::adaptive_avg_pool_2d;
use crate::squeeze_excitation::SqueezeExcitation;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{ops, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder};

pub struct Config {
    pub in_channels: usize,
    pub out_channels: usize,
}

#[derive(Debug)]
struct BlockConv {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    // se: SqueezeExcitation,
}

impl BlockConv {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        squeeze_channels: usize,
        kernel_size: usize,
        cfg: Conv2dConfig,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            conv1: candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv1"))?,
            conv2: candle_nn::conv2d(out_channels, out_channels, kernel_size, cfg, vb.pp("conv2"))?,
            // se: SqueezeExcitation::new(out_channels, squeeze_channels, vb.pp("se"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        println!("{:#?}", xs);
        let xs = ops::silu(&self.conv1.forward(xs)?)?;
        ops::silu(&self.conv2.forward(&xs)?)
        // self.se.forward(&xs)
    }
}

#[derive(Debug)]
pub struct DownConv {
    conv_block: BlockConv,
}

impl DownConv {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        squeeze_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        Ok(DownConv {
            conv_block: BlockConv::new(
                in_channels,
                out_channels,
                squeeze_channels,
                kernel_size,
                cfg,
                vb.pp("conv_block"),
            )?,
            // pool1: 2,
            // norm1: group_norm(8, in_channels, 1e-06, vb.pp("norm1"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.conv_block.forward(&xs.max_pool2d(2)?)
    }
}

#[derive(Debug)]
pub struct UpConv {
    conv_trans1: ConvTranspose2d,
    conv_block: BlockConv,
}

impl UpConv {
    fn new(
        in_channels: usize,
        skip_channels: usize,
        out_channels: usize,
        squeeze_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let transpose_cfg = ConvTranspose2dConfig {
            padding: 0,
            stride: 2,
            ..Default::default()
        };
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };

        let conv_trans1 = candle_nn::conv_transpose2d(
            in_channels,
            in_channels,
            2,
            transpose_cfg,
            vb.pp("conv_trans1"),
        )?;

        let conv_block = BlockConv::new(
            in_channels + skip_channels,
            out_channels,
            squeeze_channels,
            kernel_size,
            cfg,
            vb.pp("conv_block"),
        )?;

        Ok(Self {
            conv_trans1,
            conv_block,
        })
    }

    fn forward(&self, xs: &Tensor, xs_skip: &Tensor) -> Result<Tensor> {
        let xs = self.conv_trans1.forward(xs)?;
        println!("{:#?} {:#?}", xs, xs_skip);
        let xs = Tensor::cat(&[&xs, xs_skip], 1)?;
        self.conv_block.forward(&xs)
    }
}

#[derive(Debug)]
pub struct AestheticPredictorConv {
    init_conv: BlockConv,
    down1: DownConv,
    down2: DownConv,
    down3: DownConv,
    up3: UpConv,
    up2: UpConv,
    up1: UpConv,
    out: candle_nn::Conv2d,
}

impl AestheticPredictorConv {
    pub fn new(vs: candle_nn::VarBuilder, c: &Config) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let init_conv =
            BlockConv::new(c.in_channels, c.out_channels, 8, 3, cfg, vs.pp("init_conv"))?;
        let down1 = DownConv::new(c.out_channels, c.out_channels * 2, 16, 3, vs.pp("down1"))?;
        let down2 = DownConv::new(
            c.out_channels * 2,
            c.out_channels * 4,
            32,
            3,
            vs.pp("down2"),
        )?;
        let down3 = DownConv::new(
            c.out_channels * 4,
            c.out_channels * 8,
            64,
            3,
            vs.pp("down3"),
        )?;

        let up3 = UpConv::new(
            c.out_channels * 8,
            c.out_channels * 4,
            c.out_channels * 4,
            32,
            3,
            vs.pp("up3"),
        )?;
        let up2 = UpConv::new(
            c.out_channels * 4,
            c.out_channels * 2,
            c.out_channels * 2,
            16,
            3,
            vs.pp("up2"),
        )?;
        let up1 = UpConv::new(
            c.out_channels * 2,
            c.out_channels,
            c.out_channels,
            8,
            3,
            vs.pp("up1"),
        )?;

        let cfg = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let out = candle_nn::conv2d(c.out_channels, 1, 3, cfg, vs.pp("out"))?;

        Ok(AestheticPredictorConv {
            init_conv,
            down1,
            down2,
            down3,
            up3,
            up2,
            up1,
            out,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x0 = self.init_conv.forward(xs)?;
        let x1 = self.down1.forward(&x0)?;
        let x2 = self.down2.forward(&x1)?;
        let x3 = self.down3.forward(&x2)?;
        let x_up = self.up3.forward(&x3, &x2)?;
        let x_up = self.up2.forward(&x_up, &x1)?;
        let x_up = self.up1.forward(&x_up, &x0)?;
        let x_out = self.out.forward(&x_up)?;

        adaptive_avg_pool_2d(&x_out)
    }
}

impl Load for AestheticPredictorConv {
    fn load_file<P: AsRef<std::path::Path>>(
        weights_path: P,
        config: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let vs = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
        };

        AestheticPredictorConv::new(vs, config)
    }
}

trait Load {
    fn load_file<P: AsRef<std::path::Path>>(
        weights_path: P,
        config: &Config,
        device: &Device,
        dtype: DType,
    ) -> Result<AestheticPredictorConv>;
}
