use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    group_norm, ops, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder,
};
// use candle_nn::VarBuilder;

pub struct Config {}

fn adaptive_avg_pool_2d(xs: &Tensor) -> Result<Tensor> {
    xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)
}

#[derive(Debug)]
struct BlockConv {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    se: SqueezeExcitation,
}

impl BlockConv {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        squeeze_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            ..Default::default()
        };

        Ok(Self {
            conv1: candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv1"))?,
            conv2: candle_nn::conv2d(out_channels, out_channels, kernel_size, cfg, vb.pp("conv2"))?,
            se: SqueezeExcitation::new(out_channels, squeeze_channels, vb.pp("se"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = ops::silu(&self.conv1.forward(xs)?)?.max_pool2d(2)?;
        ops::silu(&self.conv2.forward(&xs)?)?.max_pool2d(2)
        // self.se.forward(&xs)
    }
}

#[derive(Debug)]
struct SqueezeExcitation {
    fc1: candle_nn::Conv2d,
    fc2: candle_nn::Conv2d,
}

impl SqueezeExcitation {
    pub fn new(in_channels: usize, squeeze_channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            ..Default::default()
        };

        Ok(SqueezeExcitation {
            fc1: candle_nn::conv2d(in_channels, squeeze_channels, 1, cfg, vb.pp("fc1"))?,
            fc2: candle_nn::conv2d(squeeze_channels, in_channels, 1, cfg, vb.pp("fc2"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = adaptive_avg_pool_2d(xs)?;
        let xs = candle_nn::ops::silu(&self.fc1.forward(&xs)?)?;
        candle_nn::ops::sigmoid(&self.fc2.forward(&xs)?)
    }
}

#[derive(Debug)]
pub struct DownConv {
    // pool1: usize,
    // norm1: candle_nn::GroupNorm,
    conv_block: BlockConv,
}

impl DownConv {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        squeeze_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(DownConv {
            conv_block: BlockConv::new(
                in_channels,
                out_channels,
                squeeze_channels,
                kernel_size,
                vb.pp("conv_block"),
            )?,
            // pool1: 2,
            // norm1: group_norm(8, in_channels, 1e-06, vb.pp("norm1"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        println!("{}", xs);
        panic!("yo");
        // let xs = self.norm1.forward(&xs)?;
        self.conv_block.forward(xs)
    }
}

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
            vb.pp("conv_block"),
        )?;

        Ok(Self {
            conv_trans1,
            conv_block,
        })
    }

    fn forward(&self, xs: &Tensor, xs_skip: &Tensor) -> Result<Tensor> {
        let xs = self.conv_trans1.forward(xs)?;
        let xs = Tensor::cat(&[&xs, xs_skip], 1)?;
        self.conv_block.forward(&xs)
    }
}

pub struct AestheticPredictorConv {
    init_conv: BlockConv,
    down1: DownConv,
    down2: DownConv,
    down3: DownConv,
    up3: UpConv,
    up2: UpConv,
    up1: UpConv,
    out: UpConv,
}

impl AestheticPredictorConv {
    pub fn new(vs: candle_nn::VarBuilder, c: &Config) -> Result<Self> {
        let init_conv = BlockConv::new(4, 64, 8, 3, vs.pp("init_conv"))?;
        let down1 = DownConv::new(64, 128, 16, 3, vs.pp("down1"))?;
        let down2 = DownConv::new(128, 256, 32, 3, vs.pp("down2"))?;
        let down3 = DownConv::new(256, 512, 64, 3, vs.pp("down3"))?;

        let up3 = UpConv::new(512, 256, 256, 32, 3, vs.pp("up3"))?;
        let up2 = UpConv::new(256, 128, 128, 16, 3, vs.pp("up2"))?;
        let up1 = UpConv::new(128, 64, 64, 8, 3, vs.pp("up1"))?;
        let out = UpConv::new(64, 16, 16, 8, 3, vs.pp("out"))?;

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
        let xs = self.up3.forward(&x3, &x0)?;
        let xs = self.up2.forward(&xs, &x1)?;
        let xs = self.up1.forward(&xs, &x2)?;
        let xs = self.out.forward(&xs, &x3)?;

        adaptive_avg_pool_2d(&xs)
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
