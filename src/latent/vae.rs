use crate::latent::image::image_preprocess;
use crate::Result;
use candle_transformers::models::stable_diffusion::vae::{self, DiagonalGaussianDistribution};

pub struct Vae<'a> {
    model: vae::AutoEncoderKL,
    dtype: candle_core::DType,
    device: &'a candle_core::Device,
}

impl Vae<'_> {
    pub fn new_from_file<'a>(
        model_path: &str,
        dtype: candle_core::DType,
        device: &'a candle_core::Device,
    ) -> Result<Vae<'a>> {
        let vae_cfg = vae::AutoEncoderKLConfig {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            latent_channels: 4,
            norm_num_groups: 32,
        };

        let vae_vs = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], dtype, device)?
        };

        Ok(Vae {
            model: vae::AutoEncoderKL::new(vae_vs, 3, 3, vae_cfg)?,
            dtype,
            device,
        })
    }

    pub fn encode(
        &self,
        images: &[candle_core::Tensor],
    ) -> Result<DiagonalGaussianDistribution> {
        Ok(self
            .model
            .encode(&candle_core::Tensor::stack(images, 1)?.squeeze(0)?)?)
    }

    pub fn encode_image(&self, image_path: &str) -> Result<DiagonalGaussianDistribution> {
        let image_tensor = image_preprocess(image_path)?
            .to_device(self.device)?
            .to_dtype(self.dtype)?;

        Ok(self.model.encode(&image_tensor)?)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.device
    }

    pub fn dtype(&self) -> &candle_core::DType {
        &self.dtype
    }
}
