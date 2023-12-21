pub mod latent;
pub mod model;
pub mod squeeze_excitation;
pub mod error;

pub use error::{AestheticPredictorError, Result};

pub fn parse_device(device: Option<String>) -> candle_core::Result<candle_core::Device> {
    match device.as_deref() {
        Some("cpu") => Ok(candle_core::Device::Cpu),
        Some("cuda") => candle_core::Device::cuda_if_available(0),
        Some("0") => candle_core::Device::cuda_if_available(0),
        Some("1") => candle_core::Device::cuda_if_available(1),
        Some("2") => candle_core::Device::cuda_if_available(2),
        _ => candle_core::Device::cuda_if_available(0),
    }
}

pub fn set_seed(seed: Option<u64>, device: &candle_core::Device) -> candle_core::Result<()> {
    match device {
        candle_core::Device::Cpu => {
            println!("Device: CPU");
        }
        candle_core::Device::Cuda(_) => {
            println!("Seed: {}", seed.unwrap_or(1234));
            device.set_seed(seed.unwrap_or(1234))?;
            println!("Device: CUDA");
        }
        _ => {
            println!("Seed: {}", seed.unwrap_or(1234));
            device.set_seed(seed.unwrap_or(1234))?;
        }
    };

    Ok(())
}

pub fn adaptive_avg_pool_2d(xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    xs.mean_keepdim(candle_core::D::Minus2)?
        .mean_keepdim(candle_core::D::Minus1)
}
