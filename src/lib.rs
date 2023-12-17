mod aesthetic_predictor_simple;
use candle_core::{Result, Device};

pub fn parse_device(device: Option<String>) -> Result<Device> {
    match device.as_deref() {
        Some("cpu") => Ok(Device::Cpu),
        Some("cuda") => Device::cuda_if_available(0),
        Some("0") => Device::cuda_if_available(0),
        Some("1") => Device::cuda_if_available(1),
        Some("2") => Device::cuda_if_available(2),
        _ => Device::cuda_if_available(0),
    }
}

pub fn set_seed(seed: Option<u64>, device: &Device) -> Result<()> {
    match device {
        Device::Cpu => {
            println!("Device: CPU");
        }
        Device::Cuda(_) => {
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
