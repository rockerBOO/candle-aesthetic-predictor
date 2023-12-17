mod train;

use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let xs = Tensor::arange::<f32>(0., 128. * 5. * 5., &Device::Cpu)?.reshape((1, 128, 5, 5))?;

    let xs = xs.max_pool2d(2)?;

    println!("{}", xs);

    let xs = Tensor::arange::<f32>(0., 128. * 5. * 5., &Device::cuda_if_available(0)?)?
        .reshape((1, 128, 5, 5))?;

    let xs = xs.max_pool2d(2)?;
    println!("{}", xs);

    Ok(())
}
