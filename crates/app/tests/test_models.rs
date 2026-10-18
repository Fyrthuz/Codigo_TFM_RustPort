use app::models::unet::Unet;
use app::models::universeg::UniverSeg;
use tch::{nn, CModule, Device, Kind, Tensor};

fn repo_root() -> String {
    format!("{}/../..", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn test_unet_matches_torchscript() {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let model = Unet::new(&vs.root(), 3, 1, 32);
    vs.load(format!("{}/unet_model.safetensors", repo_root())).unwrap();
    let cm = CModule::load(format!("{}/unet_model.pt", repo_root())).unwrap();

    let x = Tensor::randn(&[1, 3, 256, 256], (Kind::Float, Device::Cpu));
    let mine = model.forward(&x, false, 0.0);
    let ts = cm.forward_ts(&[x]).unwrap();
    let diff = (&mine - &ts).abs().max();
    let d = f64::try_from(&diff).unwrap();
    assert!(d < 1e-5, "max diff {}", d);
}

#[test]
fn test_unet_multiclass_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = Unet::new(&vs.root(), 1, 3, 16);
    let x = Tensor::randn(&[1, 1, 128, 128], (Kind::Float, Device::Cpu));
    let out = model.forward(&x, false, 0.0);
    assert_eq!(out.size(), vec![1, 3, 128, 128]);
}

#[test]
fn test_unet_small_input() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = Unet::new(&vs.root(), 1, 2, 8);
    let x = Tensor::randn(&[1, 1, 32, 32], (Kind::Float, Device::Cpu));
    let out = model.forward(&x, false, 0.0);
    assert_eq!(out.size(), vec![1, 2, 32, 32]);
}

#[test]
fn test_universeg_matches_torchscript() {
    let mut vs = nn::VarStore::new(Device::Cpu);
    let model = UniverSeg::new(&vs.root());
    vs.load(format!("{}/universeg.safetensors", repo_root())).unwrap();
    let cm = CModule::load(format!("{}/universeg.pt", repo_root())).unwrap();

    let img = Tensor::randn(&[1, 1, 128, 128], (Kind::Float, Device::Cpu));
    let sup_img = Tensor::randn(&[1, 4, 1, 128, 128], (Kind::Float, Device::Cpu));
    let sup_lbl = Tensor::rand(&[1, 4, 1, 128, 128], (Kind::Float, Device::Cpu))
        .gt(0.5)
        .to_kind(Kind::Float);

    let mine = model.forward(&img, &sup_img, &sup_lbl, 0.0);
    let ts = cm.forward_ts(&[img, sup_img, sup_lbl]).unwrap();
    let diff = (&mine - &ts).abs().max();
    let d = f64::try_from(&diff).unwrap();
    assert!(d < 1e-5, "max diff {}", d);
}
