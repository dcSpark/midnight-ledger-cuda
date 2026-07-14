fn main() {
    println!("cargo:rerun-if-changed=src/bls12_381_g1_msm.cu");
    println!("cargo:rerun-if-changed=../vendor/cgbn/include");
    println!("cargo:rerun-if-changed=../vendor/sppark-0.1.15/sppark/msm/pippenger.cuh");
    println!("cargo:rerun-if-env-changed=EC_GPU_CUDA_NVCC_ARGS");

    if std::env::var_os("EC_GPU_CUDA_NVCC_ARGS").is_none() {
        // The sppark build is already native-CUDA targeted. Keep the alternate
        // MSM fatbin equally focused while allowing packagers to override it.
        std::env::set_var(
            "EC_GPU_CUDA_NVCC_ARGS",
            "--fatbin --gpu-architecture=sm_89 --generate-code=arch=compute_89,code=sm_89",
        );
    }
    let source_builder =
        ec_gpu_gen::SourceBuilder::new().add_multiexp::<blstrs::G1Affine, blstrs::Fp>();
    ec_gpu_gen::generate(&source_builder);

    let mut build = sppark::build::ccmd();
    build.flag("-std=c++17");
    // Reserve independent accumulate cursors for the compatibility context and
    // eight proof-owned execution lanes. Each lane may use up to sixteen MSM
    // engines, with two cursors per engine.
    build.define("MSM_NSTREAMS", Some("288"));
    build.include("../vendor/cgbn/include");
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("adx") {
        build.define("__ADX__", None);
    }
    build.file("src/bls12_381_g1_msm.cu");
    build.compile("midnight_sppark_msm");
}
