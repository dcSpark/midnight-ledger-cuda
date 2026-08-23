#[test]
#[ignore = "requires a CUDA device and the sppark resident multi-open pipeline"]
fn resident_multiopen_ownership_error_isolation() {
    midnight_proofs::poly::kzg::assert_resident_multiopen_ownership_error_isolation();
}
