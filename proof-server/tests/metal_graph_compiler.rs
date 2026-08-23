#![cfg(feature = "metal")]

#[test]
fn metal_build_exercises_the_neutral_graph_family_compiler_contract() {
    midnight_proofs::plonk::resident_graph_compiler_regression()
        .expect("backend-neutral graph and graph-family compiler regression");
}
