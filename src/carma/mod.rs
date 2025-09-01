use pyo3::prelude::*;

pub mod carma_model;
pub mod estimation;
pub mod simulation;
pub mod kalman;
pub mod analysis;
pub mod selection;
pub mod utils;

// Re-export main types and functions for easier access
pub use carma_model::*;
pub use estimation::*;
pub use simulation::*;
pub use kalman::*;
pub use analysis::*;
pub use selection::*;
pub use utils::*;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_structure() {
        // Basic test to ensure module compiles
        assert!(true);
    }
}