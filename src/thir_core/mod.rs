#[macro_use]
pub mod lists;
mod ambiguity;
pub mod assumptions;
pub mod classes;
mod instantiate;
pub mod kinds;
pub mod predicates;
pub mod qualified;
pub mod scheme;
pub mod specific_inference;
pub mod specifics;
mod substitutions;
mod type_inference;
pub mod types;
mod unification;

pub type Int = usize;
pub type Id = String;
