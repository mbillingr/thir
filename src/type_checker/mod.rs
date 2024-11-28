pub mod ambiguity;
pub mod assumptions;
pub mod classes;
pub mod instantiate;
pub mod kinds;
pub mod predicates;
pub mod qualified;
pub mod scheme;
pub mod substitutions;
pub mod type_inference;
pub mod types;
pub mod unification;

/// Identifier for variables, constructors, etc.
pub type Id = String;

/// Uniquely identifies generic variables
pub type GenId = usize;
