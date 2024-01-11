use serde::Deserialize;
use std::fmt::{Debug, Display, Formatter};
use std::rc::Rc;
use std::str::FromStr;

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

#[derive(Clone, Deserialize, Eq, Hash, PartialEq)]
pub struct Id(Rc<String>);

impl Id {
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<&str> for Id {
    fn from(value: &str) -> Self {
        Id(Rc::new(value.to_string()))
    }
}

impl From<String> for Id {
    fn from(value: String) -> Self {
        Id(Rc::new(value))
    }
}

impl Display for Id {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Debug for Id {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}
