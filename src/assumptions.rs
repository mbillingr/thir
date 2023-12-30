use crate::scheme::Scheme;
use crate::substitutions::{Subst, Types};
use crate::types::Tyvar;
use crate::Id;
use std::fmt::{Debug, Formatter};

/// Represent assumptions about the type of a variable by pairing
/// a variable name with a type scheme.
#[derive(Clone)]
pub struct Assump {
    pub i: Id,
    pub sc: Scheme,
}

impl Debug for Assump {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} :>: {:?}", self.i, self.sc)
    }
}

impl Types for Assump {
    fn apply_subst(&self, s: &Subst) -> Self {
        Assump {
            i: self.i.clone(),
            sc: s.apply(&self.sc),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        self.sc.tv()
    }
}

pub fn find<'a>(i: &Id, ass: impl IntoIterator<Item = &'a Assump>) -> crate::Result<&'a Scheme> {
    for a in ass {
        if &a.i == i {
            return Ok(&a.sc);
        }
    }
    Err(format!("unbound identifier: {i}"))
}
