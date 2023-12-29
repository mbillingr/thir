use crate::kinds::Kind;
use crate::qualified::Qual;
use crate::substitutions::{Subst, Types};
use crate::types::{Type, Tyvar};
use crate::List;

#[derive(Clone, Debug, PartialEq)]
pub enum Scheme {
    Forall(List<Kind>, Qual<Type>),
}

impl Types for Scheme {
    fn apply_subst(&self, s: &Subst) -> Self {
        match self {
            Scheme::Forall(ks, qt) => Scheme::Forall(ks.clone(), s.apply(qt)),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        match self {
            Scheme::Forall(_, qt) => qt.tv(),
        }
    }
}
