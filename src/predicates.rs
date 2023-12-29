use crate::substitutions::{Subst, Types};
use crate::types::{Type, Tyvar};
use crate::unification::{matches, mgu};
use crate::Id;

/// A predicate imposes constraints on types
#[derive(Clone, Debug, PartialEq)]
pub enum Pred {
    /// Assert that the type (2nd field) is a member of class (1st field)  
    IsIn(Id, Type),
}

impl Types for Pred {
    fn apply_subst(&self, s: &Subst) -> Self {
        match self {
            Pred::IsIn(i, t) => Pred::IsIn(i.clone(), s.apply(t)),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        match self {
            Pred::IsIn(i, t) => t.tv(),
        }
    }
}

pub fn mgu_pred(a: &Pred, b: &Pred) -> crate::Result<Subst> {
    lift(mgu, a, b)
}

pub fn match_pred(a: &Pred, b: &Pred) -> crate::Result<Subst> {
    lift(matches, a, b)
}

fn lift(
    m: impl Fn(&Type, &Type) -> crate::Result<Subst>,
    a: &Pred,
    b: &Pred,
) -> crate::Result<Subst> {
    match (a, b) {
        (Pred::IsIn(i1, t1), Pred::IsIn(i2, t2)) if i1 == i2 => m(t1, t2),
        _ => Err("classes differ")?,
    }
}
