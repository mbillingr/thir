use crate::thir_core::kinds::{HasKind, Kind};
use crate::thir_core::lists::List;
use crate::thir_core::qualified::Qual;
use crate::thir_core::substitutions::{Subst, Types};
use crate::thir_core::types::{Type, Tyvar};
use crate::thir_core::Int;

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

impl Scheme {
    pub fn quantify(vs: &[Tyvar], qt: &Qual<Type>) -> Self {
        let vs_: Vec<_> = qt.tv().into_iter().filter(|v| vs.contains(v)).collect();
        let ks = vs_.iter().map(|v| v.kind().unwrap().clone()).collect();
        let n = vs_.len() as Int;
        let s = Subst::from_rev_iter(
            vs_.into_iter()
                .rev()
                .zip((0..n).rev().map(|k| Type::TGen(k))),
        );
        Scheme::Forall(ks, s.apply(qt))
    }
}

impl Type {
    pub fn to_scheme(self) -> Scheme {
        Scheme::Forall(List::Nil, Qual(vec![], self))
    }
}
