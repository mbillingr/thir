use crate::kinds::{HasKind, Kind};
use crate::lists::List;
use crate::qualified::Qual;
use crate::substitutions::{Subst, Types};
use crate::types::{Type, Tyvar};
use crate::Int;

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
    /// The order of the kinds in the scheme is determined by the order in which the variables
    /// appear in the qualified type, and not by the order in which they appear in vs. This is
    /// the behavior specified in THIH.
    pub fn quantify(vs: &[Tyvar], qt: &Qual<Type>) -> Self {
        let vs_: Vec<_> = qt.tv().into_iter().filter(|v| vs.contains(v)).collect();
        Self::quant(vs_, qt)
    }

    /// Like `quantify`, but the order of the variables in the scheme is determined by the order
    /// in which they appear in vs. This seems to be the correct approach when constructing
    /// declared type schemes.
    pub fn quantify_by_var_order(vs: &[Tyvar], qt: &Qual<Type>) -> Self {
        let qttv = qt.tv();
        let vs_: Vec<_> = vs.iter().cloned().filter(|v| qttv.contains(v)).collect();
        Self::quant(vs_, qt)
    }

    fn quant(vs_: Vec<Tyvar>, qt: &Qual<Type>) -> Scheme {
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
