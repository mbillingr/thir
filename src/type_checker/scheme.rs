use crate::type_checker::kinds::{HasKind, Kind};
use crate::type_checker::qualified::Qual;
use crate::type_checker::substitutions::{Subst, Types};
use crate::type_checker::types::{Type, Tyvar};
use crate::type_checker::GenId;
use crate::utils::lists::List;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
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
        let n = vs_.len() as GenId;
        let s = Subst::from_rev_iter(
            vs_.into_iter()
                .rev()
                .zip((0..n).rev().map(|k| Type::TGen(k))),
        );
        Scheme::Forall(ks, s.apply(qt))
    }

    pub fn is_constant(&self) -> bool {
        match self {
            Scheme::Forall(_, Qual(_, t)) => t.as_fn().is_none(),
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Scheme::Forall(_, Qual(_, t)) => t.fn_types().0.len(),
        }
    }
}

impl Type {
    pub fn to_scheme(self) -> Scheme {
        Scheme::Forall(List::Nil, Qual(vec![], self))
    }
}
