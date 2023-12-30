use crate::substitutions::{Subst, Types};
use crate::types::{Type, Tyvar};
use crate::unification::{matches, mgu};
use crate::Id;

/// A predicate imposes constraints on types
#[derive(Clone, Debug, PartialEq)]
pub enum Pred {
    /// Assert that the combination of types (2nd field) is a member of class (1st field)  
    IsIn(Id, Vec<Type>),
}

impl Types for Pred {
    fn apply_subst(&self, s: &Subst) -> Self {
        match self {
            Pred::IsIn(i, t) => Pred::IsIn(i.clone(), s.apply(t)),
        }
    }

    fn tv(&self) -> Vec<Tyvar> {
        match self {
            Pred::IsIn(_, t) => t.tv(),
        }
    }
}

impl Pred {
    /// test if a predicate is in head-normal form
    pub fn in_hnf(&self) -> bool {
        fn hnf(t: &Type) -> bool {
            match t {
                Type::TVar(_) => true,
                Type::TCon(_) => false,
                Type::TApp(app) => hnf(&app.0),
                Type::TGen(_) => panic!("don't know what to do!"),
            }
        }

        match self {
            Pred::IsIn(_, ts) => ts.iter().all(hnf),
        }
    }
}

// todo: I'm not sure if the merging/composition of the match/mgu is correct.

pub fn mgu_pred(a: &Pred, b: &Pred) -> crate::Result<Subst> {
    lift(mgu, |s1, s2| Ok(s2.compose(s1)), a, b)
}

pub fn match_pred(a: &Pred, b: &Pred) -> crate::Result<Subst> {
    lift(matches, Subst::merge, a, b)
}

fn lift(
    m: impl Fn(&Type, &Type) -> crate::Result<Subst>,
    c: impl Fn(&Subst, &Subst) -> crate::Result<Subst>,
    a: &Pred,
    b: &Pred,
) -> crate::Result<Subst> {
    match (a, b) {
        (Pred::IsIn(i1, ts1), Pred::IsIn(i2, ts2)) if i1 == i2 => {
            let mut s = Subst::null_subst();
            for (t1, t2) in ts1.iter().zip(ts2) {
                s = c(&s, &m(t1, t2)?)?;
            }
            Ok(s)
        }
        _ => Err("classes differ")?,
    }
}
