/*!
Types
!*/

use crate::kinds::{HasKind, Kind};
use crate::{Id, Int};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

/// The type of a value
#[derive(Clone, PartialEq)]
pub enum Type {
    /// A type variable
    TVar(Tyvar),

    /// A type constant
    TCon(Tycon),

    /// A type application (applying a type of kind `k1` to a type of
    /// kind `k1 -> k2` results in a type of kind `k2`.)
    TApp(Rc<(Self, Self)>),

    /// A generic (quantified) type variable
    TGen(Int),
}

impl Debug for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::TVar(tv) => write!(f, "{}", tv.0),
            Type::TCon(tc) => write!(f, "{}", tc.0),
            Type::TApp(rc) => write!(f, "({:?} {:?})", rc.0, rc.1),
            Type::TGen(k) => write!(f, "'{k}"),
        }
    }
}

/// A type variable
#[derive(Clone, Debug, PartialEq)]
pub struct Tyvar(pub Id, pub Kind);

/// A type constant
#[derive(Clone, Debug, PartialEq)]
pub struct Tycon(pub Id, pub Kind);

impl Type {
    /// construct a type application (convenience method)
    pub fn tapp(a: Type, b: Type) -> Type {
        Type::TApp(Rc::new((a, b)))
    }

    pub fn subst(&self, subs: &HashMap<Id, (Type, Kind)>) -> Self {
        match self {
            Type::TVar(tv) => match subs.get(&tv.0) {
                Some((t, k)) => {
                    assert_eq!(k, &tv.1, "Cannot substitute variable with different kind");
                    t.clone()
                }
                None => self.clone(),
            },
            Type::TCon(_) => self.clone(),
            Type::TApp(app) => Type::TApp(Rc::new((app.0.subst(subs), app.1.subst(subs)))),
            Type::TGen(_) => self.clone(),
        }
    }
}

impl HasKind for Tyvar {
    fn kind(&self) -> crate::Result<&Kind> {
        Ok(&self.1)
    }
}

impl HasKind for Tycon {
    fn kind(&self) -> crate::Result<&Kind> {
        Ok(&self.1)
    }
}

impl HasKind for Type {
    fn kind(&self) -> crate::Result<&Kind> {
        match self {
            Type::TCon(tc) => tc.kind(),
            Type::TVar(u) => u.kind(),
            Type::TApp(app) => match app.0.kind()? {
                Kind::Kfun(kf) => Ok(&kf.1),
                _ => Err("Invalid Kind in TApp")?,
            },
            Type::TGen(_) => panic!("Don't know what to do :(   (maybe ignore somehow?)"),
        }
    }
}
