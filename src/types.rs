/*!
Types
!*/
use crate::{Id, Int};
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
pub struct Tyvar(pub Id);

/// A type constant
#[derive(Clone, Debug, PartialEq)]
pub struct Tycon(pub Id);

impl Type {
    /// construct a type application (convenience method)
    pub fn tapp(a: Type, b: Type) -> Type {
        Type::TApp(Rc::new((a, b)))
    }
}
