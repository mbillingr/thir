/*!
Types
!*/
use std::rc::Rc;
use crate::{Id, Int};
use crate::kinds::Kind;

/// The type of a value
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    /// A type variable
    TVar(Tyvar),

    /// A type constant
    TCon(Tycon),

    /// A type application (applying a type of kind `k1` to a type of
    /// kind `k1 -> k2` results in a type of kind `k2`.)
    TApp(Rc<(crate::Type, crate::Type)>),

    /// A generic (quantified) type variable
    TGen(Int),
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
}