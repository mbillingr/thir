/*!
Types
!*/
use crate::kinds::{HasKind, Kind};
use crate::{Id, Int};
use std::rc::Rc;

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
