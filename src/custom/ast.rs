use crate::custom::persistent::PersistentMap as Map;
use crate::thir_core::kinds::Kind;
use crate::thir_core::lists::List;
use crate::thir_core::predicates::Pred;
use crate::thir_core::scheme::Scheme;
use crate::thir_core::types::{Type, Tyvar};
use crate::thir_core::Id;
use serde::Deserialize;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

#[derive(Debug, Deserialize, PartialEq)]
pub struct Toplevel {
    pub interface_defs: Vec<Interface>,
    pub interface_impls: Vec<Impl>,
}

/** An Interface in Idris style looks like this:
```Idris
interface Show a where
    show : a -> String

We assume implicitly that `a` corresponds to the generic
type variable 0.
```
**/
#[derive(Debug, Deserialize, PartialEq)]
pub struct Interface {
    /// The name of the interface
    pub name: Id,
    /// Super-interfaces
    pub supers: List<Id>,
    /// Methods defined by the interface
    pub methods: Map<Id, Scheme>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Impl {
    /// The name of the interface
    pub name: Id,
    /// The type implementing the interface
    #[serde(rename = "for")]
    pub ty: Type,
    /// Predicates
    pub preds: Vec<Pred>,
    /// Method implementations
    pub methods: Map<Id, Vec<Alt>>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Alt(pub Vec<Pat>, pub Expr);

#[derive(Debug, Deserialize, PartialEq)]
pub enum Pat {
    #[serde(alias = "pvar")]
    PVar(Id),
}

#[derive(Debug, Deserialize, PartialEq)]
pub enum Expr {
    #[serde(alias = "var")]
    Var(Id),
}
