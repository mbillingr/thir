/*!
Defines the possible kinds of types.
Kinds play the same role for types, as types do for values.
!*/

use std::rc::Rc;

/// The kind of a type
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// The kind of simple (nullary) types such as `Int` or `Int -> Bool`.
    Star,

    /// The kind of type constructors such as `List t`.
    Kfun(Rc<(Kind, Kind)>),
}


impl Kind {
    pub fn kfun(a: Kind, b: Kind) -> Kind {
        Kind::Kfun(Rc::new((a, b)))
    }
}
