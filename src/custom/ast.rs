use crate::custom::persistent::PersistentMap as Map;
use crate::thir_core::kinds::Kind;
use crate::thir_core::predicates::Pred;
use crate::thir_core::scheme::Scheme;
use crate::thir_core::types::Type;
use std::rc::Rc;

struct Id(Rc<String>);

struct Toplevel {
    interface_defs: Vec<Interface>,
    //interface_impls: Vec<Implementation>,
}

struct Interface {
    /// The name of the interface
    name: Id,
    /// Super-interfaces
    supers: Vec<Id>,
    /// Methods defined by the interface
    methods: Map<Id, SchemeDecl>,
}

struct SchemeDecl {
    // todo
}
