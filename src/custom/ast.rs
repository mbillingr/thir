use crate::custom::persistent::PersistentMap as Map;
use crate::thir_core::kinds::Kind;
use crate::thir_core::predicates::Pred;
use crate::thir_core::types::Type;
use std::rc::Rc;

struct Id(Rc<String>);

struct Toplevel {
    interface_defs: Vec<Interface>,
    interface_impls: Vec<Implementation>,
}

struct Interface {
    /// The name of the interface
    name: Id,
    /// The type variables that parametrize the interface
    tvars: Vec<Id>,
    /// Type constraints form a hierarchy of interfaces. Must be acyclic!
    predicates: Vec<Predicate>,
    /// List of methods defined by the interface
    methods: Map<Id, Scheme>,
}

/// An interface implementation
struct Implementation {
    /// A qualified predicate specifies how the interface is implemented.
    /// Either unconditionally (e.g. implement `Add` for `Int` and `Int`),
    /// or conditionally (e.g. when `T` implements `Foo`, `List` `T` implements foo).
    decl: Qual<Predicate>,
    method_impls: Map<Id, ()>, // todo
}

/// Predicates impose constraints on types
enum Predicate {
    /// Declare that a group of types implement an interface
    Implements(Id, Vec<Type>),
}

/// Something "qualified" is restricted by a list of predicates.
pub struct Qual<T>(pub Vec<Predicate>, pub T);

enum Scheme {
    Forall(Map<Id, Kind>, Qual<Type>),
}
