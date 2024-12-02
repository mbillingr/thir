use crate::interpreter::Value;
use crate::type_checker::types::{Type};


pub fn type_matches_type(ty: &Type, t: &Type) -> bool {
    ty == t
}


pub fn type_matches(t: &Type, v: &Value) -> bool {
    v.is_a(t)
}
