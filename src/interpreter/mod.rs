mod dispatch;
mod evaluator;
mod value;

pub use evaluator::Context;
use std::collections::HashMap;
pub use value::Value;
use crate::utils::assoc_list::AssocList;

pub type Env = AssocList<String, Value>;
