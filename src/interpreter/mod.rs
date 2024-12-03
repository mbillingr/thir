mod dispatch;
mod evaluator;
pub mod value;

use crate::utils::assoc_list::AssocList;
pub use evaluator::Context;
pub use value::Value;

pub type Env = AssocList<String, Value>;
