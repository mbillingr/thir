mod dispatch;
mod evaluator;
mod value;

pub use evaluator::Context;
use std::collections::HashMap;
pub use value::Value;

pub type Env = HashMap<String, Value>;
