use crate::frontend::type_inference::Alt;
use crate::interpreter::evaluator::Context;
use crate::interpreter::{dispatch, Env};
use crate::type_checker::scheme::Scheme;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Value {
    Uninitialized,
    Boxed(Rc<RefCell<Self>>),

    Unit,
    Bool(bool),
    I64(i64),
    Char(char),
    F64(f64),
    String(Rc<str>),

    Closure(Closure),

    Constructor(Rc<str>, Rc<str>, Vec<Value>),

    Method(Rc<RefCell<HashMap<Scheme, Value>>>, Vec<Value>),

    Primitive(Primitive, Vec<Value>),
}

impl Value {
    pub fn is_a(&self, tyname: &str) -> bool {
        match self {
            Value::Boxed(bx) => bx.borrow().is_a(tyname),
            Value::I64(_) => tyname == "Int",
            Value::Char(_) => tyname == "Char",
            Value::F64(_) => tyname == "Double",
            Value::String(_) => tyname == "String",
            Value::Constructor(ty, _, _) => &**ty == tyname,
            _ => false,
        }
    }

    pub fn boxed(self) -> Self {
        Self::Boxed(Rc::new(self.into()))
    }

    pub fn update(&self, value: Value) {
        match self {
            Value::Boxed(bx) => *bx.borrow_mut() = value,
            _ => panic!("immutable value"),
        }
    }

    pub fn resolve(&self) -> Self {
        match self {
            Value::Boxed(bx) => bx.borrow().resolve(),
            _ => self.clone(),
        }
    }

    pub fn constructor(ty: impl Into<Rc<str>>, tag: impl Into<Rc<str>>) -> Self {
        Value::Constructor(ty.into(), tag.into(), vec![])
    }

    pub fn as_constructor(&self) -> Option<(Rc<str>, Vec<Value>)> {
        match self {
            Value::Boxed(bx) => bx.borrow().as_constructor(),
            Value::Constructor(_, tag, fields) => Some((tag.clone(), fields.clone())),
            _ => None,
        }
    }

    pub fn method() -> Self {
        Value::Method(Rc::new(RefCell::new(HashMap::new())), vec![])
    }

    pub fn add_impl(&self, sc: Scheme, value: Value) {
        match self {
            Value::Boxed(bx) => bx.borrow().add_impl(sc, value),
            Value::Method(impls, _) => {
                impls.borrow_mut().insert(sc, value);
            }
            _ => panic!("expected method"),
        }
    }

    pub fn primitive(name: &'static str, arity: usize, f: fn(&[Value]) -> Value) -> Self {
        Value::Primitive(Primitive { name, arity, f }, vec![])
    }

    pub fn is_unit(&self) -> bool {
        match self {
            Value::Boxed(bx) => bx.borrow().is_unit(),
            Value::Unit => true,
            _ => false,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Value::Boxed(bx) => bx.borrow().as_bool(),
            Value::Bool(x) => *x,
            _ => panic!("expected boolean"),
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Value::Boxed(bx) => bx.borrow().as_int(),
            Value::I64(x) => *x,
            _ => panic!("expected int"),
        }
    }

    pub fn as_char(&self) -> char {
        match self {
            Value::Boxed(bx) => bx.borrow().as_char(),
            Value::Char(ch) => *ch,
            _ => panic!("expected char"),
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Value::Boxed(bx) => bx.borrow().as_float(),
            Value::F64(x) => *x,
            _ => panic!("expected float"),
        }
    }

    pub fn as_string(&self) -> Rc<str> {
        match self {
            Value::Boxed(bx) => bx.borrow().as_string(),
            Value::String(s) => s.clone(),
            _ => panic!("expected string"),
        }
    }

    pub fn apply(&self, args: Vec<Value>) -> Value {
        match self {
            Value::Boxed(bx) => bx.borrow().apply(args),

            Value::Constructor(ty, tag, fields) => {
                let mut fields = fields.clone();
                fields.extend(args);
                Value::Constructor(ty.clone(), tag.clone(), fields)
            }

            Value::Primitive(prim, gathered_args) => {
                let mut gathered_args = gathered_args.clone();
                gathered_args.extend(args);

                assert!(gathered_args.len() <= prim.arity);

                if gathered_args.len() < prim.arity {
                    return Value::Primitive(prim.clone(), gathered_args);
                }

                (prim.f)(&gathered_args)
            }

            Value::Closure(Closure {
                alts,
                env,
                ctx,
                gathered_args,
            }) => {
                let mut gathered_args = gathered_args.clone();
                gathered_args.extend(args);

                'next_alternative: for Alt(pats, body) in alts.iter() {
                    if gathered_args.len() < pats.len() {
                        return Value::Closure(Closure {
                            alts: alts.clone(),
                            env: env.clone(),
                            ctx: ctx.clone(),
                            gathered_args,
                        });
                    }

                    let mut local_env = env.clone();
                    for (pat, arg) in pats.iter().zip(gathered_args.iter()) {
                        if !ctx.match_pat(pat, arg, &mut local_env) {
                            continue 'next_alternative;
                        }
                    }
                    let mut result = ctx.eval_expr(body, &local_env);

                    if gathered_args.len() > pats.len() {
                        result = result.apply(gathered_args[pats.len()..].to_vec());
                    }

                    return result;
                }
                panic!("no pattern matched")
            }

            Value::Method(impls, gathered_args) => {
                // although this late binding / dynamic dispatch feels super inefficient,
                // it's needed for generic code, where the concrete type is not known during checking/annotation
                let mut gathered_args = gathered_args.clone();
                gathered_args.extend(args);

                for (sc, value) in impls.borrow().iter() {
                    if dispatch::scheme_matches(sc, &gathered_args, None) {
                        return value.apply(gathered_args);
                    }
                }

                Value::Method(impls.clone(), gathered_args)
            }
            _ => panic!("non-callable value {}", self),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Uninitialized => write!(f, "<uninitialized>"),
            Value::Boxed(bx) => write!(f, "@{:?}", bx.borrow()),
            Value::Unit => write!(f, "()"),
            Value::Bool(x) => write!(f, "{}", x),
            Value::I64(x) => write!(f, "{}", x),
            Value::Char(ch) => write!(f, "{}", ch),
            Value::F64(x) => write!(f, "{}", x),
            Value::String(s) => write!(f, "{:?}", s),
            Value::Closure(c) => write!(f, "{:?}", c),
            Value::Constructor(_, tag, fields) => {
                write!(f, "{}", tag)?;
                if !fields.is_empty() {
                    write!(
                        f,
                        " {}",
                        fields
                            .iter()
                            .map(|v| format!("{}", v))
                            .collect::<Vec<_>>()
                            .join(" ")
                    )?;
                }
                Ok(())
            }
            Value::Method(_, args) => write!(
                f,
                "<method [{}]>",
                args.iter()
                    .map(|v| format!("{}", v))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Value::Primitive(p, _) => write!(f, "{:?}", p),
        }
    }
}

/// this is currently super expensive to clone... todo: optimize
#[derive(Clone)]
pub struct Closure {
    pub alts: Rc<Vec<Alt>>,
    pub env: Env,
    pub ctx: Context,
    pub gathered_args: Vec<Value>,
}

impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<closure [{}]>",
            self.gathered_args
                .iter()
                .map(|v| format!("{}", v))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[derive(Clone)]
pub struct Primitive {
    pub name: &'static str,
    pub arity: usize,
    pub f: fn(&[Value]) -> Value,
}

impl std::fmt::Debug for Primitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<primitive {}>", self.name)
    }
}
