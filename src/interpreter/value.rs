use crate::frontend::type_inference::Alt;
use crate::interpreter::evaluator::Context;
use crate::interpreter::{dispatch, Env};
use crate::type_checker;
use crate::type_checker::types;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

thread_local! {
    pub static GLOBAL_HASH_FN: RefCell<Option<Value>> = Default::default();
    pub static GLOBAL_EQ_FN: RefCell<Option<Value>> = Default::default();
}

#[derive(Clone, Debug)]
pub enum Value {
    Uninitialized,
    Boxed(Rc<RefCell<Self>>),

    Unit,
    Bool(bool),
    Int(Rc<num::BigInt>),
    Char(char),
    F64(f64),
    String(Rc<str>),

    Dict(Rc<im_rc::HashMap<Value, Value>>),

    Closure(Rc<Closure>),

    Constructor(Rc<types::Type>, Rc<str>, Rc<Vec<Value>>),

    Method(
        Rc<str>,
        usize,
        Rc<RefCell<Vec<(types::Type, Value)>>>,
        Rc<Vec<Value>>,
    ),

    Primitive(Primitive, Rc<Vec<Value>>),
}

impl Value {
    pub fn is_a(&self, ty: &type_checker::types::Type) -> bool {
        use type_checker::types::{Tycon, Type::*, Tyvar};
        match (ty, self) {
            (ty, Value::Boxed(bx)) => bx.borrow().is_a(ty),
            (TVar(Tyvar(tn, _)) | TCon(Tycon(tn, _)), Value::Int(_)) => tn == "Int",
            (TVar(Tyvar(tn, _)) | TCon(Tycon(tn, _)), Value::Char(_)) => tn == "Char",
            (TVar(Tyvar(tn, _)) | TCon(Tycon(tn, _)), Value::F64(_)) => tn == "Double",
            (TVar(Tyvar(tn, _)) | TCon(Tycon(tn, _)), Value::String(_)) => tn == "String",
            (ty, Value::Constructor(tc, _, _)) => types::Type::soft_eq(tc, ty),
            (TGen(_), _) => true,
            (TApp(rc), Value::Dict(_)) => {
                if let TApp(rc2) = &rc.0 {
                    if let TCon(Tycon(tn, _)) = &rc2.0 {
                        tn == "Dict"
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            _ => false, //todo!("{:?} {:?}", ty, self),
        }
    }

    pub fn int(x: impl Into<num::BigInt>) -> Self {
        Value::Int(Rc::new(x.into()))
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

    pub fn dict() -> Self {
        Value::Dict(Default::default())
    }

    pub fn dict_insert(&self, key: Value, value: Value) -> Self {
        match self {
            Value::Boxed(bx) => bx.borrow().dict_insert(key, value),
            Value::Dict(dict) => {
                let mut dict = (**dict).clone();
                dict.insert(key, value);
                Value::Dict(Rc::new(dict))
            }
            _ => panic!("expected dict"),
        }
    }

    pub fn as_dict(&self) -> Option<Rc<im_rc::HashMap<Value, Value>>> {
        match self {
            Value::Boxed(bx) => bx.borrow().as_dict(),
            Value::Dict(dict) => Some(dict.clone()),
            _ => None,
        }
    }

    pub fn constructor(ty: type_checker::types::Type, tag: impl Into<Rc<str>>) -> Self {
        Value::Constructor(ty.into(), tag.into(), Rc::new(vec![]))
    }

    pub fn applied_constructor(
        ty: type_checker::types::Type,
        tag: impl Into<Rc<str>>,
        args: Vec<Self>,
    ) -> Self {
        Value::Constructor(ty.into(), tag.into(), Rc::new(args))
    }

    pub fn as_constructor(&self) -> Option<(Rc<str>, Rc<Vec<Value>>)> {
        match self {
            Value::Boxed(bx) => bx.borrow().as_constructor(),
            Value::Constructor(_, tag, fields) => Some((tag.clone(), fields.clone())),
            _ => None,
        }
    }

    pub fn make_list(items: impl DoubleEndedIterator<Item = Value>) -> Self {
        Self::make_list_reverse(items.rev())
    }

    pub fn make_list_reverse(items: impl Iterator<Item = Value>) -> Self {
        let strlist = types::Type::list(types::Type::t_string());
        let mut result = Value::constructor(strlist.clone(), "Nil");
        for item in items {
            result = Value::applied_constructor(strlist.clone(), "::", vec![item, result]);
        }
        result
    }

    pub fn method(name: impl Into<Rc<str>>, dispatch_arg: usize) -> Self {
        Value::Method(
            name.into(),
            dispatch_arg,
            Rc::new(RefCell::new(Vec::new())),
            Rc::new(vec![]),
        )
    }

    pub fn add_impl(&self, ty: types::Type, value: Value) {
        let (_, _, impls, _) = self.as_method().expect("expected method");
        impls.borrow_mut().push((ty, value));
    }

    pub fn as_method(
        &self,
    ) -> Option<(
        Rc<str>,
        usize,
        Rc<RefCell<Vec<(types::Type, Value)>>>,
        Rc<Vec<Value>>,
    )> {
        match self {
            Value::Boxed(bx) => bx.borrow().as_method(),
            Value::Method(name, dispatch_arg, impls, args) => {
                Some((name.clone(), *dispatch_arg, impls.clone(), args.clone()))
            }
            _ => None,
        }
    }

    pub fn primitive(name: &'static str, arity: usize, f: fn(&[Value]) -> Value) -> Self {
        Value::Primitive(Primitive { name, arity, f }, Rc::new(vec![]))
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

    pub fn as_int(&self) -> Rc<num::BigInt> {
        match self {
            Value::Boxed(bx) => bx.borrow().as_int(),
            Value::Int(x) => x.clone(),
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
            _ => panic!("expected string {:?}", self),
        }
    }

    pub fn apply(&self, args: Vec<Value>) -> Value {
        match self {
            Value::Boxed(bx) => bx.borrow().apply(args),

            Value::Constructor(ty, tag, fields) => {
                let mut fields = (**fields).clone();
                fields.extend(args);
                Value::Constructor(ty.clone(), tag.clone(), Rc::new(fields))
            }

            Value::Primitive(prim, gathered_args) => {
                let mut gathered_args = (**gathered_args).clone();
                gathered_args.extend(args);

                assert!(gathered_args.len() <= prim.arity);

                if gathered_args.len() < prim.arity {
                    return Value::Primitive(prim.clone(), Rc::new(gathered_args));
                }

                (prim.f)(&gathered_args)
            }

            Value::Closure(cls) => {
                let Closure {
                    alts,
                    env,
                    ctx,
                    gathered_args,
                } = &**cls;
                let mut gathered_args = gathered_args.clone();
                gathered_args.extend(args);

                'next_alternative: for Alt(pats, body) in alts.iter() {
                    if gathered_args.len() < pats.len() {
                        return Value::Closure(Rc::new(Closure {
                            alts: alts.clone(),
                            env: env.clone(),
                            ctx: ctx.clone(),
                            gathered_args,
                        }));
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

            Value::Method(name, dispatch_arg, impls, gathered_args) => {
                // although this late binding / dynamic dispatch feels super inefficient,
                // it's needed for generic code, where the concrete type is not known during checking/annotation
                let mut gathered_args = (**gathered_args).clone();
                gathered_args.extend(args);

                if gathered_args.len() > *dispatch_arg {
                    for (ty, value) in impls.borrow().iter() {
                        if dispatch::type_matches(ty, &gathered_args[*dispatch_arg]) {
                            return value.apply(gathered_args);
                        }
                    }
                }

                Value::Method(
                    name.clone(),
                    *dispatch_arg,
                    impls.clone(),
                    Rc::new(gathered_args),
                )
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
            Value::Int(x) => write!(f, "{}", x),
            Value::Char(ch) => write!(f, "{}", ch),
            Value::F64(x) => write!(f, "{}", x),
            Value::String(s) => write!(f, "{:?}", s),
            Value::Dict(dict) => write!(
                f,
                "{}",
                dict.iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
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
            Value::Method(name, _, _, args) => write!(
                f,
                "<method {} [{}]>",
                name,
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

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Uninitialized => {}
            Value::Boxed(rc) => Rc::as_ptr(rc).hash(state),
            Value::Unit => ().hash(state),
            Value::Bool(b) => b.hash(state),
            Value::Int(x) => x.hash(state),
            Value::Char(ch) => ch.hash(state),
            Value::F64(_) => unimplemented!(),
            Value::String(s) => s.hash(state),
            Value::Dict(_) => unimplemented!(),
            Value::Closure(cls) => Rc::as_ptr(cls).hash(state),
            Value::Method(_, _, _, _) => unimplemented!(),
            Value::Primitive(p, args) => {
                p.f.hash(state);
                Rc::as_ptr(args).hash(state);
            }
            Value::Constructor(_, _, _) => {
                GLOBAL_HASH_FN.with_borrow(|fn_| {
                    let fn_ = fn_.as_ref().unwrap();
                    fn_.apply(vec![self.clone()]).hash(state);
                });
            }
        }
    }
}

impl Eq for Value {}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Uninitialized, Value::Uninitialized) => true,
            (Value::Boxed(a), Value::Boxed(b)) => Rc::ptr_eq(a, b),
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::F64(a), Value::F64(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Dict(a), Value::Dict(b)) => a == b,
            (Value::Closure(a), Value::Closure(b)) => Rc::ptr_eq(a, b),
            (Value::Method(a, b, c, d), Value::Method(e, f, g, h)) => {
                a == e && b == f && Rc::ptr_eq(c, g) && h == d
            }
            (Value::Primitive(a, b), Value::Primitive(c, d)) => a.f == c.f && Rc::ptr_eq(b, d),
            (Value::Constructor(_, _, _), Value::Constructor(_, _, _)) => {
                GLOBAL_EQ_FN.with_borrow(|fn_| {
                    let fn_ = fn_.as_ref().unwrap();
                    fn_.apply(vec![self.clone(), other.clone()]).as_bool()
                })
            }
            _ => false,
        }
    }
}
