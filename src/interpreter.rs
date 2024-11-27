use crate::assumptions::Assump;
use crate::qualified::Qual;
use crate::scheme::Scheme;
use crate::specific_inference::{Alt, BindGroup, Expl, Expr, Impl, Literal, Pat, Program};
use crate::types::{Tycon, Type, Tyvar};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone)]
pub struct Context {}

pub type Env = HashMap<String, Value>;

#[derive(Clone, Debug)]
pub enum Value {
    Uninitialized,
    Boxed(Rc<RefCell<Self>>),

    I64(i64),
    Char(char),
    F64(f64),
    String(Rc<str>),

    Closure(Closure),

    Constructor(Rc<str>, Rc<str>, Vec<Value>),

    Method(Rc<RefCell<HashMap<Scheme, Value>>>, Vec<Value>),
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

    pub fn as_constructor(&self) -> (Rc<str>, Vec<Value>) {
        match self {
            Value::Boxed(bx) => bx.borrow().as_constructor(),
            Value::Constructor(_, tag, fields) => (tag.clone(), fields.clone()),
            _ => panic!("expected constructor"),
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
                let mut gathered_args = gathered_args.clone();
                gathered_args.extend(args);

                // todo: this late binding / dynamic dispatch feels super inefficient

                for (sc, value) in impls.borrow().iter() {
                    if scheme_matches(sc, &gathered_args) {
                        return value.apply(gathered_args);
                    }
                }

                Value::Method(impls.clone(), gathered_args)
            }

            _ => panic!("non-callable value"),
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Context {}
    }
    pub fn exec_program(&self, Program(bgs): &Program, env: &mut Env) {
        for bg in bgs {
            self.exec_bindgroup(bg, env)
        }
    }

    fn exec_bindgroup(&self, BindGroup(expls, implss): &BindGroup, env: &mut Env) {
        for Expl(id, _, alts) in expls {
            let value = self.eval_alts(alts, env);
            env.insert(id.clone(), value);
        }

        for impls in implss {
            for Impl(id, _) in impls {
                env.insert(id.clone(), Value::boxed(Value::Uninitialized));
            }

            for Impl(id, alts) in impls {
                let value = self.eval_alts(alts, env);
                env[id].update(value)
            }
        }
    }

    pub fn eval_alts(&self, alts: &Rc<Vec<Alt>>, env: &Env) -> Value {
        if alts[0].0.is_empty() {
            return self.eval_expr(&alts[0].1, env);
        }

        Value::Closure(Closure {
            alts: alts.clone(),
            env: env.clone(),
            ctx: self.clone(),
            gathered_args: vec![],
        })
    }

    pub fn eval_expr(&self, expr: &Expr, env: &Env) -> Value {
        match expr {
            Expr::Var(x) => env
                .get(x)
                .unwrap_or_else(|| panic!("unbound {x}"))
                .clone()
                .resolve(),
            Expr::Lit(l) => self.eval_lit(l),
            Expr::Const(_) => unimplemented!(),
            Expr::App(rator, rand) => {
                let rator_ = self.eval_expr(rator, env);
                let rand_ = self.eval_expr(rand, env);
                rator_.apply(vec![rand_])
            }
            Expr::Let(bg, body) => {
                let mut local_env = env.clone();
                self.exec_bindgroup(bg, &mut local_env);
                self.eval_expr(body, &local_env)
            }
        }
    }

    fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(x) => Value::I64(*x),
            Literal::Char(ch) => Value::Char(*ch),
            Literal::Rat(x) => Value::F64(*x),
            Literal::Str(s) => Value::String(s.clone()),
        }
    }

    fn match_pat(&self, pat: &Pat, val: &Value, env: &mut Env) -> bool {
        match pat {
            Pat::PVar(name) => {
                env.insert(name.clone(), val.clone());
                true
            }

            Pat::PWildcard => true,

            Pat::PAs(name, pat) => {
                if self.match_pat(pat, val, env) {
                    env.insert(name.clone(), val.clone());
                    true
                } else {
                    false
                }
            }

            Pat::PLit(lit) => self.match_lit(lit, val),

            Pat::PNpk(name, k) => {
                if let Value::I64(x) = val {
                    env.insert(name.clone(), Value::I64(*x - k));
                    true
                } else {
                    false
                }
            }

            Pat::PCon(Assump { i, .. }, pats) => {
                let (tag, fields) = val.as_constructor();
                if &*tag != i {
                    return false;
                }

                for (pat, field) in pats.iter().zip(fields.iter()) {
                    if !self.match_pat(pat, field, env) {
                        return false;
                    }
                }

                true
            }
        }
    }

    fn match_lit(&self, lit: &Literal, val: &Value) -> bool {
        match lit {
            Literal::Int(x) => val.as_int() == *x,
            Literal::Char(ch) => val.as_char() == *ch,
            Literal::Rat(x) => val.as_float() == *x,
            Literal::Str(s) => val.as_string() == *s,
        }
    }
}

fn scheme_matches(Scheme::Forall(_, Qual(_, ty)): &Scheme, args: &[Value]) -> bool {
    let param_tys = ty.fn_arg_types();
    if param_tys.len() != args.len() {
        return false;
    }
    for (t, v) in param_tys.into_iter().zip(args.iter()) {
        match t {
            Type::TVar(Tyvar(name, _)) => {
                if !v.is_a(name) {
                    return false;
                }
            }
            Type::TCon(Tycon(name, _)) => {
                if !v.is_a(name) {
                    return false;
                }
            }
            Type::TApp(_) => todo!(),
            Type::TGen(_) => {}
            Type::Unknown => unreachable!(),
        }
    }
    true
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

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Uninitialized => write!(f, "<uninitialized>"),
            Value::Boxed(bx) => write!(f, "@{:?}", bx.borrow()),
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
        }
    }
}
