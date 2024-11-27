use crate::assumptions::Assump;
use crate::specific_inference::{Alt, BindGroup, Expl, Expr, Impl, Literal, Pat, Program};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub type Env = HashMap<String, Value>;

#[derive(Clone, Debug)]
pub enum Value {
    Uninitialized,
    Boxed(Rc<RefCell<Self>>),

    Int(i64),
    Char(char),
    Float(f64),
    String(Rc<str>),

    Closure(Closure),

    Constructor(Rc<str>, Vec<Value>),
}

impl Value {
    pub fn boxed(self) -> Self {
        Self::Boxed(Rc::new(self.into()))
    }

    pub fn update(&self, value: Value) {
        match self {
            Value::Boxed(bx) => *bx.borrow_mut() = value,
            _ => panic!("immutable value"),
        }
    }

    pub fn constructor(tag: impl Into<Rc<str>>) -> Self {
        Value::Constructor(tag.into(), vec![])
    }

    pub fn as_constructor(&self) -> (Rc<str>, Vec<Value>) {
        match self {
            Value::Boxed(bx) => bx.borrow().as_constructor(),
            Value::Constructor(tag, fields) => (tag.clone(), fields.clone()),
            _ => panic!("expected constructor"),
        }
    }

    pub fn as_int(&self) -> i64 {
        match self {
            Value::Boxed(bx) => bx.borrow().as_int(),
            Value::Int(x) => *x,
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
            Value::Float(x) => *x,
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

            Value::Constructor(tag, fields) => {
                let mut fields = fields.clone();
                fields.extend(args);
                Value::Constructor(tag.clone(), fields)
            }

            Value::Closure(Closure {
                alts,
                env,
                gathered_args,
            }) => {
                let mut gathered_args = gathered_args.clone();
                gathered_args.extend(args);

                'next_alternative: for Alt(pats, body) in alts.iter() {
                    if gathered_args.len() < pats.len() {
                        return Value::Closure(Closure {
                            alts: alts.clone(),
                            env: env.clone(),
                            gathered_args,
                        });
                    }

                    let mut local_env = env.clone();
                    for (pat, arg) in pats.iter().zip(gathered_args.iter()) {
                        if !match_pat(pat, arg, &mut local_env) {
                            continue 'next_alternative;
                        }
                    }
                    let mut result = eval_expr(body, &local_env);

                    if gathered_args.len() > pats.len() {
                        result = result.apply(gathered_args[pats.len()..].to_vec());
                    }

                    return result;
                }
                panic!("no pattern matched")
            }
            _ => panic!("non-callable value"),
        }
    }
}

pub fn exec_program(Program(bgs): &Program, env: &mut Env) {
    for bg in bgs {
        exec_bindgroup(bg, env)
    }
}

fn exec_bindgroup(BindGroup(expls, implss): &BindGroup, env: &mut Env) {
    for Expl(id, _, alts) in expls {
        let value = eval_alts(alts, env);
        env.insert(id.clone(), value);
    }

    for impls in implss {
        for Impl(id, _) in impls {
            env.insert(id.clone(), Value::boxed(Value::Uninitialized));
        }

        for Impl(id, alts) in impls {
            let value = eval_alts(alts, env);
            env[id].update(value)
        }
    }
}

fn eval_alts(alts: &Rc<Vec<Alt>>, env: &Env) -> Value {
    if alts[0].0.is_empty() {
        return eval_expr(&alts[0].1, env);
    }

    Value::Closure(Closure {
        alts: alts.clone(),
        env: env.clone(),
        gathered_args: vec![],
    })
}

pub fn eval_expr(expr: &Expr, env: &Env) -> Value {
    match expr {
        Expr::Var(x) => env.get(x).unwrap_or_else(|| panic!("unbound {x}")).clone(),
        Expr::Lit(l) => eval_lit(l),
        Expr::Const(_) => unimplemented!(),
        Expr::App(rator, rand) => {
            let rator_ = eval_expr(rator, env);
            let rand_ = eval_expr(rand, env);
            rator_.apply(vec![rand_])
        }
        Expr::Let(bg, body) => {
            let mut local_env = env.clone();
            exec_bindgroup(bg, &mut local_env);
            eval_expr(body, &local_env)
        }
    }
}

fn eval_lit(lit: &Literal) -> Value {
    match lit {
        Literal::Int(x) => Value::Int(*x),
        Literal::Char(ch) => Value::Char(*ch),
        Literal::Rat(x) => Value::Float(*x),
        Literal::Str(s) => Value::String(s.clone()),
    }
}

fn match_pat(pat: &Pat, val: &Value, env: &mut Env) -> bool {
    match pat {
        Pat::PVar(name) => {
            env.insert(name.clone(), val.clone());
            true
        }

        Pat::PWildcard => true,

        Pat::PAs(name, pat) => {
            if match_pat(pat, val, env) {
                env.insert(name.clone(), val.clone());
                true
            } else {
                false
            }
        }

        Pat::PLit(lit) => match_lit(lit, val),

        Pat::PNpk(name, k) => {
            if let Value::Int(x) = val {
                env.insert(name.clone(), Value::Int(*x - k));
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
                if !match_pat(pat, field, env) {
                    return false;
                }
            }

            true
        }
    }
}

fn match_lit(lit: &Literal, val: &Value) -> bool {
    match lit {
        Literal::Int(x) => val.as_int() == *x,
        Literal::Char(ch) => val.as_char() == *ch,
        Literal::Rat(x) => val.as_float() == *x,
        Literal::Str(s) => val.as_string() == *s,
    }
}

/// this is currently super expensive to clone... todo: optimize
#[derive(Clone)]
pub struct Closure {
    pub alts: Rc<Vec<Alt>>,
    pub env: Env,
    pub gathered_args: Vec<Value>,
}

impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<closure {:?}>", self.gathered_args)
    }
}
