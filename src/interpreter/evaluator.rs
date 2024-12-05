use crate::ast::Literal;
use crate::frontend::type_inference::{Alt, BindGroup, Expl, Expr, Impl, Pat, Program};
use crate::interpreter::value::{Closure, Return};
use crate::interpreter::{dispatch, Env, Value};
use crate::type_checker::assumptions::Assump;
use crate::type_checker::type_inference::TI;
use std::rc::Rc;

#[derive(Clone)]
pub struct Context {
    ti: Rc<TI>,
}

impl Context {
    pub fn new(ti: TI) -> Self {
        Context { ti: Rc::new(ti) }
    }
    pub fn exec_program(&mut self, Program(bgs): &Program, env: &mut Env) {
        for bg in bgs {
            self.exec_bindgroup(bg, env)
        }
    }

    fn exec_bindgroup(&mut self, BindGroup(expls, implss): &BindGroup, env: &mut Env) {
        for Expl(id, _, alts) in expls.iter() {
            env.insert(id.clone(), Value::transparent_boxed(Value::Uninitialized));
            let value = self.eval_alts(alts, env);
            env[id].update(value)
        }

        for impls in implss.iter() {
            for Impl(id, _) in impls {
                env.insert(id.clone(), Value::transparent_boxed(Value::Uninitialized));
            }

            for Impl(id, alts) in impls {
                let value = self.eval_alts(alts, env);
                env[id].update(value)
            }
        }
    }

    pub fn eval_alts(&mut self, alts: &Rc<Vec<Alt>>, env: &Env) -> Value {
        if alts[0].0.is_empty() {
            return self.eval_expr(&alts[0].1, env);
        }

        Value::Closure(Rc::new(Closure {
            alts: alts.clone(),
            env: env.clone(),
            ctx: self.clone(),
            gathered_args: vec![],
        }))
    }

    pub fn eval_expr(&mut self, expr: &Expr, env: &Env) -> Value {
        let mut expr = expr.clone();
        let mut env = env.clone();
        loop {
            match &expr {
                Expr::Var(x) => {
                    let val = env
                        .get(x)
                        .unwrap_or_else(|| panic!("unbound {x}"))
                        .clone()
                        .resolve();

                    // I guess this can be considered static dispatch
                    if let Value::Method(_, dispatch_arg, impls, args) = &val {
                        // if it has args, the method is already in the process of being dynamically dispatched
                        if args.is_empty() {
                            if let Some(t) = self.ti.get_annotation(&expr) {
                                let (argtys, _) = t.fn_types();
                                let t = argtys[*dispatch_arg];
                                for (ty, value) in impls.borrow().iter() {
                                    if dispatch::type_matches_type(ty, t) {
                                        return value.clone();
                                    }
                                }
                            }
                        }
                    }
                    return val;
                }
                Expr::Lit(l) => return self.eval_lit(l),
                Expr::App(rator, rand) => {
                    let rator_ = self.eval_expr(rator, &env);
                    let rand_ = self.eval_expr(rand, &env);
                    match rator_.apply(vec![rand_]) {
                        Return::Result(val) => return val,
                        Return::TailCall(ctx, expr_, env_) => {
                            *self = ctx;
                            expr = expr_;
                            env = env_;
                            continue;
                        }
                    }
                }
                Expr::Let(bg, body) => {
                    let mut local_env = env.clone();
                    self.exec_bindgroup(bg, &mut local_env);
                    expr = (**body).clone();
                    env = local_env;
                    continue;
                }
                Expr::Sequence(stmts, expr_) => {
                    for stmt in stmts.iter() {
                        self.eval_expr(stmt, &env);
                    }
                    expr = (**expr_).clone();
                    continue;
                }
                Expr::If(cond, t, f) => {
                    if self.eval_expr(cond, &env).as_bool() {
                        expr = (**t).clone();
                    } else {
                        expr = (**f).clone();
                    }
                    continue;
                }
                Expr::While(cond, body) => {
                    while self.eval_expr(cond, &env).as_bool() {
                        self.eval_expr(body, &env);
                    }
                    return Value::Unit;
                }
            }
        }
    }

    fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Unit => Value::transparent_boxed(Value::Unit),
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Int(x) => Value::int(*x),
            Literal::Char(ch) => Value::Char(*ch),
            Literal::Rat(x) => Value::F64(*x),
            Literal::Str(s) => Value::String(s.clone()),
        }
    }

    pub fn match_pat(&self, pat: &Pat, val: &Value, env: &mut Env) -> bool {
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

            Pat::PCon(Assump { i, .. }, pats) => {
                let (tag, fields) = val.as_constructor().expect("expected constructor");
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
            Literal::Unit => val.is_unit(),
            Literal::Bool(b) => val.as_bool() == *b,
            Literal::Int(x) => *val.as_int() == (*x).into(),
            Literal::Char(ch) => val.as_char() == *ch,
            Literal::Rat(x) => val.as_float() == *x,
            Literal::Str(s) => val.as_string() == *s,
        }
    }
}
