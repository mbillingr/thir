use crate::ast::Literal;
use crate::frontend::type_inference::{Alt, BindGroup, Expl, Expr, Impl, Pat, Program};
use crate::interpreter::value::Closure;
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
    pub fn exec_program(&self, Program(bgs): &Program, env: &mut Env) {
        for bg in bgs {
            self.exec_bindgroup(bg, env)
        }
    }

    fn exec_bindgroup(&self, BindGroup(expls, implss): &BindGroup, env: &mut Env) {
        for Expl(id, _, alts) in expls {
            env.insert(id.clone(), Value::boxed(Value::Uninitialized));
            let value = self.eval_alts(alts, env);
            env[id].update(value)
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
            Expr::Var(x) => {
                let val = env
                    .get(x)
                    .unwrap_or_else(|| panic!("unbound {x}"))
                    .clone()
                    .resolve();

                // I guess this can be considered static dispatch
                if let Value::Method(impls, args) = &val {
                    assert!(args.is_empty());
                    if let Some(t) = self.ti.get_annotation(expr) {
                        for (sc, value) in impls.borrow().iter() {
                            if dispatch::scheme_matches_type(sc, &t) {
                                return value.clone();
                            }
                        }
                        println!("WARNING: no method matched for {:?}", t);
                    }
                }
                val
            }
            Expr::Lit(l) => self.eval_lit(l),
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
            Expr::Sequence(stmts, expr) => {
                for stmt in stmts.iter() {
                    self.eval_expr(stmt, env);
                }
                self.eval_expr(expr, env)
            }
        }
    }

    fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Unit => Value::boxed(Value::Unit),
            Literal::Int(x) => Value::I64(*x),
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
            Literal::Unit => val.is_unit(),
            Literal::Int(x) => val.as_int() == *x,
            Literal::Char(ch) => val.as_char() == *ch,
            Literal::Rat(x) => val.as_float() == *x,
            Literal::Str(s) => val.as_string() == *s,
        }
    }
}
