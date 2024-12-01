//! Transform the parsed AST into the representation used by the type checker.

use crate::frontend::ast::InfixToken;
use crate::frontend::runner::Runner;
use crate::frontend::{ast, type_inference as si};
use crate::type_checker::assumptions::find;
use crate::type_checker::kinds::Kind;
use crate::type_checker::predicates::Pred;
use crate::type_checker::scheme::Scheme;
use crate::type_checker::{predicates, qualified, scheme, types, Id};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub type TEnv = HashMap<Id, types::Type>;

impl Runner {
    pub fn with_tyenv<T>(&mut self, temporary_env: TEnv, f: impl FnOnce(&mut Self) -> T) -> T {
        let backup = std::mem::replace(&mut self.type_env, temporary_env);
        let result = f(self);
        self.type_env = backup;
        result
    }

    pub fn build_program(&mut self, bgs: Vec<ast::BindGroup>) -> si::Program {
        si::Program(bgs.into_iter().map(|bg| self.build_bindgroup(bg)).collect())
    }

    pub fn build_bindgroup(&mut self, ast::BindGroup(bg): ast::BindGroup) -> si::BindGroup {
        let mut binds = vec![];

        let mut bound_names = HashSet::new();
        for x in bg
            .iter()
            .map(|b| match b {
                ast::Bind::Declaration(decl) => {
                    let items: Box<dyn Iterator<Item = &str>> =
                        Box::new(std::iter::once(decl.0.as_str()));
                    items
                }
                ast::Bind::Implicit(imp) => Box::new(std::iter::once(imp.0.as_str())),
                ast::Bind::Mutual(mu) => Box::new(mu.iter().map(|impl_| impl_.0.as_str())),
            })
            .flatten()
        {
            if bound_names.insert(x) == false {
                panic!("duplicate binding: {}", x);
            }
        }

        for b in bg {
            match b {
                ast::Bind::Declaration(decl) => {
                    self.free_decls.insert(decl.0.clone(), decl);
                }
                ast::Bind::Implicit(_) => binds.push(b),
                ast::Bind::Mutual(_) => binds.push(b),
            };
        }

        let mut expls = vec![];
        let mut impls = vec![];

        for b in binds {
            match b {
                ast::Bind::Declaration(_) => unreachable!(),
                ast::Bind::Implicit(impl_) => match self.free_decls.remove(&impl_.0) {
                    None => impls.push(vec![self.build_impl(impl_)]),
                    Some(decl) => expls.push(self.build_expl(decl, impl_)),
                },
                ast::Bind::Mutual(mut_) => impls.push(
                    mut_.into_iter()
                        .map(|impl_| self.build_impl(impl_))
                        .collect(),
                ),
            }
        }

        si::BindGroup(expls, impls)
    }

    pub fn build_impl(&mut self, ast::Impl(id, alts): ast::Impl) -> si::Impl {
        let alts = self.build_alts(alts);
        si::Impl(id, alts)
    }

    pub fn build_expl(
        &mut self,
        ast::Decl(id, sc): ast::Decl,
        ast::Impl(_, alts): ast::Impl,
    ) -> si::Expl {
        let sc = self.build_scheme(sc);
        let alts = self.build_alts(alts);
        si::Expl(id, sc, alts)
    }

    pub fn build_alts(&mut self, alts: Vec<ast::Alt>) -> Rc<Vec<si::Alt>> {
        Rc::new(alts.into_iter().map(|alt| self.build_alt(alt)).collect())
    }

    pub fn build_alt(&mut self, ast::Alt(pats, expr): ast::Alt) -> si::Alt {
        let pats = pats.into_iter().map(|pat| self.build_pat(pat)).collect();
        let expr = self.build_expr(expr);
        si::Alt(pats, expr)
    }

    pub fn build_pat(&mut self, pat: ast::Pat) -> si::Pat {
        match pat {
            ast::Pat::PVar(id) => si::Pat::PVar(id),
            ast::Pat::PWildcard => si::Pat::PWildcard,
            ast::Pat::PAs(id, pat) => si::Pat::PAs(id, Rc::new(self.build_pat(*pat))),
            ast::Pat::PLit(lit) => si::Pat::PLit(lit),
            ast::Pat::PNpk(id, n) => si::Pat::PNpk(id, n),
            ast::Pat::PCon(id, pats) => {
                let constructor = find(&id, &self.constructors).unwrap().clone();
                let ps = pats.into_iter().map(|p| self.build_pat(p)).collect();
                si::Pat::PCon(constructor, ps)
            }
        }
    }

    pub fn build_expr(&mut self, expr: ast::Expr) -> si::Expr {
        match expr {
            ast::Expr::Infix(tokens) => self.build_expr(self.apply_precedence(tokens)),
            ast::Expr::Var(id) => si::Expr::Var(id),
            ast::Expr::Lit(lit) => si::Expr::Lit(lit),
            ast::Expr::App(e1, e2) => {
                let e1 = Rc::new(self.build_expr(*e1));
                let e2 = Rc::new(self.build_expr(*e2));
                si::Expr::App(e1, e2)
            }
            ast::Expr::Let(bg, e) => {
                let bg = self.build_bindgroup(bg);
                let e = Rc::new(self.build_expr(*e));
                si::Expr::Let(bg, e)
            }
            ast::Expr::Seq(stmts, last) => {
                let stmts = stmts.into_iter().map(|s| self.build_expr(s)).collect();
                let last = self.build_expr(*last);
                si::Expr::Sequence(Rc::new(stmts), Rc::new(last))
            }

            ast::Expr::Lambda(alt) => {
                let alt = self.build_alt(*alt);

                let name = unique_id("λ");

                si::Expr::Let(
                    si::BindGroup(
                        vec![],
                        vec![vec![si::Impl(name.clone(), Rc::new(vec![alt]))]],
                    ),
                    Rc::new(si::Expr::Var(name)),
                )
            }
        }
    }

    pub fn build_type(&mut self, ty: ast::Type) -> types::Type {
        match ty {
            ast::Type::Named(name) => {
                if let Some(ty) = self.type_env.get(&name) {
                    ty.clone()
                } else {
                    panic!("unknown type name: {}", name)
                }
            }
            ast::Type::Apply(t1, t2) => {
                let t1 = self.build_type(*t1);
                let t2 = self.build_type(*t2);
                types::Type::tapp(t1, t2)
            }
        }
    }

    pub fn build_scheme(&mut self, sc: ast::Scheme) -> scheme::Scheme {
        let mut tyenv = self.type_env.clone();
        let (vs, preds) = self.build_typeargs(sc.genvars, &mut tyenv);

        let ty = self.with_tyenv(tyenv, |ctx| ctx.build_type(sc.ty));

        let qt = qualified::Qual(preds, ty);

        Scheme::quantify(&vs, &qt)
    }

    pub fn build_typeargs(
        &mut self,
        genvars: Vec<(Id, Kind, Vec<Id>)>,
        tyenv: &mut TEnv,
    ) -> (Vec<types::Tyvar>, Vec<Pred>) {
        let mut preds = vec![];
        let mut vs = vec![];

        for (name, kind, constraints) in genvars {
            let tv = types::Tyvar(name.clone(), kind);
            vs.push(tv.clone());
            let t = types::Type::TVar(tv);

            for c in constraints {
                let pred = predicates::Pred::IsIn(c, t.clone());
                preds.push(pred);
            }

            tyenv.insert(name, t.clone());
        }

        (vs, preds)
    }

    fn apply_precedence(&self, tokens: Vec<InfixToken>) -> ast::Expr {
        let mut stack = vec![];
        let mut output = vec![];

        // insert function call operators
        let mut tokens_ = vec![];
        for tok in tokens {
            match tok {
                InfixToken::Op(_) => tokens_.push(tok),
                InfixToken::Expr(_) => {
                    if let Some(InfixToken::Expr(_)) = tokens_.last() {
                        tokens_.push(InfixToken::Op("".to_string()));
                    }
                    tokens_.push(tok);
                }
            }
        }

        // shunting yard algorithm
        for token in tokens_ {
            match token {
                InfixToken::Op(op) => {
                    while let Some(top) = stack.last() {
                        if let InfixToken::Op(top_op) = top {
                            let top_pred = self.precedence(top_op);
                            let pred = self.precedence(&op);
                            let la = self.left_associative(&op);
                            if pred <= top_pred && la || pred < top_pred && !la {
                                output.push(stack.pop().unwrap());
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    stack.push(InfixToken::Op(op));
                }

                InfixToken::Expr(expr) => output.push(InfixToken::Expr(expr)),
            }
        }

        while let Some(top) = stack.pop() {
            output.push(top);
        }

        // build expression
        let mut stack = vec![];
        for token in output {
            match token {
                InfixToken::Expr(x) => stack.push(x),

                // function call
                InfixToken::Op(op) if op == "" => {
                    let e2 = stack.pop().unwrap();
                    let e1 = stack.pop().unwrap();
                    stack.push(ast::Expr::app(e1, e2));
                }

                // binary operator
                InfixToken::Op(op) => {
                    let e2 = stack.pop().unwrap();
                    let e1 = stack.pop().unwrap();
                    stack.push(ast::Expr::app(ast::Expr::app(ast::Expr::Var(op), e1), e2));
                }
            }
        }

        assert_eq!(stack.len(), 1);
        stack.pop().unwrap()
    }

    fn precedence(&self, op: &str) -> i32 {
        match op {
            "::" => 8,
            "*" | "/" => 7,
            "+" | "-" => 6,
            "==" | "!=" | "<" | ">" | "<=" | ">=" => 4,
            "&&" => 3,
            "||" => 2,
            "" => 0, // function call
            _ => 1,
        }
    }

    fn left_associative(&self, op: &str) -> bool {
        match op {
            "*" | "/" | "+" | "-" | "==" | "!=" | "<" | ">" | "<=" | ">=" | "&&" | "||" | "" => {
                true
            }
            "::" => false,
            _ => true,
        }
    }
}

fn unique_id(prefix: &str) -> Id {
    format!(
        "{}{}",
        prefix,
        UNIQUE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

const UNIQUE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
