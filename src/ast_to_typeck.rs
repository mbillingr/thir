use crate::assumptions::find;
use crate::kinds::Kind;
use crate::lists::List;
use crate::predicates::Pred;
use crate::{
    ast, predicates, qualified, scheme, specific_inference as si, types, GlobalContext, Id,
};
use std::collections::HashMap;
use std::rc::Rc;

pub type TEnv = HashMap<Id, types::Type>;

impl GlobalContext {
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

        for b in bg {
            match b {
                ast::Bind::Declaration(decl) => {
                    self.free_decls.insert(decl.0.clone(), decl);
                }
                ast::Bind::Implicit(_) => binds.push(b),
                ast::Bind::Mutual(_) => binds.push(b),
            }
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

    pub fn build_alts(&mut self, alts: Vec<ast::Alt>) -> Vec<si::Alt> {
        alts.into_iter().map(|alt| self.build_alt(alt)).collect()
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
        let (kinds, preds) = self.build_typeargs(sc.genvars, &mut tyenv);

        let ty = self.with_tyenv(tyenv, |ctx| ctx.build_type(sc.ty));

        let qual_ty = qualified::Qual(preds, ty);
        scheme::Scheme::Forall(kinds, qual_ty)
    }

    pub fn build_typeargs(
        &mut self,
        genvars: Vec<(Id, Kind, Vec<Id>)>,
        tyenv: &mut TEnv,
    ) -> (List<Kind>, Vec<Pred>) {
        let mut kinds = List::Nil;
        let mut idx = 0;
        let mut preds = vec![];

        for (name, kind, constraints) in genvars {
            kinds = kinds.cons(kind);
            tyenv.insert(name, types::Type::TGen(idx));

            for c in constraints {
                let pred = predicates::Pred::IsIn(c, types::Type::TGen(idx));
                preds.push(pred);
            }

            idx += 1;
        }

        (kinds, preds)
    }
}
