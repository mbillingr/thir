use chumsky::prelude::Spanned;
use ustr::Ustr;

#[derive(Debug)]
struct TypeDef {
    tname: Spanned<Ustr>,
    params: Vec<Spanned<Ustr>>,
    constraints: Vec<Spanned<Constraint>>,
    variants: Vec<Spanned<VariantDef>>,
}

#[derive(Debug)]
struct VariantDef {
    name: Spanned<Ustr>,
    fields: Vec<Spanned<TExpr>>,
}

#[derive(Debug)]
pub struct Constraint {
    pub cls: Spanned<ClassName>,
    pub tys: Vec<Spanned<TExpr>>,
}

#[derive(Debug)]
pub struct ClassName(pub Ustr);

#[derive(Debug)]
pub enum TExpr {
    Sym(Ustr),
    App(Box<Spanned<TExpr>>, Box<Spanned<TExpr>>),
}
