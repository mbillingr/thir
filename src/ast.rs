use chumsky::prelude::Spanned;
use ustr::Ustr;

#[derive(Debug)]
pub struct TypeDef {
    pub tname: Spanned<TypeName>,
    pub params: Vec<Spanned<TypeVar>>,
    pub constraints: Vec<Spanned<Constraint>>,
    pub variants: Vec<Spanned<VariantDef>>,
}

#[derive(Debug)]
pub struct VariantDef {
    pub name: Spanned<ConstructorName>,
    pub fields: Vec<Spanned<TExpr>>,
}

#[derive(Debug)]
pub struct Constraint {
    pub cls: Spanned<ClassName>,
    pub tys: Vec<Spanned<TExpr>>,
}

#[derive(Debug)]
pub struct ClassName(pub Ustr);

#[derive(Debug)]
pub struct TypeName(pub Ustr);

#[derive(Debug)]
pub struct ConstructorName(pub Ustr);

#[derive(Debug)]
pub struct TypeVar(pub Ustr);

#[derive(Debug)]
pub enum TExpr {
    Sym(Ustr),
    App(Box<Spanned<TExpr>>, Box<Spanned<TExpr>>),
}
