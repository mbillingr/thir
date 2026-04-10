use crate::ast::{
    ClassName, Constraint, ConstructorName, Expr, TExpr, TopLevel, TypeDef, TypeName, TypeVar,
    VariantDef,
};
use crate::parsing_tokenize::{RawToken, Token};
use crate::specific_inference::Literal;
use chumsky::input::MappedInput;
use chumsky::pratt::*;
use chumsky::prelude::*;
use ustr::ustr;

pub fn toplevel<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<TopLevel>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    choice((
        type_def().map(TopLevel::TypeDef),
        expr().map(TopLevel::Expr),
    ))
    .spanned()
}

pub fn type_def<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<TypeDef>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    just(RawToken::LowerIdent("data"))
        .ignore_then(type_name().then(typevar().repeated().collect()))
        .then(
            just(RawToken::LowerIdent("where"))
                .ignore_then(
                    constraint()
                        .separated_by(just(RawToken::Operator(",")))
                        .collect(),
                )
                .or_not()
                .map(Option::unwrap_or_default),
        )
        .then(
            just(RawToken::Operator("=")).ignore_then(
                variant_def()
                    .separated_by(just(RawToken::Operator("|")))
                    .collect(),
            ),
        )
        .map(|(((tname, params), constraints), variants)| TypeDef {
            tname,
            params,
            constraints,
            variants,
        })
        .spanned()
}

pub fn variant_def<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<VariantDef>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    constructor_name()
        .spanned()
        .then(type_expr().repeated().collect())
        .map(|(name, fields)| VariantDef { name, fields })
        .spanned()
}

pub fn constraint<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<Constraint>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    class_name()
        .then(type_expr().repeated().at_least(1).collect())
        .map(|(cls, tys)| Constraint { cls, tys })
        .spanned()
}

pub fn class_name<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<ClassName>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    select_ref! {RawToken::UpperIdent(cls) => ClassName(ustr(cls))}.spanned()
}

pub fn type_name<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<TypeName>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    select_ref! {RawToken::UpperIdent(cls) => TypeName(ustr(cls))}.spanned()
}

pub fn constructor_name<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    ConstructorName,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    select_ref! {RawToken::UpperIdent(cls) => ConstructorName(ustr(cls))}
}

pub fn typevar<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<TypeVar>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    select_ref! {RawToken::LowerIdent(cls) => TypeVar(ustr(cls))}.spanned()
}

pub fn type_expr<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<TExpr>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    recursive(|texpr| {
        choice((
            select_ref! {
                RawToken::LowerIdent(s) => TExpr::Sym(ustr(s)),
                RawToken::UpperIdent(s) => TExpr::Sym(ustr(s)),
            }
            .spanned(),
            texpr.nested_in(
                select_ref! {RawToken::Parenthised(ts) = e => ts.split_spanned(e.span())},
            ),
        ))
        .pratt((
            infix(
                left(10),
                just(RawToken::Operator("->")).spanned(),
                |f, tc: Token, a, e| {
                    TExpr::App(
                        Box::new(
                            TExpr::App(
                                Box::new(TExpr::Sym(ustr("->")).with_span(tc.span)),
                                Box::new(f),
                            )
                            .with_span(tc.span),
                        ),
                        Box::new(a),
                    )
                    .with_span(e.span())
                },
            ),
            infix(left(1), empty(), |tc, _, t, e| {
                TExpr::App(Box::new(tc), Box::new(t)).with_span(e.span())
            }),
        ))
    })
    .labelled("type expression")
}

pub fn expr<'tokens, 'src: 'tokens>() -> impl Parser<
    'tokens,
    MappedInput<'tokens, RawToken<'src>, SimpleSpan, &'tokens [Token<'src>]>,
    Spanned<Expr>,
    extra::Err<Rich<'tokens, RawToken<'src>>>,
> {
    recursive(|expr| {
        choice((
            select_ref! {
                RawToken::Int(s) => Expr::Literal(Literal::Int(s.parse().unwrap())),
                RawToken::Float(s) => Expr::Literal(Literal::Rat(s.parse().unwrap())),
                RawToken::Operator(s) => Expr::Var(ustr(s)),
                RawToken::LowerIdent(s) => Expr::Var(ustr(s)),
                RawToken::UpperIdent(s) => Expr::Var(ustr(s)),
            },
            expr.repeated()
                .collect()
                .nested_in(select_ref! {
                    RawToken::Parenthised(ts) = e => ts.split_spanned(e.span())
                })
                .map(|xs| Expr::App(xs)),
        ))
        .spanned()
    })
}
