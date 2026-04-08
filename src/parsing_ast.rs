use crate::ast::{ClassName, Constraint, TExpr};
use crate::parsing_tokenize::{RawToken, Token};
use chumsky::input::MappedInput;
use chumsky::pratt::*;
use chumsky::prelude::*;
use ustr::ustr;

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
