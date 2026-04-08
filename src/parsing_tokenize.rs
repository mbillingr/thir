use chumsky::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub enum RawToken<'a> {
    LowerIdent(&'a str),
    UpperIdent(&'a str),
    Operator(&'a str),
    Int(&'a str),
    Float(&'a str),
    String(&'a str),
    Parenthised(Vec<Token<'a>>),
}

pub type Token<'a> = Spanned<RawToken<'a>>;

pub fn lexer<'src>() -> impl Parser<'src, &'src str, Vec<Token<'src>>, extra::Err<Rich<'src, char>>>
{
    recursive(|token| {
        choice((
            text::ident().map(|s: &str| match s {
                s if s.chars().next().unwrap().is_uppercase() => RawToken::UpperIdent(s),
                s if s.chars().next().unwrap().is_lowercase() => RawToken::LowerIdent(s),
                _ => unreachable!(),
            }),
            text::int(10).map(RawToken::Int),
            operator(),
            token
                .repeated()
                .collect()
                .delimited_by(just('('), just(')'))
                .labelled("token tree")
                .as_context()
                .map(RawToken::Parenthised),
        ))
        .spanned()
        .padded()
    })
    .repeated()
    .collect()
}

fn operator<'src>(
) -> impl Parser<'src, &'src str, RawToken<'src>, extra::Err<Rich<'src, char>>> + Clone {
    one_of("/+-*=<>,;.|&")
        .repeated()
        .at_least(1)
        .to_slice()
        .map(RawToken::Operator)
}
