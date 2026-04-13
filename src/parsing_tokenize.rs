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
    Braced(Vec<Token<'a>>),

    KeywordData,
    KeywordTrait,
    KeywordWhere,
}

impl std::fmt::Display for RawToken<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RawToken::Parenthised(_) => write!(f, "token tree"),
            _ => write!(f, "{:?}", self),
        }
    }
}

pub type Token<'a> = Spanned<RawToken<'a>>;

pub fn lexer<'src>() -> impl Parser<'src, &'src str, Vec<Token<'src>>, extra::Err<Rich<'src, char>>>
{
    recursive(|token| {
        choice((
            text::ident().map(|s: &str| match s {
                s if s == "data" => RawToken::KeywordData,
                s if s == "trait" => RawToken::KeywordTrait,
                s if s == "where" => RawToken::KeywordWhere,
                s if s.chars().next().unwrap().is_uppercase() => RawToken::UpperIdent(s),
                s if s.chars().next().unwrap().is_lowercase() => RawToken::LowerIdent(s),
                _ => unreachable!(),
            }),
            text::int(10).map(RawToken::Int),
            operator(),
            token
                .clone()
                .repeated()
                .collect()
                .padded()
                .delimited_by(just('('), just(')'))
                .labelled("(token tree)")
                .as_context()
                .map(RawToken::Parenthised),
            token
                .clone()
                .repeated()
                .collect()
                .padded()
                .delimited_by(just('{'), just('}'))
                .labelled("{token tree}")
                .as_context()
                .map(RawToken::Braced),
        ))
        .spanned()
        .padded()
    })
    .repeated()
    .collect()
}

fn operator<'src>(
) -> impl Parser<'src, &'src str, RawToken<'src>, extra::Err<Rich<'src, char>>> + Clone {
    one_of("/+-*=<>:;,.|&")
        .repeated()
        .at_least(1)
        .to_slice()
        .map(RawToken::Operator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_braces0() {
        let tokens = lexer().parse("{}").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].inner, RawToken::Braced(vec![]));

        let tokens = lexer().parse("{   }").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].inner, RawToken::Braced(vec![]));
    }

    #[test]
    fn parse_braces1() {
        let tokens = lexer().parse("{x}").unwrap();

        assert_eq!(tokens.len(), 1);
        let RawToken::Braced(inner) = &tokens[0].inner else {
            panic!("not braced")
        };

        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].inner, RawToken::LowerIdent("x"));
    }
}
