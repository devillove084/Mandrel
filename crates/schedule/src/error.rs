use snafu::Snafu;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Snafu)]
pub enum ScheduleError {
    #[snafu(display("attention schedule requires non-empty shape"))]
    EmptyShape,
    #[snafu(display("attention schedule supports i8 element type only"))]
    UnsupportedElementType,
    #[snafu(display("no viable attention schedule candidate matched the target constraints"))]
    NoViableCandidate,
    #[snafu(display("attention schedule shape arithmetic overflowed"))]
    ShapeOverflow,
}
