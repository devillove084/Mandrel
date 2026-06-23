#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Query,
    Key,
    HeadDim,
}

/// Tiny affine expression used for early schedule analysis. Coefficients map to
/// `[query, key, head_dim]` axes for attention-like domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineExpr {
    pub coefficients: [isize; 3],
    pub constant: isize,
}

impl AffineExpr {
    pub const fn new(coefficients: [isize; 3], constant: isize) -> Self {
        Self {
            coefficients,
            constant,
        }
    }

    pub const fn axis_query() -> Self {
        Self::new([1, 0, 0], 0)
    }

    pub const fn axis_key() -> Self {
        Self::new([0, 1, 0], 0)
    }

    pub const fn axis_head_dim() -> Self {
        Self::new([0, 0, 1], 0)
    }

    pub fn evaluate(self, indices: [isize; 3]) -> Option<isize> {
        let query = self.coefficients[0].checked_mul(indices[0])?;
        let key = self.coefficients[1].checked_mul(indices[1])?;
        let head_dim = self.coefficients[2].checked_mul(indices[2])?;
        query
            .checked_add(key)?
            .checked_add(head_dim)?
            .checked_add(self.constant)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessMap {
    pub row: AffineExpr,
    pub col: AffineExpr,
}

impl AccessMap {
    pub const fn query_dense() -> Self {
        Self {
            row: AffineExpr::axis_query(),
            col: AffineExpr::axis_head_dim(),
        }
    }

    pub const fn key_dense() -> Self {
        Self {
            row: AffineExpr::axis_key(),
            col: AffineExpr::axis_head_dim(),
        }
    }

    pub const fn score_tile() -> Self {
        Self {
            row: AffineExpr::axis_query(),
            col: AffineExpr::axis_key(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AccessMap;

    #[test]
    fn maps_attention_indices() {
        let query = AccessMap::query_dense();
        let key = AccessMap::key_dense();
        let score = AccessMap::score_tile();

        assert_eq!(query.row.evaluate([2, 3, 4]), Some(2));
        assert_eq!(query.col.evaluate([2, 3, 4]), Some(4));
        assert_eq!(key.row.evaluate([2, 3, 4]), Some(3));
        assert_eq!(key.col.evaluate([2, 3, 4]), Some(4));
        assert_eq!(score.row.evaluate([2, 3, 4]), Some(2));
        assert_eq!(score.col.evaluate([2, 3, 4]), Some(3));
    }
}
