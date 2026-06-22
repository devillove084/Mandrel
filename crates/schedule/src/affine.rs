#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    M,
    N,
    K,
    Sequence,
    HeadDim,
}

/// Tiny affine expression used for early schedule analysis. Coefficients map to
/// `[m, n, k]` axes for matmul domains.
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

    pub const fn axis_m() -> Self {
        Self::new([1, 0, 0], 0)
    }

    pub const fn axis_n() -> Self {
        Self::new([0, 1, 0], 0)
    }

    pub const fn axis_k() -> Self {
        Self::new([0, 0, 1], 0)
    }

    pub fn evaluate(self, indices: [isize; 3]) -> Option<isize> {
        let m = self.coefficients[0].checked_mul(indices[0])?;
        let n = self.coefficients[1].checked_mul(indices[1])?;
        let k = self.coefficients[2].checked_mul(indices[2])?;
        m.checked_add(n)?.checked_add(k)?.checked_add(self.constant)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessMap {
    pub row: AffineExpr,
    pub col: AffineExpr,
}

impl AccessMap {
    pub const fn lhs_matmul() -> Self {
        Self {
            row: AffineExpr::axis_m(),
            col: AffineExpr::axis_k(),
        }
    }

    pub const fn rhs_matmul() -> Self {
        Self {
            row: AffineExpr::axis_k(),
            col: AffineExpr::axis_n(),
        }
    }

    pub const fn out_matmul() -> Self {
        Self {
            row: AffineExpr::axis_m(),
            col: AffineExpr::axis_n(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AccessMap;

    #[test]
    fn maps_matmul_indices() {
        let lhs = AccessMap::lhs_matmul();
        let rhs = AccessMap::rhs_matmul();
        let out = AccessMap::out_matmul();

        assert_eq!(lhs.row.evaluate([2, 3, 4]), Some(2));
        assert_eq!(lhs.col.evaluate([2, 3, 4]), Some(4));
        assert_eq!(rhs.row.evaluate([2, 3, 4]), Some(4));
        assert_eq!(rhs.col.evaluate([2, 3, 4]), Some(3));
        assert_eq!(out.row.evaluate([2, 3, 4]), Some(2));
        assert_eq!(out.col.evaluate([2, 3, 4]), Some(3));
    }
}
