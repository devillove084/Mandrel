/// CuTe-inspired 2D shape used by schedule-level layout and thread-map analysis.
///
/// This is intentionally small and `no_std` friendly. It models the part we need before
/// backend code generation: logical tile extents and how threads cover tile coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape2D {
    pub rows: usize,
    pub cols: usize,
}

impl Shape2D {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub const fn element_count(self) -> usize {
        self.rows * self.cols
    }

    pub const fn is_nonzero(self) -> bool {
        self.rows != 0 && self.cols != 0
    }
}

/// Linear strides in element units for a 2D logical layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stride2D {
    pub row: usize,
    pub col: usize,
}

impl Stride2D {
    pub const fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    pub const fn row_major(cols: usize) -> Self {
        Self::new(cols, 1)
    }

    pub const fn column_major(rows: usize) -> Self {
        Self::new(1, rows)
    }
}

/// A small CuTe-like layout: `(row, col) -> linear element offset`.
///
/// Unlike `core::Layout`, this keeps explicit strides so schedule/codegen can reason about
/// row-major, transposed, padded, and future ggml-strided tensors with one representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout2D {
    pub shape: Shape2D,
    pub stride: Stride2D,
}

impl Layout2D {
    pub const fn new(shape: Shape2D, stride: Stride2D) -> Self {
        Self { shape, stride }
    }

    pub const fn row_major(rows: usize, cols: usize) -> Self {
        Self::new(Shape2D::new(rows, cols), Stride2D::row_major(cols))
    }

    pub const fn column_major(rows: usize, cols: usize) -> Self {
        Self::new(Shape2D::new(rows, cols), Stride2D::column_major(rows))
    }

    pub fn element_offset(self, row: usize, col: usize) -> Option<usize> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return None;
        }

        row.checked_mul(self.stride.row)?
            .checked_add(col.checked_mul(self.stride.col)?)
    }

    pub fn span_elements(self) -> Option<usize> {
        if !self.shape.is_nonzero() {
            return Some(0);
        }

        self.element_offset(self.shape.rows - 1, self.shape.cols - 1)?
            .checked_add(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    Global,
    Local,
    Register,
}

/// Minimal copy descriptor, analogous to a tiny CuTe copy atom.
///
/// It is schedule metadata only: the Vortex backend decides whether this becomes plain global
/// loads, `__local_mem` staging, vectorized loads, or future target-specific intrinsics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyAtom2D {
    pub src: MemoryScope,
    pub dst: MemoryScope,
    pub tile: Shape2D,
}

impl CopyAtom2D {
    pub const fn new(src: MemoryScope, dst: MemoryScope, tile: Shape2D) -> Self {
        Self { src, dst, tile }
    }

    pub const fn global_to_local(tile: Shape2D) -> Self {
        Self::new(MemoryScope::Global, MemoryScope::Local, tile)
    }

    pub const fn local_to_register(tile: Shape2D) -> Self {
        Self::new(MemoryScope::Local, MemoryScope::Register, tile)
    }
}

/// 2D thread-block shape. `x` maps to columns and `y` maps to rows for GPU-style kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadBlock2D {
    pub x: usize,
    pub y: usize,
}

impl ThreadBlock2D {
    pub const fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    pub const fn thread_count(self) -> usize {
        self.x * self.y
    }
}

/// CuTe-inspired mapping from workgroup threads to logical tile coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadTileMap2D {
    pub tile: Shape2D,
    pub block: ThreadBlock2D,
    pub values_per_thread: Shape2D,
}

impl ThreadTileMap2D {
    pub const fn new(tile: Shape2D, block: ThreadBlock2D, values_per_thread: Shape2D) -> Self {
        Self {
            tile,
            block,
            values_per_thread,
        }
    }

    pub const fn direct_one_value_per_thread(tile: Shape2D) -> Self {
        Self::new(
            tile,
            ThreadBlock2D::new(tile.cols, tile.rows),
            Shape2D::new(1, 1),
        )
    }

    pub const fn is_one_value_per_thread(self) -> bool {
        self.values_per_thread.rows == 1 && self.values_per_thread.cols == 1
    }

    pub fn logical_coord(
        self,
        thread_x: usize,
        thread_y: usize,
        value_row: usize,
        value_col: usize,
    ) -> Option<[usize; 2]> {
        if thread_x >= self.block.x
            || thread_y >= self.block.y
            || value_row >= self.values_per_thread.rows
            || value_col >= self.values_per_thread.cols
        {
            return None;
        }

        let row = thread_y
            .checked_mul(self.values_per_thread.rows)?
            .checked_add(value_row)?;
        let col = thread_x
            .checked_mul(self.values_per_thread.cols)?
            .checked_add(value_col)?;

        if row < self.tile.rows && col < self.tile.cols {
            Some([row, col])
        } else {
            None
        }
    }

    pub fn covers_tile(self) -> bool {
        if !self.values_per_thread.is_nonzero() {
            return false;
        }

        let Some(rows) = self.block.y.checked_mul(self.values_per_thread.rows) else {
            return false;
        };
        let Some(cols) = self.block.x.checked_mul(self.values_per_thread.cols) else {
            return false;
        };

        rows >= self.tile.rows && cols >= self.tile.cols
    }
}

#[cfg(test)]
mod tests {
    use super::{CopyAtom2D, Layout2D, MemoryScope, Shape2D, ThreadBlock2D, ThreadTileMap2D};

    #[test]
    fn row_major_layout_maps_coordinates_to_offsets() {
        let layout = Layout2D::row_major(2, 3);

        assert_eq!(layout.element_offset(0, 0), Some(0));
        assert_eq!(layout.element_offset(1, 2), Some(5));
        assert_eq!(layout.element_offset(2, 0), None);
        assert_eq!(layout.span_elements(), Some(6));
    }

    #[test]
    fn column_major_layout_maps_coordinates_to_offsets() {
        let layout = Layout2D::column_major(2, 3);

        assert_eq!(layout.element_offset(0, 0), Some(0));
        assert_eq!(layout.element_offset(1, 0), Some(1));
        assert_eq!(layout.element_offset(0, 2), Some(4));
        assert_eq!(layout.element_offset(1, 2), Some(5));
    }

    #[test]
    fn direct_thread_map_matches_gpu_x_columns_y_rows() {
        let mapping = ThreadTileMap2D::direct_one_value_per_thread(Shape2D::new(4, 4));

        assert_eq!(mapping.block, ThreadBlock2D::new(4, 4));
        assert!(mapping.is_one_value_per_thread());
        assert!(mapping.covers_tile());
        assert_eq!(mapping.logical_coord(3, 2, 0, 0), Some([2, 3]));
        assert_eq!(mapping.logical_coord(4, 0, 0, 0), None);
    }

    #[test]
    fn multi_value_thread_map_covers_register_tile() {
        let mapping = ThreadTileMap2D::new(
            Shape2D::new(4, 8),
            ThreadBlock2D::new(4, 2),
            Shape2D::new(2, 2),
        );

        assert!(mapping.covers_tile());
        assert_eq!(mapping.logical_coord(1, 1, 1, 0), Some([3, 2]));
        assert_eq!(mapping.logical_coord(3, 1, 1, 1), Some([3, 7]));
    }

    #[test]
    fn copy_atom_records_memory_scope_transition() {
        let atom = CopyAtom2D::global_to_local(Shape2D::new(16, 32));

        assert_eq!(atom.src, MemoryScope::Global);
        assert_eq!(atom.dst, MemoryScope::Local);
        assert_eq!(atom.tile.element_count(), 512);
    }
}
