use std::ops::Range;

use crate::error::NeuralNetworkError;

#[derive(Eq, PartialEq, Debug, Clone, Default, PartialOrd, Ord, Hash)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Layout {
    pub fn new(shape: Vec<usize>) -> Self {
        let strides: Vec<usize> = Self::compute_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn with_offset(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let dimension: usize = shape.len();
        let mut strides: Vec<usize> = vec![1, dimension];
        for i in (0..dimension.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());

        self.offset
            + indices
                .iter()
                .zip(&self.strides)
                .map(|(index, stride): (&usize, &usize)| index * stride)
                .sum::<usize>()
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_contiguous(&self) -> bool {
        let expected_strides: Vec<usize> = Self::compute_strides(&self.shape);
        self.strides == expected_strides
    }

    pub fn permute(&self, dimensions: &[usize]) -> Result<Self, NeuralNetworkError> {
        if dimensions.len() != self.ndim() {
            return Err(NeuralNetworkError::InvalidDimension {
                got: dimensions.len(),
                max_dimension: self.ndim(),
            });
        }
        let mut seen: Vec<bool> = vec![false; self.ndim()];
        for &dimension in dimensions {
            if dimension >= self.ndim() || seen[dimension] {
                return Err(NeuralNetworkError::InvalidDimension {
                    got: dimensions.len(),
                    max_dimension: self.ndim(),
                });
            }
            seen[dimension] = true;
        }
        let new_shape: Vec<usize> = dimensions
            .iter()
            .map(|&dimension: &usize| self.shape[dimension])
            .collect();
        let new_strides: Vec<usize> = dimensions
            .iter()
            .map(|&dimension: &usize| self.strides[dimension])
            .collect();
        Ok(Layout {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, NeuralNetworkError> {
        if !self.is_contiguous() {
            return Err(NeuralNetworkError::NotContiguous);
        }
        let old_size: usize = self.num_elements();
        let new_size: usize = new_shape.iter().product();
        if old_size != new_size {
            return Err(NeuralNetworkError::IncompatibleShape {
                shape_given: new_shape,
                tensor_shape: self.shape().to_vec(),
            });
        }
        Ok(Layout::new(new_shape))
    }

    pub fn slice(&self, dimension: usize, range: Range<usize>) -> Result<Self, NeuralNetworkError> {
        if dimension >= self.ndim() {
            return Err(NeuralNetworkError::InvalidDimension {
                got: dimension,
                max_dimension: self.ndim(),
            });
        }
        if range.end > self.shape[dimension] {
            return Err(NeuralNetworkError::OutOfBounds);
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape[dimension] = range.len();
        let new_offset: usize = self.offset + range.start * self.strides[dimension];
        Ok(Layout {
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }
}
