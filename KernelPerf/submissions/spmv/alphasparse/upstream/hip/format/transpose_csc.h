#pragma once

#include "alphasparse.h"
#include "alphasparse_create_csc.h"
#include <memory.h>
#include <stdlib.h>
#include <vector>

template<typename T, typename U>
alphasparseStatus_t
transpose_csc(alphasparseSpMatDescr_t& A)
{
  // Compute the transpose of the matrix
  int num_row = A->rows;
  int num_nonzeros = A->nnz;
  U* Avalues = (U*)malloc(sizeof(U) * A->nnz);
  int* Acols_offset = (int*)malloc(sizeof(int) * (A->cols + 1));
  int* Arow_indx = (int*)malloc(sizeof(int) * A->nnz);
  hipMemcpy(Avalues, A->val_data, A->nnz * sizeof(U), hipMemcpyDeviceToHost);
  hipMemcpy(Acols_offset,
             A->col_data,
             (A->cols + 1) * sizeof(int),
             hipMemcpyDeviceToHost);
  hipMemcpy(
    Arow_indx, A->row_data, A->nnz * sizeof(int), hipMemcpyDeviceToHost);
  // Count the non-zero elements contained in each row
  std::vector<int> row_counts(num_row, 0);
  for (int i = 0; i < num_nonzeros; i++) {
    row_counts[Arow_indx[i]]++;
  }
  // Compute the start offset of each transposed column in values and col_ids
  int* col_offset = (int*)alpha_memalign((uint64_t)(num_row + 1) * sizeof(int),
                                         DEFAULT_ALIGNMENT);
  col_offset[0] = 0;
  for (int i = 1; i <= num_row; i++) {
    col_offset[i] = col_offset[i - 1] + row_counts[i - 1];
  }
  U* values =
    (U*)alpha_memalign((uint64_t)A->nnz * sizeof(U), DEFAULT_ALIGNMENT);
  int* row_indx =
    (int*)alpha_memalign((uint64_t)A->nnz * sizeof(int), DEFAULT_ALIGNMENT);

  // Place the non-zero elements into the transposed matrix by column index
  for (int i = 0; i < num_row; i++) {
    for (int j = Acols_offset[i]; j < Acols_offset[i + 1]; j++) {
      int row_id = Arow_indx[j];
      int dest_index = col_offset[row_id];
      row_indx[dest_index] = i;
      values[dest_index] = Avalues[j];
      col_offset[row_id]++;
    }
  }

  // Restore the start offset of each row in values and col_ids
  for (int i = num_row; i >= 1; i--) {
    col_offset[i] = col_offset[i - 1];
  }
  col_offset[0] = 0;

  hipMemcpy(A->val_data, values, A->nnz * sizeof(U), hipMemcpyHostToDevice);
  hipMemcpy(A->col_data,
             col_offset,
             (num_row + 1) * sizeof(int),
             hipMemcpyHostToDevice);
  hipMemcpy(
    A->row_data, row_indx, A->nnz * sizeof(int), hipMemcpyHostToDevice);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
