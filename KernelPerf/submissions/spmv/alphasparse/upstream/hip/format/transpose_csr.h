#pragma once

#include "alphasparse.h"
#include "alphasparse_create_csr.h"
#include <memory.h>
#include <stdlib.h>
#include <vector>

template<typename T, typename U>
alphasparseStatus_t
transpose_csr(alphasparseSpMatDescr_t& A)
{
  // Compute the transpose of the matrix
  int num_col = A->cols;
  int num_nonzeros = A->nnz;
  U* Avalues = (U*)malloc(sizeof(U) * A->nnz);
  int* Arows_offset = (int*)malloc(sizeof(int) * (A->rows + 1));
  int* Acol_indx = (int*)malloc(sizeof(int) * A->nnz);
  hipMemcpy(Avalues, A->val_data, A->nnz * sizeof(U), hipMemcpyDeviceToHost);
  hipMemcpy(Arows_offset,
             A->row_data,
             (A->rows + 1) * sizeof(int),
             hipMemcpyDeviceToHost);
  hipMemcpy(
    Acol_indx, A->col_data, A->nnz * sizeof(int), hipMemcpyDeviceToHost);
  // Count the non-zero elements contained in each column
  std::vector<int> column_counts(num_col, 0);
  for (int i = 0; i < num_nonzeros; i++) {
    column_counts[Acol_indx[i]]++;
  }
  // Compute the start offset of each transposed row in values and row_ids
  int* rows_offset = (int*)alpha_memalign((uint64_t)(num_col + 1) * sizeof(int),
                                          DEFAULT_ALIGNMENT);
  rows_offset[0] = 0;
  for (int i = 1; i <= num_col; i++) {
    rows_offset[i] = rows_offset[i - 1] + column_counts[i - 1];
  }
  U* values =
    (U*)alpha_memalign((uint64_t)A->nnz * sizeof(U), DEFAULT_ALIGNMENT);
  int* col_indx =
    (int*)alpha_memalign((uint64_t)A->nnz * sizeof(int), DEFAULT_ALIGNMENT);

  // Place the non-zero elements into the transposed matrix by column index
  for (int i = 0; i < num_col; i++) {
    for (int j = Arows_offset[i]; j < Arows_offset[i + 1]; j++) {
      int col_id = Acol_indx[j];
      int dest_index = rows_offset[col_id];
      col_indx[dest_index] = i;
      values[dest_index] = Avalues[j];
      rows_offset[col_id]++;
    }
  }

  // Restore the start offset of each row in values and row_ids
  for (int i = num_col; i >= 1; i--) {
    rows_offset[i] = rows_offset[i - 1];
  }
  rows_offset[0] = 0;
  hipMemcpy(A->val_data, values, A->nnz * sizeof(U), hipMemcpyHostToDevice);
  hipMemcpy(A->row_data,
             rows_offset,
             (num_col + 1) * sizeof(int),
             hipMemcpyHostToDevice);
  hipMemcpy(
    A->col_data, col_indx, A->nnz * sizeof(int), hipMemcpyHostToDevice);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
