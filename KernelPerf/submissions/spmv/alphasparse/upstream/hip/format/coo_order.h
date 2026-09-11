#pragma once

#include "alphasparse/spmat.h"

#include <stdio.h>
#include <stdlib.h>

template <typename T, typename U>
struct Node {
  T x;
  T y;
  U v;
};

template <typename T, typename U>
static int cmp(const void *a, const void *b) {
  if (((Node<T, U> *)a)->x ==
      ((Node<T, U> *)b)->x)  // If the x values of two structs are equal, sort by their y values in ascending order
    return ((Node<T, U> *)a)->y > ((Node<T, U> *)b)->y;
  else
    return ((Node<T, U> *)a)->x > ((Node<T, U> *)b)->x;  // Otherwise sort by x in ascending order
}

template <typename T, typename U>
alphasparseStatus_t coo_order(T nnz, T *row_indx, T *col_indx, U *values) {
  struct Node<T, U> *nodes = {};
  nodes = (Node<T, U> *)malloc(sizeof(Node<T, U>) * nnz);
  for (int i = 0; i < nnz; i++) {
    nodes[i].x = row_indx[i];
    nodes[i].y = col_indx[i];
    nodes[i].v = values[i];
  }

  qsort(nodes, nnz, sizeof(nodes[0]), cmp<T,U>);

  for (int i = 0; i < nnz; i++) {
    row_indx[i] = nodes[i].x;
    col_indx[i] = nodes[i].y;
    values[i] = nodes[i].v;
  }
  free(nodes);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
