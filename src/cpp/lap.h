#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>
#include <iostream>

#ifdef __GNUC__
#define always_inline __attribute__((always_inline)) inline
#define restrict __restrict__
#elif _WIN32
#define always_inline __forceinline
#define restrict __restrict
#else
#define always_inline inline
#define restrict
#endif

template <typename idx, typename cost>
void augment(idx freerow, int nr, int nc, const cost *restrict assign_cost,
             idx *restrict rowsol, idx *restrict colsol, cost *restrict v,
             bool verbose)
{
  idx endofpath;
  if (verbose) {
    printf("lapjv: AUGMENT SOLUTION row [%lld / %d]\n", freerow, nr);
  }

  auto d = std::unique_ptr<cost[]>(new cost[nc]);  // 'cost-distance' in augmenting path calculation.
  auto pred = std::unique_ptr<idx[]>(new idx[nc]);   // row-predecessor of column in augmenting/alternating path.
  auto collist = std::unique_ptr<idx[]>(new idx[nc]);  // list of columns to be scanned in various ways.

  // Dijkstra shortest path algorithm.
  // runs until unassigned column added to shortest path tree.
  #if _OPENMP >= 201307
  #pragma omp simd
  #endif
  for (idx j = 0; j < nc; j++) {
    d[j] = assign_cost[freerow * nc + j] - v[j];
    pred[j] = freerow;
    collist[j] = nc - j - 1;  // init column list.
  }

  idx low = 0; // columns in 0..low-1 are ready, now none.
  idx up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
               // columns in up..dim-1 are to be considered later to find new minimum,
               // at this stage the list simply contains all columns
  bool unassigned_found = false;
  // initialized in the first iteration: low == up == 0
  idx last = 0;
  cost min = 0;
  do {
    if (up == low) {        // no more columns to be scanned for current minimum.
      last = low - 1;
      // scan columns for up..dim-1 to find all indices for which new minimum occurs.
      // store these indices between low..up-1 (increasing up).
      min = d[collist[up++]];
      for (idx k = up; k < nc; k++) {
        idx j = collist[k];
        cost h = d[j];
        if (h <= min) {
          if (h < min) {   // new minimum.
            up = low;      // restart list at index low.
            min = h;
          }
          // new index with same minimum, put on undex up, and extend list.
          collist[k] = collist[up];
          collist[up++] = j;
        }
      }

      // check if any of the minimum columns happens to be unassigned.
      // if so, we have an augmenting path right away.
      for (idx k = low; k < up; k++) {
        if (colsol[collist[k]] < 0) {
          endofpath = collist[k];
          unassigned_found = true;
          break;
        }
      }
    }

    if (min == INFINITY){
      throw "cost matrix is infeasible";
    }

    if (!unassigned_found) {
      // update 'distances' between freerow and all unscanned columns, via next scanned column.
      idx j1 = collist[low];
      low++;
      idx i = colsol[j1];
      const cost *local_cost = &assign_cost[i * nc];
      cost h = local_cost[j1] - v[j1] - min;
      for (idx k = up; k < nc; k++) {
        idx j = collist[k];
        cost v2 = local_cost[j] - v[j] - h;
        if (v2 < d[j]) {
          pred[j] = i;
          if (v2 == min) {  // new column found at same minimum value
            if (colsol[j] < 0) {
              // if unassigned, shortest augmenting path is complete.
              endofpath = j;
              unassigned_found = true;
              break;
            } else {  // else add to list to be scanned right away.
              collist[k] = collist[up];
              collist[up++] = j;
            }
          }
          d[j] = v2;
        }
      }
    }
  } while (!unassigned_found);

  // update column prices.
  #if _OPENMP >= 201307
  #pragma omp simd
  #endif
  for (idx k = 0; k <= last; k++) {
    idx j1 = collist[k];
    v[j1] = v[j1] + d[j1] - min;
  }

  // reset row and column assignments along the alternating path.
  {
    idx i;
    do {
      i = pred[endofpath];
      colsol[endofpath] = i;
      idx j1 = endofpath;
      endofpath = rowsol[i];
      rowsol[i] = j1;
    } while (i != freerow);
  }

  // End of the current augment step
  if (verbose) {
    std::cout << "v:  ";
    for (idx i = 0; i < nc; i++){
      std::cout << v[i] << " ";
    }
    std::cout << std::endl;
  }

  if (verbose) {
    std::cout << "rowsol: ";
    for (idx i = 0; i < nr; i++){
      std::cout << rowsol[i] << " ";
    }
    std::cout << std::endl;
  }

  if (verbose) {
    std::cout << "colsol: ";
    for (idx i = 0; i < nc; i++){
      std::cout << colsol[i] << " ";
    }
    std::cout << std::endl;
  }

  if (verbose) {
    std::cout << "End of this augmentation step." << std::endl;
  }
}

/// @brief Jonker-Volgenant algorithm.
/// @param dim in problem size
/// @param assign_cost in cost matrix
/// @param verbose in indicates whether to report the progress to stdout
/// @param rowsol out column assigned to row in solution / size dim
/// @param colsol out row assigned to column in solution / size dim
/// @param u out dual variables, row reduction numbers / size dim
/// @param v out dual variables, column reduction numbers / size dim
/// @return achieved minimum assignment cost
template <typename idx, typename cost>
void lap(int nr, int nc, const cost *restrict assign_cost,
         idx *restrict rowsol, idx *restrict colsol, cost *restrict v,
         bool verbose) {
  // Initialization
  #if _OPENMP >= 201307
  #pragma omp simd
  #endif

  for (idx i = 0; i < nr; i++){
    rowsol[i] = -1; // col4row
  }
  for (idx i = 0; i < nc; i++){
    colsol[i] = -1; // row4col
  }

  for (idx i = 0; i < nc; i++){
    v[i] = 0;
  }

  if (verbose) {
    std::cout << "v:  ";
    for (idx i = 0; i < nc; i++){
      std::cout << v[i] << " ";
    }
    std::cout << std::endl;
  }

  if (verbose) {
    std::cout << "rowsol: ";
    for (idx i = 0; i < nr; i++){
      std::cout << rowsol[i] << " ";
    }
    std::cout << std::endl;
  }

  if (verbose) {
    std::cout << "colsol: ";
    for (idx i = 0; i < nc; i++){
      std::cout << colsol[i] << " ";
    }
    std::cout << std::endl;
  }

  // AUGMENT SOLUTION for each free row.
  for (idx freerow = 0; freerow < nr; freerow++) {

    try {
      augment(freerow, nr, nc, assign_cost, rowsol, colsol, v, verbose);
    }
    catch (char const* e){
      throw;
    }

    if (verbose) {
      std::cout << "v:  ";
      for (idx i = 0; i < nc; i++){
        std::cout << v[i] << " ";
      }
      std::cout << std::endl;
    }

    if (verbose) {
      std::cout << "rowsol: ";
      for (idx i = 0; i < nr; i++){
        std::cout << rowsol[i] << " ";
      }
      std::cout << std::endl;
    }

    if (verbose) {
      std::cout << "colsol: ";
      for (idx i = 0; i < nc; i++){
        std::cout << colsol[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  if (verbose) {
    printf("lapjv: AUGMENT SOLUTION finished\n");
  }

}
