# Preparing Submission

## Presentation

## Cleanup

* [ ] include/batch_merge.cuh
  * [ ] Revision `build_and_sort_batches` -> `randomCudaVector.cuh`.
* [ ] include/batch_sort.cuh  
  * [ ] build_M_batches
  * [ ] sort_batch_cpu -> Maybe not here
* [ ] include/config.h -> Try to distribute or make into a MakeConfig
* [ ] include/cuda_timing.h  -> Resolve macro for activation.
* [ ] include/diag_search.cuh -> merge into big file for mergeLarge stuff
* [ ] include/partition.cuh  -> merge into big file for mergeLarge stuff
* [ ] include/path_merge.cuh  -> merge into big file for mergeLarge stuff
* [ ] include/randomCudaVector.cuh ->  Generalize!
* [ ] include/thrust_merge.cuh -> Necessary? I would inline it to the only call that we have to it, is "ideologicaly" not part of our implementation.
* [ ] include/utils.hpp -> Maybe could be removed too
  
## Makefile

* [ ] Adjust makefile to compile all files in source/entrypoint to executable for build.
* [ ] Adjust for Windows compilation.
* [ ] Ensure compile with OpenMP.
* [ ] Adapt to use nvcc if nvc++ not exist.

## Se c'abbiamo sbatta

* [ ] Point b of proj 2.2.
* [ ] Merge Algorithm.
* [ ] Merge-Sort for array that not are a power of 2.
