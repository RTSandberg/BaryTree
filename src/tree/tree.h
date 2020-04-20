#ifndef H_TREE_FUNCTIONS_H
#define H_TREE_FUNCTIONS_H

#include "../particles/struct_particles.h"
#include "../run_params/struct_run_params.h"

#include "struct_tree_linked_list_node.h"
#include "struct_tree.h"


void Tree_Sources_Construct(struct Tree **tree_addr, struct Particles *sources, struct RunParams *run_params);

void Tree_Targets_Construct(struct Tree **tree_addr, struct Particles *targets, struct RunParams *run_params);

void Tree_Alloc(struct Tree **tree_addr, int length);

void Tree_Free(struct Tree **tree_addr);

void Tree_Fill(struct Tree *tree, struct TreeLinkedListNode *p);

void Tree_Print(struct Tree *tree);


#endif /* H_TREEFUNCTIONS_H */
