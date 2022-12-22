#pragma once
// #include <torch/csrc/jit/codegen/cuda/disjoint_set.h>
// #include <torch/csrc/jit/codegen/cuda/evaluator_common.h>
// #include <torch/csrc/jit/codegen/cuda/executor.h>
// #include <torch/csrc/jit/codegen/cuda/fusion.h>
// #include <torch/csrc/jit/codegen/cuda/fusion_segmenter.h>
// #include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
// #include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>

// #include <c10/core/DeviceType.h>
// #include <c10/cuda/CUDAFunctions.h>
// #include <torch/csrc/distributed/c10d/TCPStore.hpp>
// #include <torch/csrc/distributed/c10d/ProcessGroupBuilder.hpp>

#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_container.h>


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class Group;
class MultiGroupFusion;


//! Adding a Val:
//! Right now adding a Val is quite involved. Val's can be defined in ir.h or in
//! their own header file. The following is what is currently needed to add a
//! new Val:
//!
//! 1) Definition inheriting from Val
//!     - Members must be private or protected
//!     - Accessor functions for members
//!     - Must call Val constructor, Val constructor registers with fusion
//!     - Implementation of bool sameAs(...)
//!     - Must implement a "cloning" constructor, ex.
//!        Int::Int(const Int* src, IrCloner* ir_cloner)
//! 2) dispatch.h/.cpp must be updated to include dispatch of the new Val
//! 3) Default mutator function should be added to mutator.cpp
//! 4a) Printing functions should be added to ir_iostream.h/.cpp
//! 4b) Graphviz generation must be added to ir_graphviz.h/.cpp
//! 5) An enum value must be added to ValType in type.h
//! 6) A string entry must be added in val_type_string_map
//!
class TORCH_CUDA_CU_API AggregateVal : public Val {
public:

  AggregateVal(IrBuilderPasskey passkey, Val* val, Group* group);

  const Val* getOriginalVal() const{
    return original_val_;
  }

  const Group* getGroup() const{
    return group_;
  }

  bool is_ready=false;

private:
  Val* original_val_;
  Group* group_;
};





//! 1) Definition inheriting from Expr.
//!      - Members must be private or protected -- why ?
//!      - Accessor functions for members
//!      - Constructors need to register with the Fusion after inputs/outputs
//!         are defined
//!      - Implementation of bool sameAs(...)
//!  2) dispatch.h/.cpp must be updated to include dispatch of the new Val
//!  3) Default mutator function should be added to mutator.h/.cpp
//!  4) Printing functions should be added to ir_iostream.h/.cpp
//!  5) Lower case convenience functions should be added to arith.h/.cpp (If
//!     user facing)
//!  6) An enum value must be added to ExprType in type.h
//!  7) A string entry must be added in expr_type_string_map
//!  8) Entry added to ir_graphviz .cpp/.h


class TORCH_CUDA_CU_API AggregateExpr : public Expr {
public:

  AggregateExpr(IrBuilderPasskey, Group* group);

//   AggregateExpr(const AggregateExpr* src, IrCloner* ir_cloner);

  Group* getGroup() const{
    return group_;
  }

  void addInput(Val* input) {
    Expr::addInput(input);
  }

  void addOutput(Val* input) {
    Expr::addInput(input);
  }

private:
  Group* group_;
};



class TORCH_CUDA_CU_API SendRecv : public Expr {
public:

  SendRecv(IrBuilderPasskey);

  void addInput(Val* input) {
    Expr::addInput(input);
  }

  void addOutput(Val* input) {
    Expr::addInput(input);
  }


};


class TORCH_CUDA_CU_API AggregateDag : public IrContainer {
public:

  AggregateDag();

  void build(MultiGroupFusion* fusion);
private:
  friend MultiGroupFusion;
//   using GroupPtr = std::shared_ptr<Group>;
//   using GroupPtrVector = VectorOfUniqueEntries<GroupPtr>;

  std::unordered_map<Val*, AggregateVal*> producer;

//   std::unordered_map<Val*, GroupPtrVector> consumers;
};


} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

