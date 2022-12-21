#include <torch/csrc/jit/codegen/cuda/aggregate_dag.h>
#include <torch/csrc/jit/codegen/cuda/multidevice_runtime.h>


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

AggregateExpr::AggregateExpr(
    IrBuilderPasskey passkey,
    Group* group)
    : Expr(passkey, ExprType::AggregateExpr), group_(group) {
  for (auto input: group_->input_vals){
    addInput(input);
  }
  for (auto output: group_->output_vals){
    addOutput(output);
  }
}

AggregateExpr::AggregateExpr(const AggregateExpr* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner), group_(src->getGroup()) {}


// void Group::buildGroupedExpr_(){
//     IrContainer* container = getMultiGroupFusion();
//     gexpr_ = IrBuilder::create<AggregateExpr>(container, this);
// }

MultiGroupTv::MultiGroupTv(
    IrBuilderPasskey passkey, TensorView* tv, Group* group)
    : TensorView(passkey, tv->domain(), tv->dtype(), tv->getMemoryType()),
    tv_(tv), group_(group) {
      //TODO: add a mapping from original tv to MultiDevicetv.
    }


} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

