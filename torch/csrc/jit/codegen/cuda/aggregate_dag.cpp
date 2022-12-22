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

// AggregateExpr::AggregateExpr(const AggregateExpr* src, IrCloner* ir_cloner)
//     : Expr(src, ir_cloner), group_(src->getGroup()) {}


SendRecv::SendRecv(IrBuilderPasskey passkey)
    : Expr(passkey, ExprType::SendRecv) {}

AggregateVal::AggregateVal(
    IrBuilderPasskey passkey, Val* val, Group* group)
    : Val(passkey, val->vtype(), val->dtype()),
    original_val_(val), group_(group) {
      //TODO: add a mapping from original val to AggregateVal.
    }

AggregateDag::AggregateDag():IrContainer(){}

void AggregateDag::build(MultiGroupFusion* fusion) {
  for (auto group: fusion->groups()) {

    // auto expr = IrBuilder::create<AggregateExpr>(this, group);

    for (auto output_val : group->output_vals) {
      auto val = IrBuilder::create<AggregateVal>((IrContainer*)this, output_val, group.get());
      // producer[output_val] = val;
      // expr->addOutput(val);
      // val->setDefinition(expr);
    }

    // for (auto input_val : group->input_vals) {
    //   auto val = IrBuilder::create<AggregateVal>(this, input_val, group);
    //   // consumers[input_val].pushBack(val);
    //   expr->addInput(val);
    //   if (producer.find(input_val) != producer.end()){
    //     //means that input_val is not a global input and 
    //     // is produced by another group
    //     auto src = producer[input_val];
    //     auto sendRecv = IrBuilder::create<SendRecv>(this);
    //     sendRecv->addInput(src);
    //     sendRecv->addOutput(val);
    //     val->setDefinition(sendRecv);
    //   }
    // }
  }
}





} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

