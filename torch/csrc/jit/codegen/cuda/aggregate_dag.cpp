#include <torch/csrc/jit/codegen/cuda/aggregate_dag.h>
#include <torch/csrc/jit/codegen/cuda/multigroup_fusion.h>


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

AggregateExpr::AggregateExpr(
    IrBuilderPasskey passkey, GroupPtr group)
    : Expr(passkey, ExprType::AggregateExpr), group_(group) {
}

AggregateExpr::AggregateExpr(const AggregateExpr* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner), group_(src->group_) {}

Expr* AggregateExpr::shallowCopy() const {
  auto result = IrBuilder::create<AggregateExpr>(this->group_);
  result->copyPredicatesFrom(this);
  return result;
}

bool AggregateExpr::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<AggregateExpr>()) {
    return false;
  }
  const auto other_op = other->as<AggregateExpr>();

  return group_== other_op->getGroup();
}


SendRecv::SendRecv(IrBuilderPasskey passkey, AggregateVal* out, AggregateVal* in)
    : Expr(passkey, ExprType::SendRecv), out_{out}, in_{in} {
  addOutput(out);
  addInput(in);
}

SendRecv::SendRecv(const SendRecv* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner), out_{src->out_}, in_{src->in_} {}

Expr* SendRecv::shallowCopy() const {
  auto result = IrBuilder::create<SendRecv>(this->out_, this->in_);
  result->copyPredicatesFrom(this);
  return result;
}

bool SendRecv::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<SendRecv>()) {
    return false;
  }
  // const auto other_op = other->as<SendRecv>();

  return Expr::sameAs(other);
}


AggregateVal::AggregateVal(
    IrBuilderPasskey passkey, Val* val, GroupPtr group)
    : Val(passkey, ValType::AggregateVal, val->dtype()),
    original_val_(val), group_(group) {
      //TODO: add a mapping from original val to AggregateVal.
    }

AggregateVal::AggregateVal(const AggregateVal* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), original_val_(src->original_val_), group_(src->group_) {}

bool AggregateVal::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<AggregateVal>()) {
    return false;
  }
  const auto other_aggregate_val = other->as<AggregateVal>();
  return original_val_->sameAs(other_aggregate_val->original_val_)
          && group_->unique_id == other_aggregate_val->group_->unique_id;
}


AggregateDag::AggregateDag():Fusion(), IterVisitor(){}

void AggregateDag::build(MultiGroupFusion* fusion) {
  for (auto group: fusion->groups()) {
    IrContainer* container = (IrContainer*)this;
    std::vector<AggregateVal*> inputs;
    std::vector<AggregateVal*> outputs;

    for (auto output_val : group->output_vals) {
      auto val = IrBuilder::create<AggregateVal>(container, output_val, group);
      producer[output_val] = val;
      outputs.push_back(val);
    }

    for (auto input_val : group->input_vals) {
      auto val = IrBuilder::create<AggregateVal>(container, input_val, group);
      inputs.push_back(val);
      consumers.insert({input_val, val});
      if (producer.find(input_val) != producer.end()){
        //means that input_val is not a global input and 
        // is produced by another group
        auto src = producer[input_val];
        auto sendRecv = IrBuilder::create<SendRecv>(container, val, src);
        val->setDefinition(sendRecv);
      } else {
        //add val as global input of the aggregate dag
        addInput(val);
      }
    }

    auto expr = IrBuilder::create<AggregateExpr>(container, group);
    for (auto out: outputs){
      expr->addOutput(out);
      out->setDefinition(expr);
    }
    for (auto in: inputs){
      expr->addInput(in);
    }
  }
  for (auto it: producer){
    if (consumers.count(it.first)==0){
      //add val as global output of the aggregate dag
        addOutput(it.second);
    }
  }
}


} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

